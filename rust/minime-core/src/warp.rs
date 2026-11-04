//! The pure sub-pixel shift (`cv2.warpAffine` INTER_LINEAR with a translation) four pixels at a
//! time, bit for bit as the scalar `preprocess::shift_with`: the same three fused lerps per
//! pixel, each `fma` emulated exactly by `fsimd`.
//!
//! Per output row the two source rows are float32 slices (a uint8 row is widened first). Where
//! the source column of output column `x` is `x + c` for one constant `c` and both neighbours
//! are inside the image, four columns are lerped per step from contiguous loads; every other
//! pixel (the border strips, rows whose source rows leave the image) takes the scalar path.

use crate::fsimd::{ld, narrow16, st, sub, widen16, D4};
use crate::preprocess::fma;

const LANES: usize = 4;

/// Where the pixels of a row come from: the row as float32 (widened into `buf` if needed).
trait Rows {
    fn width(&self) -> usize;
    fn row<'a>(&'a self, y: usize, buf: &'a mut Vec<f32>) -> &'a [f32];
}

struct F32Rows<'a> {
    src: &'a [f32],
    w: usize,
}

impl Rows for F32Rows<'_> {
    fn width(&self) -> usize {
        self.w
    }

    fn row<'a>(&'a self, y: usize, _buf: &'a mut Vec<f32>) -> &'a [f32] {
        &self.src[y * self.w..(y + 1) * self.w]
    }
}

struct U8Rows<'a> {
    src: &'a [u8],
    w: usize,
}

impl Rows for U8Rows<'_> {
    fn width(&self) -> usize {
        self.w
    }

    fn row<'a>(&'a self, y: usize, buf: &'a mut Vec<f32>) -> &'a [f32] {
        let line = &self.src[y * self.w..(y + 1) * self.w];
        buf.resize(line.len(), 0.0);
        let (blocks, rest) = line.as_chunks::<16>();
        let (out, out_rest) = buf.as_chunks_mut::<16>();
        for (b, o) in blocks.iter().zip(out) {
            for (i, v) in widen16(b).into_iter().enumerate() {
                st(o, 4 * i, v);
            }
        }
        for (o, &v) in out_rest.iter_mut().zip(rest) {
            *o = f32::from(v);
        }
        buf
    }
}

/// The columns of a shift by `dx`: `(x0, ax)` per output column, and the run `[xa, xb)` (a
/// multiple of four long) whose source column is `x + c` with both `x + c` and `x + c + 1`
/// inside the image.
struct Cols {
    ax: Vec<f32>,
    x0: Vec<i64>,
    c: i64,
    xa: usize,
    xb: usize,
}

fn columns(w: usize, dx: f32) -> Cols {
    let (mut ax, mut x0) = (Vec::with_capacity(w), Vec::with_capacity(w));
    for x in 0..w {
        let sx = x as f32 + dx;
        let f = sx.floor();
        x0.push(f as i64);
        ax.push(sx - f);
    }
    let c = x0[w / 2] - (w / 2) as i64;
    let ok = |x: usize| x0[x] - x as i64 == c && x0[x] >= 0 && x0[x] + 1 < w as i64;
    let (mut xa, mut xb) = (0, 0);
    if ok(w / 2) {
        xa = w / 2;
        xb = w / 2 + 1;
        while xa > 0 && ok(xa - 1) {
            xa -= 1;
        }
        while xb < w && ok(xb) {
            xb += 1;
        }
        xb = xa + (xb - xa) / LANES * LANES;
    }
    Cols { ax, x0, c, xa, xb }
}

/// `out(y, x) = src(y + dy, x + dx)` by fused bilinear lerps, zeros outside, one float32 row at
/// a time to `put(y, row)`.
fn warp<R: Rows>(src: &R, h: usize, dy: f32, dx: f32, mut put: impl FnMut(usize, &[f32])) {
    let w = src.width();
    if w == 0 {
        return;
    }
    let cols = columns(w, dx);
    let (mut buf0, mut buf1) = (Vec::new(), Vec::new());
    let mut out = vec![0f32; w];
    let at = |r: Option<&[f32]>, xx: i64| match r {
        Some(r) if xx >= 0 && xx < w as i64 => r[xx as usize],
        _ => 0.0,
    };
    for y in 0..h {
        let sy = y as f32 + dy;
        let y0f = sy.floor();
        let (y0, ay) = (y0f as i64, sy - y0f);
        let inside = |yy: i64| yy >= 0 && yy < h as i64;
        let r0 = inside(y0).then(|| src.row(y0 as usize, &mut buf0));
        let r1 = inside(y0 + 1).then(|| src.row(y0 as usize + 1, &mut buf1));
        let pixel = |x: usize| -> f32 {
            let (x0, ax) = (cols.x0[x], cols.ax[x]);
            let (p00, p01, p10, p11) = (at(r0, x0), at(r0, x0 + 1), at(r1, x0), at(r1, x0 + 1));
            let top = fma(p01 - p00, ax, p00);
            let bot = fma(p11 - p10, ax, p10);
            fma(bot - top, ay, top)
        };
        let vector_cols = match (r0, r1) {
            (Some(r0), Some(r1)) => {
                let ayd = D4::splat(ay);
                for x in (cols.xa..cols.xb).step_by(LANES) {
                    let s = (x as i64 + cols.c) as usize;
                    let (a, b, c, d) = (ld(r0, s), ld(r0, s + 1), ld(r1, s), ld(r1, s + 1));
                    let axd = D4::from_f4(ld(&cols.ax, x));
                    let top = (D4::from_f4(sub(b, a)) * axd + D4::from_f4(a)).to_f4();
                    let bot = (D4::from_f4(sub(d, c)) * axd + D4::from_f4(c)).to_f4();
                    let v = (D4::from_f4(sub(bot, top)) * ayd + D4::from_f4(top)).to_f4();
                    st(&mut out, x, v);
                }
                cols.xa..cols.xb
            }
            _ => 0..0,
        };
        for x in (0..w).filter(|x| !vector_cols.contains(x)) {
            out[x] = pixel(x);
        }
        put(y, &out);
    }
}

/// `preprocess::shift` (uint8; float32 lerps, round half to even).
pub fn shift_u8(src: &[u8], h: usize, w: usize, dy: f32, dx: f32) -> Vec<u8> {
    assert_eq!(src.len(), h * w);
    let mut out = vec![0u8; h * w];
    warp(&U8Rows { src, w }, h, dy, dx, |y, row| {
        let (blocks, rest) = row.as_chunks::<16>();
        let (o, o_rest) = out[y * w..(y + 1) * w].as_chunks_mut::<16>();
        for (b, o) in blocks.iter().zip(o) {
            *o = narrow16(std::array::from_fn(|i| ld(b, 4 * i)));
        }
        for (o, &v) in o_rest.iter_mut().zip(rest) {
            *o = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// `preprocess::shift_f32`.
pub fn shift_f32(src: &[f32], h: usize, w: usize, dy: f32, dx: f32) -> Vec<f32> {
    assert_eq!(src.len(), h * w);
    let mut out = vec![0f32; h * w];
    warp(&F32Rows { src, w }, h, dy, dx, |y, row| {
        out[y * w..(y + 1) * w].copy_from_slice(row)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow16_rounds_half_to_even_and_clamps_like_the_scalar_cast() {
        let vals = [
            0.0f32,
            -0.0,
            0.4,
            0.5,
            0.6,
            1.5,
            2.5,
            3.5,
            254.4,
            254.5,
            255.0,
            255.5,
            300.0,
            -0.4,
            -0.5,
            -7.0,
            1e10,
            -1e10,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            127.5,
            128.5,
            0.49999997,
        ];
        for chunk in vals.chunks(16).chain([&vals[8..24]]) {
            let mut v = [0f32; 16];
            v[..chunk.len()].copy_from_slice(chunk);
            let got = narrow16(std::array::from_fn(|i| ld(&v, 4 * i)));
            let want: Vec<u8> = v
                .iter()
                .map(|x| x.round_ties_even().clamp(0.0, 255.0) as u8)
                .collect();
            assert_eq!(got.to_vec(), want, "{v:?}");
        }
    }
}
