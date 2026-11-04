//! White tophat on sixteen `u8` lanes at a time, exact: erode and dilate with a `(2 r + 1)`
//! square window as separable running minima and maxima, border pixels ignored (identity
//! padding). A window of `k = 2 r + 1` samples is the `min` (or `max`) of two overlapping blocks
//! of `s = 2^floor(log2 k)` samples, and a block of `s` is built in place by `log2 s` doubling
//! steps (`m[x] = min(m[x], m[x + 1])`, then `+ 2`, `+ 4`, ...), so a pass costs `log2 s + 1`
//! vector operations per sixteen pixels whatever `r` is.

use crate::simd::{load, max, min, store, subs, V};

const LANES: usize = 16;
/// Columns per strip of the vertical pass (the strip's working plane stays in L1).
const STRIP: usize = 64;

#[inline(always)]
fn ld(buf: &[u8], i: usize) -> V {
    debug_assert!(i + LANES <= buf.len());
    // SAFETY: `i + 16 <= buf.len()`, guaranteed by the callers' buffer sizes.
    unsafe { load(buf.as_ptr().add(i)) }
}

#[inline(always)]
fn st(buf: &mut [u8], i: usize, v: V) {
    debug_assert!(i + LANES <= buf.len());
    // SAFETY: as `ld`.
    unsafe { store(buf.as_mut_ptr().add(i), v) }
}

#[inline(always)]
fn ext<const MIN: bool>(a: V, b: V) -> V {
    if MIN {
        min(a, b)
    } else {
        max(a, b)
    }
}

/// The value that changes nothing: `255` for a minimum, `0` for a maximum.
const fn ident<const MIN: bool>() -> u8 {
    if MIN {
        u8::MAX
    } else {
        0
    }
}

/// Block size `s` (largest power of two `<= k`) and the offset `k - s` of the second block.
fn blocks(r: usize) -> (usize, usize) {
    let k = 2 * r + 1;
    let s = 1 << k.ilog2();
    (s, k - s)
}

/// Stores the first `valid` lanes of `v` (`<= 16`) at `dst[x..]`.
#[inline(always)]
fn put(dst: &mut [u8], x: usize, v: V) {
    if x + LANES <= dst.len() {
        st(dst, x, v);
    } else {
        let mut tail = [0u8; LANES];
        st(&mut tail, 0, v);
        let n = dst.len() - x;
        dst[x..].copy_from_slice(&tail[..n]);
    }
}

/// `src[x..x + 16]`, with lanes past the end of `src` zero.
#[inline(always)]
fn get(src: &[u8], x: usize) -> V {
    if x + LANES <= src.len() {
        ld(src, x)
    } else {
        let mut tail = [0u8; LANES];
        tail[..src.len() - x].copy_from_slice(&src[x..]);
        ld(&tail, 0)
    }
}

/// Running extreme over `2 r + 1` pixels along every row.
fn pass_rows<const MIN: bool>(src: &[u8], dst: &mut [u8], h: usize, w: usize, r: usize) {
    let (s, off) = blocks(r);
    let xn = w.div_ceil(LANES) * LANES; // output positions
    let ln = xn + off; // the final blocks start at `x` and `x + off`, `x < xn`
    let nb = (ln + s).max(r + w) + 3 * LANES;
    let mut buf = vec![ident::<MIN>(); nb];
    for y in 0..h {
        buf.fill(ident::<MIN>());
        buf[r..r + w].copy_from_slice(&src[y * w..(y + 1) * w]);
        let mut step = 1;
        while step < s {
            // blocks of `2 * step` are needed at the positions `< ln + s - 2 * step`
            for x in (0..ln + s - 2 * step).step_by(LANES) {
                let v = ext::<MIN>(ld(&buf, x), ld(&buf, x + step));
                st(&mut buf, x, v);
            }
            step *= 2;
        }
        let row = &mut dst[y * w..(y + 1) * w];
        for x in (0..xn).step_by(LANES) {
            put(row, x, ext::<MIN>(ld(&buf, x), ld(&buf, x + off)));
        }
    }
}

/// Running extreme over `2 r + 1` pixels down every column, in strips of `STRIP` columns;
/// with `sub_from` the result is subtracted from it (saturating) before it is stored.
fn pass_cols<const MIN: bool>(
    src: &[u8],
    dst: &mut [u8],
    h: usize,
    w: usize,
    r: usize,
    sub_from: Option<&[u8]>,
) {
    let (s, off) = blocks(r);
    let ln = h + off; // the final blocks start at `y` and `y + off`, `y < h`
    let nr = (ln + s).max(r + h) + 2;
    let mut plane = vec![ident::<MIN>(); nr * STRIP];
    for x0 in (0..w).step_by(STRIP) {
        let wd = STRIP.min(w - x0);
        let nv = wd.div_ceil(LANES);
        plane.fill(ident::<MIN>());
        for y in 0..h {
            let p = (r + y) * STRIP;
            plane[p..p + wd].copy_from_slice(&src[y * w + x0..y * w + x0 + wd]);
        }
        let mut step = 1;
        while step < s {
            // blocks of `2 * step` are needed at the rows `< ln + s - 2 * step`
            for y in 0..ln + s - 2 * step {
                for v in 0..nv {
                    let i = y * STRIP + v * LANES;
                    let e = ext::<MIN>(ld(&plane, i), ld(&plane, i + step * STRIP));
                    st(&mut plane, i, e);
                }
            }
            step *= 2;
        }
        for y in 0..h {
            let row = &mut dst[y * w..(y + 1) * w];
            for v in 0..nv {
                let i = y * STRIP + v * LANES;
                let mut e = ext::<MIN>(ld(&plane, i), ld(&plane, i + off * STRIP));
                if let Some(f) = sub_from {
                    e = subs(get(&f[y * w..(y + 1) * w], x0 + v * LANES), e);
                }
                put(row, x0 + v * LANES, e);
            }
        }
    }
}

/// White tophat with a `(2 wnd + 1)` square kernel: `f - dilate(erode(f))`.
pub fn tophat(src: &[u8], h: usize, w: usize, wnd: usize) -> Vec<u8> {
    assert_eq!(src.len(), h * w);
    if wnd == 0 || h == 0 || w == 0 {
        return src.to_vec();
    }
    let mut a = vec![0u8; h * w];
    let mut b = vec![0u8; h * w];
    pass_rows::<true>(src, &mut a, h, w, wnd);
    pass_cols::<true>(&a, &mut b, h, w, wnd, None);
    pass_rows::<false>(&b, &mut a, h, w, wnd);
    pass_cols::<false>(&a, &mut b, h, w, wnd, Some(src));
    b
}
