//! Per-frame preprocessing on uint8 frames, bit-exact with MiniMe's OpenCV calls:
//! `cv2.medianBlur` (replicated border), `cv2.warpAffine` INTER_LINEAR with a pure shift
//! (OpenCV 5: float32 lerp, round half to even, zeros outside) and `cv2.morphologyEx`
//! MORPH_TOPHAT with a square kernel (border pixels ignored by erode and dilate).

/// Median filter with a `k x k` window (odd `k`), replicated border.
///
/// Huang's sliding histogram per row: the window moves one column at a time and the
/// median is tracked with the count of values below it, so it moves only a few levels
/// per step.
pub fn median_blur(src: &[u8], h: usize, w: usize, k: usize) -> Vec<u8> {
    if k <= 1 {
        return src.to_vec();
    }
    let r = k / 2;
    let rank = (k * k) / 2; // 0-based rank of the median
    let col = |x: isize| x.clamp(0, w as isize - 1) as usize;
    let row = |y: isize| y.clamp(0, h as isize - 1) as usize;
    let mut out = vec![0u8; h * w];
    for y in 0..h {
        let rows: Vec<usize> = (0..k)
            .map(|i| row(y as isize + i as isize - r as isize))
            .collect();
        let mut hist = [0u32; 256];
        for &yy in &rows {
            for i in 0..k {
                hist[src[yy * w + col(i as isize - r as isize)] as usize] += 1;
            }
        }
        // median and count below it
        let mut med = 0usize;
        let mut lt = 0u32;
        while lt + hist[med] <= rank as u32 {
            lt += hist[med];
            med += 1;
        }
        out[y * w] = med as u8;
        for x in 1..w {
            let xo = col(x as isize - r as isize - 1);
            let xi = col(x as isize + r as isize);
            for &yy in &rows {
                let vo = src[yy * w + xo] as usize;
                let vi = src[yy * w + xi] as usize;
                hist[vo] -= 1;
                if vo < med {
                    lt -= 1;
                }
                hist[vi] += 1;
                if vi < med {
                    lt += 1;
                }
            }
            while lt > rank as u32 {
                med -= 1;
                lt -= hist[med];
            }
            while lt + hist[med] <= rank as u32 {
                lt += hist[med];
                med += 1;
            }
            out[y * w + x] = med as u8;
        }
    }
    out
}

/// `a * b + c` rounded once, as OpenCV's NEON `v_fma`. The float32 product is exact in
/// float64 and the sum is exact there for these magnitudes, so one rounding to float32
/// matches a fused multiply-add without wasm's slow software `fmaf`.
#[inline]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    (f64::from(a) * f64::from(b) + f64::from(c)) as f32
}

/// `cv2.warpAffine(f, [[1, 0, -dx], [0, 1, -dy]], INTER_LINEAR)`: `out(y, x) = f(y + dy,
/// x + dx)` by bilinear interpolation (fused lerps, round half to even), zeros outside.
pub fn shift(src: &[u8], h: usize, w: usize, dy: f32, dx: f32) -> Vec<u8> {
    let mut out = vec![0u8; h * w];
    let at = |yy: i64, xx: i64| -> f32 {
        if yy >= 0 && yy < h as i64 && xx >= 0 && xx < w as i64 {
            f32::from(src[yy as usize * w + xx as usize])
        } else {
            0.0
        }
    };
    let xs: Vec<(i64, f32)> = (0..w)
        .map(|x| {
            let sx = x as f32 + dx;
            let x0 = sx.floor();
            (x0 as i64, sx - x0)
        })
        .collect();
    for y in 0..h {
        let sy = y as f32 + dy;
        let y0f = sy.floor();
        let (y0, ay) = (y0f as i64, sy - y0f);
        for (x, &(x0, ax)) in xs.iter().enumerate() {
            let p00 = at(y0, x0);
            let p01 = at(y0, x0 + 1);
            let p10 = at(y0 + 1, x0);
            let p11 = at(y0 + 1, x0 + 1);
            let top = fma(p01 - p00, ax, p00);
            let bot = fma(p11 - p10, ax, p10);
            let v = fma(bot - top, ay, top);
            out[y * w + x] = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Running min (`is_min`) or max over windows of `2r + 1` samples along one line, samples
/// beyond the ends ignored (van Herk / Gil-Werman; `g`, `hh` are scratch of any size).
fn running_extreme(line: &[u8], r: usize, is_min: bool, out: &mut [u8], g: &mut Vec<u8>, hh: &mut Vec<u8>) {
    let n = line.len();
    let k = 2 * r + 1;
    let pad = if is_min { 255u8 } else { 0u8 };
    let op = |a: u8, b: u8| if is_min { a.min(b) } else { a.max(b) };
    let mlen = (n + 2 * r).div_ceil(k) * k;
    let get = |i: usize| if i >= r && i - r < n { line[i - r] } else { pad };
    g.resize(mlen, pad);
    hh.resize(mlen, pad);
    for i in 0..mlen {
        g[i] = if i % k == 0 { get(i) } else { op(g[i - 1], get(i)) };
    }
    for i in (0..mlen).rev() {
        hh[i] = if i % k == k - 1 {
            get(i)
        } else {
            op(hh[i + 1], get(i))
        };
    }
    for (i, o) in out.iter_mut().enumerate().take(n) {
        *o = op(hh[i], g[i + k - 1]); // window [i, i + k - 1] in padded coordinates
    }
}

/// The same along columns, all columns at once (rows of `w` lanes, so it vectorises).
fn running_extreme_cols(src: &[u8], h: usize, w: usize, r: usize, is_min: bool, out: &mut [u8]) {
    let k = 2 * r + 1;
    let pad = if is_min { 255u8 } else { 0u8 };
    let mlen = (h + 2 * r).div_ceil(k) * k;
    let row =
        |i: usize| -> Option<&[u8]> { (i >= r && i - r < h).then(|| &src[(i - r) * w..(i - r + 1) * w]) };
    let mut g = vec![pad; mlen * w];
    let mut hh = vec![pad; mlen * w];
    let lanes = |dst: &mut [u8], a: &[u8], b: Option<&[u8]>| match (b, is_min) {
        (Some(b), true) => dst
            .iter_mut()
            .zip(a.iter().zip(b))
            .for_each(|(d, (&x, &y))| *d = x.min(y)),
        (Some(b), false) => dst
            .iter_mut()
            .zip(a.iter().zip(b))
            .for_each(|(d, (&x, &y))| *d = x.max(y)),
        (None, _) => dst.copy_from_slice(a),
    };
    for i in 0..mlen {
        let (done, cur) = g.split_at_mut(i * w);
        let cur = &mut cur[..w];
        match (i % k == 0, row(i)) {
            (true, Some(x)) => cur.copy_from_slice(x),
            (true, None) => cur.fill(pad),
            (false, x) => lanes(cur, &done[(i - 1) * w..], x),
        }
    }
    for i in (0..mlen).rev() {
        let (cur, after) = hh.split_at_mut((i + 1) * w);
        let cur = &mut cur[i * w..];
        match (i % k == k - 1, row(i)) {
            (true, Some(x)) => cur.copy_from_slice(x),
            (true, None) => cur.fill(pad),
            (false, x) => lanes(cur, &after[..w], x),
        }
    }
    for i in 0..h {
        lanes(
            &mut out[i * w..(i + 1) * w],
            &hh[i * w..(i + 1) * w],
            Some(&g[(i + k - 1) * w..(i + k) * w]),
        );
    }
}

fn morph(src: &[u8], h: usize, w: usize, r: usize, is_min: bool) -> Vec<u8> {
    let mut tmp = vec![0u8; h * w];
    let (mut g, mut hh) = (Vec::new(), Vec::new());
    for y in 0..h {
        running_extreme(
            &src[y * w..(y + 1) * w],
            r,
            is_min,
            &mut tmp[y * w..(y + 1) * w],
            &mut g,
            &mut hh,
        );
    }
    let mut out = vec![0u8; h * w];
    running_extreme_cols(&tmp, h, w, r, is_min, &mut out);
    out
}

/// White tophat with a `(2 wnd + 1)` square kernel: `f - dilate(erode(f))`.
pub fn tophat(src: &[u8], h: usize, w: usize, wnd: usize) -> Vec<u8> {
    if wnd == 0 {
        return src.to_vec();
    }
    let opened = morph(&morph(src, h, w, wnd, true), h, w, wnd, false);
    src.iter()
        .zip(&opened)
        .map(|(&a, &b)| a.saturating_sub(b))
        .collect()
}

/// Settings of `stream.FrameStream._prep`.
#[derive(Debug, Clone, Copy)]
pub struct Prep {
    pub median_ksize: usize,
    pub tophat_wnd: usize,
    pub motion_before_tophat: bool,
}

impl Prep {
    /// Median, shift and tophat of one raw frame, in MiniMe's order.
    pub fn frame(&self, raw: &[u8], h: usize, w: usize, dy: f32, dx: f32) -> Vec<u8> {
        let den = median_blur(raw, h, w, self.median_ksize);
        if self.motion_before_tophat {
            tophat(&shift(&den, h, w, dy, dx), h, w, self.tophat_wnd)
        } else {
            shift(&tophat(&den, h, w, self.tophat_wnd), h, w, dy, dx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_median(src: &[u8], h: usize, w: usize, k: usize) -> Vec<u8> {
        let r = k as isize / 2;
        let mut out = vec![0; h * w];
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut v = Vec::new();
                for dy in -r..=r {
                    for dx in -r..=r {
                        let yy = (y + dy).clamp(0, h as isize - 1) as usize;
                        let xx = (x + dx).clamp(0, w as isize - 1) as usize;
                        v.push(src[yy * w + xx]);
                    }
                }
                v.sort_unstable();
                out[y as usize * w + x as usize] = v[v.len() / 2];
            }
        }
        out
    }

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    }

    #[test]
    fn median_matches_brute_force() {
        let (h, w) = (23, 31);
        let f = noise(h * w, 3);
        assert_eq!(median_blur(&f, h, w, 7), brute_median(&f, h, w, 7));
        assert_eq!(median_blur(&f, h, w, 3), brute_median(&f, h, w, 3));
    }

    #[test]
    fn tophat_matches_brute_force() {
        let (h, w, r) = (19, 27, 3);
        let f = noise(h * w, 5);
        let ext = |img: &[u8], is_min: bool| -> Vec<u8> {
            let mut o = vec![0; h * w];
            for y in 0..h as isize {
                for x in 0..w as isize {
                    let mut v: Option<u8> = None;
                    for dy in -(r as isize)..=r as isize {
                        for dx in -(r as isize)..=r as isize {
                            let (yy, xx) = (y + dy, x + dx);
                            if yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize {
                                let p = img[yy as usize * w + xx as usize];
                                v = Some(match v {
                                    None => p,
                                    Some(q) if is_min => q.min(p),
                                    Some(q) => q.max(p),
                                });
                            }
                        }
                    }
                    o[y as usize * w + x as usize] = v.unwrap();
                }
            }
            o
        };
        let opened = ext(&ext(&f, true), false);
        let want: Vec<u8> = f.iter().zip(&opened).map(|(a, b)| a - b).collect();
        assert_eq!(tophat(&f, h, w, r), want);
    }
}
