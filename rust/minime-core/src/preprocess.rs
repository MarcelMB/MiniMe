//! Per-frame preprocessing on uint8 frames, bit-exact with MiniMe's OpenCV calls:
//! `cv2.medianBlur` (replicated border), `cv2.warpAffine` INTER_LINEAR with a pure shift
//! (OpenCV 5: float32 lerp, round half to even, zeros outside) and `cv2.morphologyEx`
//! MORPH_TOPHAT with a square kernel (border pixels ignored by erode and dilate).

/// `median_blur`'s precondition: `cv2.medianBlur` takes odd sizes only (`k <= 1` skips the
/// filter). Whatever builds the parameters of a run or a preview checks it before the first frame.
pub fn check_median_ksize(k: usize) -> Result<(), String> {
    if k > 1 && k.is_multiple_of(2) {
        return Err(format!("median_ksize must be odd, got {k}"));
    }
    Ok(())
}

/// Median filter with a `k x k` window (odd `k`, see `check_median_ksize`), replicated border.
///
/// The default `k = 7` runs on `median::median7` (sorting networks on sixteen pixels at a
/// time), any other odd `k` on `median_blur_huang`; both give the same bits.
pub fn median_blur(src: &[u8], h: usize, w: usize, k: usize) -> Vec<u8> {
    if k == 7 {
        return crate::median::median7(src, h, w);
    }
    median_blur_huang(src, h, w, k)
}

/// `median_blur` for any odd `k`, scalar: the reference of `median::median7`.
///
/// Huang's sliding histogram per row: the window moves one column at a time and the
/// median is tracked with the count of values below it, so it moves only a few levels
/// per step.
pub fn median_blur_huang(src: &[u8], h: usize, w: usize, k: usize) -> Vec<u8> {
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
pub(crate) fn fma(a: f32, b: f32, c: f32) -> f32 {
    (f64::from(a) * f64::from(b) + f64::from(c)) as f32
}

/// `cv2.warpAffine(f, [[1, 0, -dx], [0, 1, -dy]], INTER_LINEAR)` on any pixel type:
/// `out(y, x) = f(y + dy, x + dx)` by fused bilinear lerps, zeros outside; `store`
/// converts the float32 result to the output type.
fn shift_with<T: Copy, O>(
    src: &[T],
    h: usize,
    w: usize,
    dy: f32,
    dx: f32,
    load: impl Fn(T) -> f32,
    store: impl Fn(f32) -> O,
) -> Vec<O> {
    let at = |yy: i64, xx: i64| -> f32 {
        if yy >= 0 && yy < h as i64 && xx >= 0 && xx < w as i64 {
            load(src[yy as usize * w + xx as usize])
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
    let mut out = Vec::with_capacity(h * w);
    for y in 0..h {
        let sy = y as f32 + dy;
        let y0f = sy.floor();
        let (y0, ay) = (y0f as i64, sy - y0f);
        for &(x0, ax) in &xs {
            let p00 = at(y0, x0);
            let p01 = at(y0, x0 + 1);
            let p10 = at(y0 + 1, x0);
            let p11 = at(y0 + 1, x0 + 1);
            let top = fma(p01 - p00, ax, p00);
            let bot = fma(p11 - p10, ax, p10);
            out.push(store(fma(bot - top, ay, top)));
        }
    }
    out
}

/// `cv2.warpAffine(f, [[1, 0, -dx], [0, 1, -dy]], INTER_LINEAR)` on uint8 (OpenCV 5:
/// float32 lerps, round half to even), zeros outside.
pub fn shift(src: &[u8], h: usize, w: usize, dy: f32, dx: f32) -> Vec<u8> {
    shift_with(src, h, w, dy, dx, f32::from, |v| {
        v.round_ties_even().clamp(0.0, 255.0) as u8
    })
}

/// The same warp on float32 frames (`motion.apply_shifts` on the registration image).
pub fn shift_f32(src: &[f32], h: usize, w: usize, dy: f32, dx: f32) -> Vec<f32> {
    shift_with(src, h, w, dy, dx, |v| v, |v| v)
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

/// White tophat with a `(2 wnd + 1)` square kernel: `f - dilate(erode(f))`, on
/// `morph::tophat` (sixteen pixels at a time; the same bits as `tophat_scalar`).
pub fn tophat(src: &[u8], h: usize, w: usize, wnd: usize) -> Vec<u8> {
    crate::morph::tophat(src, h, w, wnd)
}

/// `tophat`, scalar van Herk / Gil-Werman: its reference.
pub fn tophat_scalar(src: &[u8], h: usize, w: usize, wnd: usize) -> Vec<u8> {
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

    /// `noise` squeezed to `levels` grey levels (many ties).
    fn coarse(n: usize, seed: u64, levels: u8) -> Vec<u8> {
        noise(n, seed)
            .iter()
            .map(|&v| v / (256 / u16::from(levels)) as u8)
            .collect()
    }

    #[test]
    fn median7_matches_huang_and_brute_force_on_every_small_size() {
        for h in 1..=20 {
            for w in 1..=40 {
                for (seed, levels) in [(1, 255), (2, 3), (3, 2)] {
                    let f = coarse(h * w, seed + (h * 100 + w) as u64, levels);
                    let got = median_blur(&f, h, w, 7);
                    assert_eq!(got, median_blur_huang(&f, h, w, 7), "{h}x{w} levels {levels}");
                    assert_eq!(got, brute_median(&f, h, w, 7), "{h}x{w} levels {levels}");
                }
            }
        }
    }

    #[test]
    fn median7_matches_huang_on_odd_and_large_sizes_and_extreme_values() {
        for (h, w) in [
            (600, 600),
            (601, 599),
            (64, 641),
            (7, 7),
            (6, 100),
            (100, 6),
            (513, 17),
        ] {
            let f = noise(h * w, 9);
            assert_eq!(
                median_blur(&f, h, w, 7),
                median_blur_huang(&f, h, w, 7),
                "{h}x{w}"
            );
            // only the extreme grey levels, and a constant image
            let ext: Vec<u8> = f.iter().map(|&v| if v & 1 == 0 { 0 } else { 255 }).collect();
            assert_eq!(
                median_blur(&ext, h, w, 7),
                median_blur_huang(&ext, h, w, 7),
                "{h}x{w}"
            );
            let flat = vec![255u8; h * w];
            assert_eq!(median_blur(&flat, h, w, 7), flat);
        }
    }

    #[test]
    fn other_kernel_sizes_still_use_the_scalar_filter() {
        let (h, w) = (31, 45);
        let f = noise(h * w, 4);
        for k in [1, 3, 5, 9, 11] {
            assert_eq!(median_blur(&f, h, w, k), median_blur_huang(&f, h, w, k));
        }
    }

    #[test]
    fn tophat_matches_the_scalar_filter_on_every_small_size_and_radius() {
        for h in [1, 2, 3, 7, 16, 17, 31, 40] {
            for w in [1, 2, 5, 15, 16, 17, 31, 33, 64, 65, 70, 130] {
                for wnd in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 20, 33] {
                    for (seed, levels) in [(h as u64 * 31 + w as u64, 255), (7, 3)] {
                        let f = coarse(h * w, seed, levels);
                        assert_eq!(
                            tophat(&f, h, w, wnd),
                            tophat_scalar(&f, h, w, wnd),
                            "{h}x{w} wnd {wnd}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tophat_matches_the_scalar_filter_on_frame_sizes() {
        for (h, w) in [(600, 600), (601, 599), (64, 641), (513, 17)] {
            let f = noise(h * w, 11);
            assert_eq!(tophat(&f, h, w, 15), tophat_scalar(&f, h, w, 15), "{h}x{w}");
        }
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
