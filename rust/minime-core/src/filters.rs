//! The filters behind MiniMe's registration image (`preprocess.bandpass`, `motion._prep`),
//! bit-exact with OpenCV 5 + KleidiCV on arm64:
//!
//! * `cv2.GaussianBlur` on uint8: OpenCV's fixed-point path (8-bit kernel from error
//!   diffusion, integer sums, round at the end);
//! * `cv2.GaussianBlur` on float32: separable, fused multiply-adds in OpenCV's SIMD order;
//! * `cv2.resize` INTER_AREA by exactly 1/2 (uint8 and float32);
//! * `cv2.resize` INTER_LINEAR by exactly 2 on float32 (KleidiCV's `resize_2x2_f32`).
//!
//! All borders are OpenCV's default `BORDER_REFLECT_101`.

use crate::fsimd::{add, even, ld, mul, odd, splat, st, zip_hi, zip_lo, D4, F4};
use crate::preprocess::fma;
use crate::simd::{load, mean2x2, store};

/// Index `i` reflected into `0..n` without repeating the edge (`gfedcb|abcdefgh|gfedcba`).
fn reflect101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// `cvRound(sigma * per_side * 2 + 1) | 1`: the kernel size `GaussianBlur` picks for
/// `ksize = (0, 0)` (`per_side` is 3 for uint8, 4 for float).
fn gauss_ksize(sigma: f64, per_side: f64) -> usize {
    ((sigma * per_side * 2.0 + 1.0).round_ties_even() as usize) | 1
}

/// OpenCV's `getGaussianKernelBitExact` (odd `n`), in float64.
fn gauss_kernel(n: usize, sigma: f64) -> Vec<f64> {
    let n2 = (n - 1) / 2;
    let scale2x = -0.125 / (sigma * sigma);
    let vals: Vec<f64> = (0..n2)
        .map(|i| {
            let x = 1 - n as i64 + 2 * i as i64;
            ((x * x) as f64 * scale2x).exp()
        })
        .collect();
    let sum = vals.iter().sum::<f64>() * 2.0 + 1.0;
    let mul1 = 1.0 / sum;
    let mut k = vec![0.0; n];
    for (i, v) in vals.iter().enumerate() {
        k[i] = v * mul1;
        k[n - 1 - i] = v * mul1;
    }
    k[n2] = mul1;
    k
}

/// OpenCV's `getGaussianKernelFixedPoint_ED`: integer weights with 8 fraction bits summing
/// to 256, rounded with error diffusion from the outside in.
fn gauss_kernel_fixed(n: usize, sigma: f64) -> Vec<u32> {
    let k = gauss_kernel(n, sigma);
    let mut out = vec![0u32; n];
    let mut err = 0.0;
    let mut sum = 0u32;
    for i in 0..n / 2 {
        let adj = k[i] * 256.0 + err;
        let v = adj.round_ties_even();
        err = adj - v;
        out[i] = v as u32;
        out[n - 1 - i] = v as u32;
        sum += v as u32;
    }
    out[n / 2] = 256 - 2 * sum;
    out
}

/// The taps `motion::bandpass(.., lo, hi)` filters with: `gauss_u8`'s fixed-point weights for `lo`
/// and `gauss_f32`'s float32 weights for `hi / 2` (the half-resolution image), as the GPU kernels
/// (`web/src/gpu/`) need them to give the same bits.
pub fn bandpass_taps(lo: f64, hi: f64) -> (Vec<u32>, Vec<f32>) {
    let s = hi / 2.0;
    let wide = gauss_kernel(gauss_ksize(s, 4.0), s)
        .iter()
        .map(|&v| v as f32)
        .collect();
    (gauss_kernel_fixed(gauss_ksize(lo, 3.0), lo), wide)
}

/// `cv2.GaussianBlur(src, (0, 0), sigma)` on uint8.
///
/// The fixed-point sums are exact integers, so the order of the two passes is free: first
/// down the columns in `u16` (the weights sum to 256, so a sum is at most 255 * 256), one
/// output row at a time, then along that row in `u32`, round and clamp. Written as plain
/// loops over equal-length slices, which LLVM vectorises on NEON and wasm `simd128`.
pub fn gauss_u8(src: &[u8], h: usize, w: usize, sigma: f64) -> Vec<u8> {
    let n = gauss_ksize(sigma, 3.0);
    let k = gauss_kernel_fixed(n, sigma);
    let r = n / 2;
    let mut out = vec![0u8; h * w];
    let mut col = vec![0u16; w];
    let mut line = vec![0u16; w + 2 * r];
    let mut acc = vec![0u32; w];
    for y in 0..h {
        let centre = &src[y * w..(y + 1) * w];
        let kc = k[r] as u16;
        for (c, &v) in col.iter_mut().zip(centre) {
            *c = kc * u16::from(v);
        }
        for j in 1..=r {
            let kj = k[r + j] as u16;
            let up = &src[reflect101(y as isize - j as isize, h) * w..][..w];
            let dn = &src[reflect101(y as isize + j as isize, h) * w..][..w];
            for ((c, &u), &d) in col.iter_mut().zip(up).zip(dn) {
                *c += kj * (u16::from(u) + u16::from(d));
            }
        }
        line[r..r + w].copy_from_slice(&col);
        for i in 1..=r {
            line[r - i] = col[reflect101(-(i as isize), w)];
            line[r + w - 1 + i] = col[reflect101((w - 1 + i) as isize, w)];
        }
        let kc = k[r];
        for (a, &v) in acc.iter_mut().zip(&line[r..r + w]) {
            *a = kc * u32::from(v);
        }
        for j in 1..=r {
            let kj = k[r + j];
            let (l, rt) = (&line[r - j..r - j + w], &line[r + j..r + j + w]);
            for ((a, &x), &z) in acc.iter_mut().zip(l).zip(rt) {
                *a += kj * (u32::from(x) + u32::from(z));
            }
        }
        for (o, &a) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = ((a + (1 << 15)) >> 16).min(255) as u8;
        }
    }
    out
}

/// `gauss_u8` as first written (per-pixel reflected indices, two full passes): the reference
/// of the fast one.
pub fn gauss_u8_scalar(src: &[u8], h: usize, w: usize, sigma: f64) -> Vec<u8> {
    let n = gauss_ksize(sigma, 3.0);
    let k = gauss_kernel_fixed(n, sigma);
    let r = (n / 2) as isize;
    let mut rows = vec![0u32; h * w];
    for y in 0..h {
        let line = &src[y * w..(y + 1) * w];
        for x in 0..w {
            rows[y * w + x] = k
                .iter()
                .enumerate()
                .map(|(j, &kj)| kj * u32::from(line[reflect101(x as isize + j as isize - r, w)]))
                .sum();
        }
    }
    let mut out = vec![0u8; h * w];
    for y in 0..h {
        let ys: Vec<usize> = (0..n)
            .map(|j| reflect101(y as isize + j as isize - r, h))
            .collect();
        for x in 0..w {
            let s: u32 = ys.iter().zip(&k).map(|(&yy, &kj)| kj * rows[yy * w + x]).sum();
            out[y * w + x] = ((s + (1 << 15)) >> 16).min(255) as u8;
        }
    }
    out
}

/// Vectors of four pixels filtered side by side in `gauss_f32`.
const CHAINS: usize = 4;

/// `cv2.GaussianBlur(src, (0, 0), sigma)` on float32: rows as `s = fma(x[k], kx[k], s)`,
/// then columns from the centre out on the summed symmetric pairs.
///
/// Four pixels at a time with `fsimd`'s exact `fma` emulation, in the scalar code's order
/// (`gauss_f32_scalar`); the last `w % 4` columns of each row take the scalar path.
pub fn gauss_f32(src: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
    let n = gauss_ksize(sigma, 4.0);
    let k: Vec<f32> = gauss_kernel(n, sigma).iter().map(|&v| v as f32).collect();
    let r = n / 2;
    let kd: Vec<D4> = k.iter().map(|&v| D4::splat(v)).collect();
    let nv = w / 4 * 4; // columns done four at a time
    let mut line = vec![0f32; w + 2 * r];
    let mut rows = vec![0f32; h * w];
    for y in 0..h {
        let s = &src[y * w..(y + 1) * w];
        for (i, l) in line.iter_mut().enumerate() {
            *l = s[reflect101(i as isize - r as isize, w)];
        }
        let o = &mut rows[y * w..(y + 1) * w];
        // `CHAINS` vectors at a time: each tap depends on the one before it, so independent
        // vectors fill the pipeline
        let mut x = 0;
        while x < nv {
            let n = CHAINS.min((nv - x) / 4);
            let mut acc = [D4::splat(0.0); CHAINS];
            for (j, kj) in kd.iter().enumerate() {
                for (c, a) in acc.iter_mut().enumerate().take(n) {
                    *a = (D4::from_f4(ld(&line, x + 4 * c + j)) * *kj + *a).round32();
                }
            }
            for (c, a) in acc.iter().enumerate().take(n) {
                st(o, x + 4 * c, a.to_f4());
            }
            x += 4 * n;
        }
        for (x, v) in o.iter_mut().enumerate().skip(nv) {
            let mut acc = 0.0;
            for (j, &kj) in k.iter().enumerate() {
                acc = fma(line[x + j], kj, acc);
            }
            *v = acc;
        }
    }
    let mut out = vec![0f32; h * w];
    let kc = splat(k[r]);
    for y in 0..h {
        let pairs: Vec<(usize, usize)> = (1..=r)
            .map(|j| {
                (
                    reflect101(y as isize - j as isize, h) * w,
                    reflect101(y as isize + j as isize, h) * w,
                )
            })
            .collect();
        let o = &mut out[y * w..(y + 1) * w];
        let mut x = 0;
        while x < nv {
            let n = CHAINS.min((nv - x) / 4);
            let mut acc: [D4; CHAINS] =
                std::array::from_fn(|c| D4::from_f4(mul(ld(&rows, y * w + x + 4 * (c % n)), kc)));
            for (j, &(a, b)) in pairs.iter().enumerate() {
                for (c, v) in acc.iter_mut().enumerate().take(n) {
                    let t = add(ld(&rows, a + x + 4 * c), ld(&rows, b + x + 4 * c));
                    *v = (D4::from_f4(t) * kd[r + j + 1] + *v).round32();
                }
            }
            for (c, v) in acc.iter().enumerate().take(n) {
                st(o, x + 4 * c, v.to_f4());
            }
            x += 4 * n;
        }
        for (x, v) in o.iter_mut().enumerate().skip(nv) {
            *v = rows[y * w + x] * k[r];
            for (j, &(a, b)) in pairs.iter().enumerate() {
                *v = fma(rows[a + x] + rows[b + x], k[r + j + 1], *v);
            }
        }
    }
    out
}

/// `gauss_f32` as first written (one pixel at a time): the reference of the vector one.
pub fn gauss_f32_scalar(src: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
    let n = gauss_ksize(sigma, 4.0);
    let k: Vec<f32> = gauss_kernel(n, sigma).iter().map(|&v| v as f32).collect();
    let r = n / 2;
    let xs: Vec<usize> = (0..w + 2 * r)
        .map(|i| reflect101(i as isize - r as isize, w))
        .collect();
    let mut rows = vec![0f32; h * w];
    for y in 0..h {
        let line = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let mut s = 0.0;
            for (j, &kj) in k.iter().enumerate() {
                s = fma(line[xs[x + j]], kj, s);
            }
            rows[y * w + x] = s;
        }
    }
    let mut out = vec![0f32; h * w];
    for y in 0..h {
        let up = |j: usize| reflect101(y as isize - j as isize, h) * w;
        let dn = |j: usize| reflect101(y as isize + j as isize, h) * w;
        let o = &mut out[y * w..(y + 1) * w];
        for (x, v) in o.iter_mut().enumerate() {
            *v = rows[y * w + x] * k[r];
        }
        for j in 1..=r {
            let (a, b) = (up(j), dn(j));
            for (x, v) in o.iter_mut().enumerate() {
                *v = fma(rows[a + x] + rows[b + x], k[r + j], *v);
            }
        }
    }
    out
}

/// `cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)` on uint8 with
/// even `h`, `w`: rounded mean of each 2 x 2 block (sixteen blocks at a time).
pub fn area_down2_u8(src: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0u8; oh * ow];
    for (y, o) in out.chunks_exact_mut(ow.max(1)).enumerate().take(oh) {
        let r0 = &src[2 * y * w..2 * y * w + 2 * ow];
        let r1 = &src[(2 * y + 1) * w..(2 * y + 1) * w + 2 * ow];
        let (o16, o_rest) = o.as_chunks_mut::<16>();
        let (a16, a_rest) = r0.as_chunks::<32>();
        let (b16, b_rest) = r1.as_chunks::<32>();
        for ((o, a), b) in o16.iter_mut().zip(a16).zip(b16) {
            // SAFETY: `a` and `b` are 32 readable bytes, `o` 16 writable ones.
            unsafe {
                let v = mean2x2(
                    load(a.as_ptr()),
                    load(a.as_ptr().add(16)),
                    load(b.as_ptr()),
                    load(b.as_ptr().add(16)),
                );
                store(o.as_mut_ptr(), v);
            }
        }
        for ((o, a), b) in o_rest
            .iter_mut()
            .zip(a_rest.as_chunks::<2>().0)
            .zip(b_rest.as_chunks::<2>().0)
        {
            let s = u16::from(a[0]) + u16::from(a[1]) + u16::from(b[0]) + u16::from(b[1]);
            *o = ((s + 2) >> 2) as u8;
        }
    }
    out
}

/// The same on float32: `((a + b) + (c + d)) * 0.25`.
pub fn area_down2_f32(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0f32; oh * ow];
    let quarter = splat(0.25);
    for (y, o) in out.chunks_exact_mut(ow.max(1)).enumerate().take(oh) {
        let r0 = &src[2 * y * w..2 * y * w + 2 * ow];
        let r1 = &src[(2 * y + 1) * w..(2 * y + 1) * w + 2 * ow];
        let mut x = 0;
        while x + 4 <= ow {
            let (a0, a1, b0, b1) = (ld(r0, 2 * x), ld(r0, 2 * x + 4), ld(r1, 2 * x), ld(r1, 2 * x + 4));
            let top = add(even(a0, a1), odd(a0, a1));
            let bottom = add(even(b0, b1), odd(b0, b1));
            st(o, x, mul(add(top, bottom), quarter));
            x += 4;
        }
        for (x, o) in o.iter_mut().enumerate().skip(x) {
            *o = ((r0[2 * x] + r0[2 * x + 1]) + (r1[2 * x] + r1[2 * x + 1])) * 0.25;
        }
    }
    out
}

/// `area_down2_u8` one pixel at a time: its reference.
pub fn area_down2_u8_scalar(src: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = Vec::with_capacity(oh * ow);
    for y in 0..oh {
        let (r0, r1) = (&src[2 * y * w..], &src[(2 * y + 1) * w..]);
        for x in 0..ow {
            let s = u16::from(r0[2 * x])
                + u16::from(r0[2 * x + 1])
                + u16::from(r1[2 * x])
                + u16::from(r1[2 * x + 1]);
            out.push(((s + 2) >> 2) as u8);
        }
    }
    out
}

/// `area_down2_f32` one pixel at a time: its reference.
pub fn area_down2_f32_scalar(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = Vec::with_capacity(oh * ow);
    for y in 0..oh {
        let (r0, r1) = (&src[2 * y * w..], &src[(2 * y + 1) * w..]);
        for x in 0..ow {
            out.push(((r0[2 * x] + r0[2 * x + 1]) + (r1[2 * x] + r1[2 * x + 1])) * 0.25);
        }
    }
    out
}

/// `cv2.resize(src, (2 w, 2 h), interpolation=cv2.INTER_LINEAR)` on float32, as KleidiCV's
/// `resize_2x2_f32`: each output pixel weights its four nearest inputs 9/16, 3/16, 3/16,
/// 1/16 (edges 3/4, 1/4). Its NEON body (input columns `< (w - 1) / 4 * 4`) fuses as
/// `fma(far, w, near * w)`, the scalar tail and edge columns as `fma(near, w, far * w)`.
pub fn resize_up2_f32(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    const W0: f32 = 0.5625;
    const W1: f32 = 0.1875;
    const W3: f32 = 0.0625;
    fn l1_vec(near: f32, far: f32) -> f32 {
        fma(far, 0.25, near * 0.75)
    }
    fn l1_sc(near: f32, far: f32) -> f32 {
        fma(near, 0.75, far * 0.25)
    }
    fn l2_sc(a: f32, b: f32, c: f32, d: f32) -> f32 {
        fma(d, W3, fma(c, W1, fma(a, W0, b * W1)))
    }
    /// The NEON body's `fma(d, W3, fma(c, W1, fma(b, W1, a * W0)))` on four columns.
    #[inline(always)]
    fn l2_lanes(a: D4, b: D4, c: D4, d: D4) -> F4 {
        let (w0, w1, w3) = (D4::splat(W0), D4::splat(W1), D4::splat(W3));
        let t = (a * w0).round32();
        let t = (b * w1 + t).round32();
        let t = (c * w1 + t).round32();
        (d * w3 + t).to_f4()
    }
    let (oh, ow) = (2 * h, 2 * w);
    let vec_end = (w - 1) / 4 * 4;
    let mut out = vec![0f32; oh * ow];
    let edge_row = |s: &[f32], d: &mut [f32]| {
        d[0] = s[0];
        d[ow - 1] = s[w - 1];
        for x in 0..w - 1 {
            let l: fn(f32, f32) -> f32 = if x < vec_end { l1_vec } else { l1_sc };
            d[2 * x + 1] = l(s[x], s[x + 1]);
            d[2 * x + 2] = l(s[x + 1], s[x]);
        }
    };
    edge_row(&src[..w], &mut out[..ow]);
    edge_row(&src[(h - 1) * w..], &mut out[(oh - 1) * ow..]);
    for y in 0..h - 1 {
        let (s0, s1) = (&src[y * w..(y + 1) * w], &src[(y + 1) * w..(y + 2) * w]);
        let (d0, d1) = out[(2 * y + 1) * ow..(2 * y + 3) * ow].split_at_mut(ow);
        d0[0] = l1_sc(s0[0], s1[0]);
        d1[0] = l1_sc(s1[0], s0[0]);
        d0[ow - 1] = l1_sc(s0[w - 1], s1[w - 1]);
        d1[ow - 1] = l1_sc(s1[w - 1], s0[w - 1]);
        // the first `vec_end` input columns, four at a time with the exact fma emulation
        for x in (0..vec_end).step_by(4) {
            let [a, b, c, d] = [ld(s0, x), ld(s0, x + 1), ld(s1, x), ld(s1, x + 1)].map(D4::from_f4);
            let (e0, o0) = (l2_lanes(a, b, c, d), l2_lanes(b, a, d, c));
            let (e1, o1) = (l2_lanes(c, a, d, b), l2_lanes(d, b, c, a));
            st(d0, 2 * x + 1, zip_lo(e0, o0));
            st(d0, 2 * x + 5, zip_hi(e0, o0));
            st(d1, 2 * x + 1, zip_lo(e1, o1));
            st(d1, 2 * x + 5, zip_hi(e1, o1));
        }
        for x in vec_end..w - 1 {
            let (a, b, c, d) = (s0[x], s0[x + 1], s1[x], s1[x + 1]);
            d0[2 * x + 1] = l2_sc(a, b, c, d);
            d0[2 * x + 2] = l2_sc(b, a, d, c);
            d1[2 * x + 1] = l2_sc(c, a, d, b);
            d1[2 * x + 2] = l2_sc(d, b, c, a);
        }
    }
    out
}

/// `resize_up2_f32` one pixel at a time: its reference.
///
/// `cv2.resize(src, (2 w, 2 h), interpolation=cv2.INTER_LINEAR)` on float32, as KleidiCV's
/// `resize_2x2_f32`: each output pixel weights its four nearest inputs 9/16, 3/16, 3/16,
/// 1/16 (edges 3/4, 1/4). Its NEON body (input columns `< (w - 1) / 4 * 4`) fuses as
/// `fma(far, w, near * w)`, the scalar tail and edge columns as `fma(near, w, far * w)`.
pub fn resize_up2_f32_scalar(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    const W0: f32 = 0.5625;
    const W1: f32 = 0.1875;
    const W3: f32 = 0.0625;
    fn l1_vec(near: f32, far: f32) -> f32 {
        fma(far, 0.25, near * 0.75)
    }
    fn l1_sc(near: f32, far: f32) -> f32 {
        fma(near, 0.75, far * 0.25)
    }
    fn l2_vec(a: f32, b: f32, c: f32, d: f32) -> f32 {
        fma(d, W3, fma(c, W1, fma(b, W1, a * W0)))
    }
    fn l2_sc(a: f32, b: f32, c: f32, d: f32) -> f32 {
        fma(d, W3, fma(c, W1, fma(a, W0, b * W1)))
    }
    let (oh, ow) = (2 * h, 2 * w);
    let vec_end = (w - 1) / 4 * 4;
    let mut out = vec![0f32; oh * ow];
    let edge_row = |s: &[f32], d: &mut [f32]| {
        d[0] = s[0];
        d[ow - 1] = s[w - 1];
        for x in 0..w - 1 {
            let l: fn(f32, f32) -> f32 = if x < vec_end { l1_vec } else { l1_sc };
            d[2 * x + 1] = l(s[x], s[x + 1]);
            d[2 * x + 2] = l(s[x + 1], s[x]);
        }
    };
    edge_row(&src[..w], &mut out[..ow]);
    edge_row(&src[(h - 1) * w..], &mut out[(oh - 1) * ow..]);
    for y in 0..h - 1 {
        let (s0, s1) = (&src[y * w..(y + 1) * w], &src[(y + 1) * w..(y + 2) * w]);
        let (d0, d1) = out[(2 * y + 1) * ow..(2 * y + 3) * ow].split_at_mut(ow);
        d0[0] = l1_sc(s0[0], s1[0]);
        d1[0] = l1_sc(s1[0], s0[0]);
        d0[ow - 1] = l1_sc(s0[w - 1], s1[w - 1]);
        d1[ow - 1] = l1_sc(s1[w - 1], s0[w - 1]);
        for x in 0..w - 1 {
            let l: fn(f32, f32, f32, f32) -> f32 = if x < vec_end { l2_vec } else { l2_sc };
            let (a, b, c, d) = (s0[x], s0[x + 1], s1[x], s1[x + 1]);
            d0[2 * x + 1] = l(a, b, c, d);
            d0[2 * x + 2] = l(b, a, d, c);
            d1[2 * x + 1] = l(c, a, d, b);
            d1[2 * x + 2] = l(d, b, c, a);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencv_kernels() {
        assert_eq!(gauss_ksize(1.0, 3.0), 7);
        assert_eq!(gauss_ksize(3.0, 4.0), 25);
        assert_eq!(gauss_kernel_fixed(7, 1.0), vec![1, 14, 62, 102, 62, 14, 1]);
    }

    #[test]
    fn reflect101_mirrors_without_the_edge() {
        let got: Vec<usize> = (-3..8).map(|i| reflect101(i, 5)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    }

    fn noise(n: usize, seed: u64, levels: u16) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as u16 % levels) as u8
            })
            .collect()
    }

    #[test]
    fn gauss_u8_matches_the_scalar_reference() {
        for h in [1, 2, 3, 4, 7, 8, 9, 31, 40] {
            for w in [1, 2, 3, 6, 7, 8, 9, 15, 16, 17, 33, 100] {
                for sigma in [0.3, 0.8, 1.0, 1.7, 2.5, 4.0] {
                    for levels in [256, 2] {
                        let f = noise(h * w, (h * 1000 + w) as u64, levels);
                        assert_eq!(
                            gauss_u8(&f, h, w, sigma),
                            gauss_u8_scalar(&f, h, w, sigma),
                            "{h}x{w} sigma {sigma}"
                        );
                    }
                }
            }
        }
        let f = noise(600 * 600, 5, 256);
        assert_eq!(gauss_u8(&f, 600, 600, 1.0), gauss_u8_scalar(&f, 600, 600, 1.0));
        let ones = vec![255u8; 601 * 599];
        assert_eq!(gauss_u8(&ones, 601, 599, 1.0), ones);
    }

    fn noise_f32(n: usize, seed: u64) -> Vec<f32> {
        noise(n, seed, 256)
            .iter()
            .zip(noise(n, seed ^ 77, 1000))
            .map(|(&a, b)| f32::from(a) + f32::from(b as u16) / 1000.0 - 0.5)
            .collect()
    }

    #[test]
    fn gauss_f32_matches_the_scalar_reference_bit_for_bit() {
        for h in [1, 2, 3, 5, 9, 13, 30] {
            for w in [1, 2, 3, 4, 5, 8, 11, 13, 16, 29, 50] {
                for sigma in [0.5, 1.0, 3.0, 4.2] {
                    let f = noise_f32(h * w, (h * 1000 + w) as u64);
                    let (a, b) = (gauss_f32(&f, h, w, sigma), gauss_f32_scalar(&f, h, w, sigma));
                    assert!(
                        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{h}x{w} sigma {sigma}"
                    );
                }
            }
        }
        // u8-valued input at the size the registration image uses
        let f: Vec<f32> = noise(300 * 300, 3, 256).iter().map(|&v| f32::from(v)).collect();
        let (a, b) = (gauss_f32(&f, 300, 300, 3.0), gauss_f32_scalar(&f, 300, 300, 3.0));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn resize_up2_matches_the_scalar_reference_bit_for_bit() {
        for h in [1, 2, 3, 6, 11] {
            for w in [1, 2, 3, 4, 5, 8, 9, 10, 13, 17, 31] {
                let f = noise_f32(h * w, (h * 1000 + w) as u64);
                let (a, b) = (resize_up2_f32(&f, h, w), resize_up2_f32_scalar(&f, h, w));
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{h}x{w}"
                );
            }
        }
        let f = noise_f32(300 * 300, 8);
        let (a, b) = (resize_up2_f32(&f, 300, 300), resize_up2_f32_scalar(&f, 300, 300));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn area_down2_matches_the_per_pixel_formula() {
        for (h, w) in [(1, 1), (2, 2), (3, 5), (8, 6), (9, 17), (600, 600), (601, 599)] {
            let f = noise(h * w, 4, 256);
            let ff = noise_f32(h * w, 4);
            let (oh, ow) = (h / 2, w / 2);
            let mut want_u8 = Vec::new();
            let mut want_f32 = Vec::new();
            for y in 0..oh {
                for x in 0..ow {
                    let i = [
                        2 * y * w + 2 * x,
                        2 * y * w + 2 * x + 1,
                        (2 * y + 1) * w + 2 * x,
                        (2 * y + 1) * w + 2 * x + 1,
                    ];
                    let s: u16 = i.iter().map(|&k| u16::from(f[k])).sum();
                    want_u8.push(((s + 2) >> 2) as u8);
                    want_f32.push(((ff[i[0]] + ff[i[1]]) + (ff[i[2]] + ff[i[3]])) * 0.25);
                }
            }
            assert_eq!(area_down2_u8(&f, h, w), want_u8, "{h}x{w}");
            assert_eq!(area_down2_u8(&f, h, w), area_down2_u8_scalar(&f, h, w), "{h}x{w}");
            assert!(same_bits_f32(&area_down2_f32(&ff, h, w), &want_f32), "{h}x{w}");
            assert!(
                same_bits_f32(&area_down2_f32(&ff, h, w), &area_down2_f32_scalar(&ff, h, w)),
                "{h}x{w}"
            );
        }
    }

    fn same_bits_f32(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn constant_images_stay_constant() {
        let (h, w) = (9, 13);
        assert!(gauss_u8(&vec![77; h * w], h, w, 1.0).iter().all(|&v| v == 77));
        let g = gauss_f32(&vec![2.5; h * w], h, w, 3.0);
        assert!(g.iter().all(|&v| (v - 2.5).abs() < 1e-5));
        assert!(resize_up2_f32(&vec![4.0; h * w], h, w).iter().all(|&v| v == 4.0));
        assert!(area_down2_f32(&[1.5; 8 * 6], 8, 6).iter().all(|&v| v == 1.5));
    }
}
