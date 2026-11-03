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

use crate::preprocess::fma;

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

/// `cv2.GaussianBlur(src, (0, 0), sigma)` on uint8.
pub fn gauss_u8(src: &[u8], h: usize, w: usize, sigma: f64) -> Vec<u8> {
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

/// `cv2.GaussianBlur(src, (0, 0), sigma)` on float32: rows as `s = fma(x[k], kx[k], s)`,
/// then columns from the centre out on the summed symmetric pairs.
pub fn gauss_f32(src: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
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
/// even `h`, `w`: rounded mean of each 2 x 2 block.
pub fn area_down2_u8(src: &[u8], h: usize, w: usize) -> Vec<u8> {
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

/// The same on float32: `((a + b) + (c + d)) * 0.25`.
pub fn area_down2_f32(src: &[f32], h: usize, w: usize) -> Vec<f32> {
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
