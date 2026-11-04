//! `compress`' noise estimate for four pixels at a time: a Stockham FFT of the real
//! signals as `T / 2` complex points (radices 2, 3, 4, 5), the four pixels in the four lanes
//! of every value (written as `[f32; 4]` so LLVM maps it to wasm simd128 / NEON, and to
//! scalar code anywhere else), then the real transform's last step and the logarithms of
//! `|X_k|^2 / T + eps` summed over the noise band bin by bin, in order, in float32 per
//! pixel as numpy sums them. `realfft` does one pixel per call and does not fill the
//! vector unit.
//!
//! Any `T` whose half is a product of 2, 3 and 5 is supported (`NoisePlan::new` returns
//! `None` otherwise and `compress` falls back to `realfft`).

use crate::fastmath::ln_f32;
use crate::prof;
use std::f64::consts::TAU;

/// Four float32 lanes, one pixel each.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct V(pub [f32; 4]);

impl V {
    pub const ZERO: V = V([0.0; 4]);

    #[inline(always)]
    pub fn splat(x: f32) -> V {
        V([x; 4])
    }
}

macro_rules! lanes {
    ($a:ident $op:tt $b:ident) => {
        V([$a.0[0] $op $b.0[0], $a.0[1] $op $b.0[1], $a.0[2] $op $b.0[2], $a.0[3] $op $b.0[3]])
    };
}

impl std::ops::Add for V {
    type Output = V;
    #[inline(always)]
    fn add(self, o: V) -> V {
        lanes!(self + o)
    }
}

impl std::ops::Sub for V {
    type Output = V;
    #[inline(always)]
    fn sub(self, o: V) -> V {
        lanes!(self - o)
    }
}

impl std::ops::Mul for V {
    type Output = V;
    #[inline(always)]
    fn mul(self, o: V) -> V {
        lanes!(self * o)
    }
}

/// Complex values of four pixels.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct C {
    pub re: V,
    pub im: V,
}

impl C {
    pub const ZERO: C = C {
        re: V::ZERO,
        im: V::ZERO,
    };

    #[inline(always)]
    fn add(self, o: C) -> C {
        C {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    #[inline(always)]
    fn sub(self, o: C) -> C {
        C {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }

    /// `self * (wr + i wi)`.
    #[inline(always)]
    fn twiddle(self, w: (f32, f32)) -> C {
        let (wr, wi) = (V::splat(w.0), V::splat(w.1));
        C {
            re: self.re * wr - self.im * wi,
            im: self.re * wi + self.im * wr,
        }
    }

    /// `self * c` for a real `c`.
    #[inline(always)]
    fn scale(self, c: f32) -> C {
        let c = V::splat(c);
        C {
            re: self.re * c,
            im: self.im * c,
        }
    }

    /// `-i self`.
    #[inline(always)]
    fn mul_neg_i(self) -> C {
        C {
            re: self.im,
            im: V::ZERO - self.re,
        }
    }
}

// ---------------------------------------------------------------------------
// Butterflies: the forward DFT of 2..5 points
// ---------------------------------------------------------------------------

#[inline(always)]
fn bf2(x: [C; 2]) -> [C; 2] {
    [x[0].add(x[1]), x[0].sub(x[1])]
}

#[inline(always)]
fn bf3(x: [C; 3]) -> [C; 3] {
    let t1 = x[1].add(x[2]);
    let t2 = x[0].sub(t1.scale(0.5));
    let t3 = x[1].sub(x[2]).scale(0.866_025_4); // sqrt(3) / 2
    let it3 = t3.mul_neg_i(); // -i t3
    [x[0].add(t1), t2.add(it3), t2.sub(it3)]
}

#[inline(always)]
fn bf4(x: [C; 4]) -> [C; 4] {
    let (a, b) = (x[0].add(x[2]), x[0].sub(x[2]));
    let (c, d) = (x[1].add(x[3]), x[1].sub(x[3]).mul_neg_i());
    [a.add(c), b.add(d), a.sub(c), b.sub(d)]
}

#[inline(always)]
fn bf5(x: [C; 5]) -> [C; 5] {
    const C1: f32 = 0.309_017; // cos(2 pi / 5)
    const C2: f32 = -0.809_017; // cos(4 pi / 5)
    const S1: f32 = 0.951_056_5; // sin(2 pi / 5)
    const S2: f32 = 0.587_785_25; // sin(4 pi / 5)
    let (a1, a2) = (x[1].add(x[4]), x[2].add(x[3]));
    let (b1, b2) = (x[1].sub(x[4]), x[2].sub(x[3]));
    let m1 = x[0].add(a1.scale(C1)).add(a2.scale(C2));
    let m2 = x[0].add(a1.scale(C2)).add(a2.scale(C1));
    let n1 = b1.scale(S1).add(b2.scale(S2)).mul_neg_i();
    let n2 = b1.scale(S2).sub(b2.scale(S1)).mul_neg_i();
    [
        x[0].add(a1).add(a2),
        m1.add(n1),
        m2.add(n2),
        m2.sub(n2),
        m1.sub(n1),
    ]
}

#[inline(always)]
fn bf<const R: usize>(x: [C; R]) -> [C; R] {
    let mut y = [C::ZERO; R];
    match R {
        2 => y.copy_from_slice(&bf2([x[0], x[1]])),
        3 => y.copy_from_slice(&bf3([x[0], x[1], x[2]])),
        4 => y.copy_from_slice(&bf4([x[0], x[1], x[2], x[3]])),
        _ => y.copy_from_slice(&bf5([x[0], x[1], x[2], x[3], x[4]])),
    }
    y
}

// ---------------------------------------------------------------------------
// The transform
// ---------------------------------------------------------------------------

struct Stage {
    radix: usize,
    s: usize,            // stride: the product of the radices before this stage
    m: usize,            // points per sub-transform / radix
    tw: Vec<(f32, f32)>, // `m (radix - 1)` twiddles `exp(-2 pi i p j / (radix m))`
}

/// One decimation-in-frequency Stockham pass: `out[q + s (R p + j)] = w^{p j} DFT_R(in[q + s (p + m .)])_j`.
fn pass<const R: usize>(st: &Stage, input: &[C], output: &mut [C]) {
    let (s, m) = (st.s, st.m);
    assert!(input.len() >= s * m * R && output.len() >= s * m * R);
    for p in 0..m {
        let tw = &st.tw[p * (R - 1)..(p + 1) * (R - 1)];
        for q in 0..s {
            let mut x = [C::ZERO; R];
            for (j, xj) in x.iter_mut().enumerate() {
                // SAFETY: q + s (p + m j) < s m R, checked above
                *xj = unsafe { *input.get_unchecked(q + s * (p + m * j)) };
            }
            let mut y = bf::<R>(x);
            if p > 0 {
                for j in 1..R {
                    y[j] = y[j].twiddle(tw[j - 1]);
                }
            }
            for (j, yj) in y.iter().enumerate() {
                // SAFETY: q + s (R p + j) < s m R
                unsafe { *output.get_unchecked_mut(q + s * (R * p + j)) = *yj };
            }
        }
    }
}

/// Radices of an FFT of `n` points, or `None` if `n` has a prime factor above 5.
fn radices(mut n: usize) -> Option<Vec<usize>> {
    let mut r = Vec::new();
    for f in [4, 3, 5, 2] {
        while n % f == 0 {
            r.push(f);
            n /= f;
        }
    }
    (n == 1).then_some(r)
}

/// The tables for the noise of a `t`-frame tile over the bins `lo..hi`.
pub struct NoisePlan {
    t: usize,
    m: usize, // t / 2
    lo: usize,
    hi: usize,
    stages: Vec<Stage>,
    post: Vec<(f32, f32)>, // exp(-2 pi i k / t) for k in lo..hi
}

/// Bins whose mantissas are multiplied before one logarithm is taken.
const MANT_BINS: usize = 64;

/// Working memory of `NoisePlan::sums`.
pub struct Work {
    a: Vec<C>,
    b: Vec<C>,
    terms: Vec<f32>, // (hi - lo, 4)
}

impl NoisePlan {
    /// `None` unless `t` is even, `t / 2` is 5-smooth, and `1 <= lo < hi <= t / 2`.
    pub fn new(t: usize, lo: usize, hi: usize) -> Option<NoisePlan> {
        if t < 4 || t % 2 != 0 || lo < 1 || hi <= lo || hi > t / 2 {
            return None;
        }
        let m = t / 2;
        let mut stages = Vec::new();
        let (mut n, mut s) = (m, 1);
        for r in radices(m)? {
            let sub = n / r;
            let mut tw = Vec::with_capacity(sub * (r - 1));
            for p in 0..sub {
                for j in 1..r {
                    let a = -TAU * (p * j) as f64 / n as f64;
                    tw.push((a.cos() as f32, a.sin() as f32));
                }
            }
            stages.push(Stage {
                radix: r,
                s,
                m: sub,
                tw,
            });
            n = sub;
            s *= r;
        }
        let post = (lo..hi)
            .map(|k| {
                let a = -TAU * k as f64 / t as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        Some(NoisePlan {
            t,
            m,
            lo,
            hi,
            stages,
            post,
        })
    }

    pub fn work(&self) -> Work {
        Work {
            a: vec![C::ZERO; self.m],
            b: vec![C::ZERO; self.m],
            terms: vec![0.0; (self.hi - self.lo) * 4],
        }
    }

    /// Number of frames.
    pub fn t(&self) -> usize {
        self.t
    }

    /// Half spectrum of four real signals: `x(i, l)` is sample `i` of lane `l`. Returns
    /// `Z[0..m]` of `z[n] = x[2n] + i x[2n + 1]` in `work` (which buffer is in the result).
    fn fft<'a>(&self, a: &'a mut [C], b: &'a mut [C]) -> &'a [C] {
        let (mut src, mut dst) = (a, b);
        for st in &self.stages {
            match st.radix {
                2 => pass::<2>(st, src, dst),
                3 => pass::<3>(st, src, dst),
                4 => pass::<4>(st, src, dst),
                _ => pass::<5>(st, src, dst),
            }
            std::mem::swap(&mut src, &mut dst);
        }
        src
    }

    /// For the four pixels whose frames are `frames` (`(t, 4)` row-major float32, one lane
    /// per pixel): per pixel the sum of `ln(|X_k|^2 / t + eps)` over `lo..hi`.
    pub fn sums(&self, frames: &[f32], w: &mut Work) -> [f64; 4] {
        debug_assert_eq!(frames.len(), self.t * 4);
        for (z, pair) in w.a.iter_mut().zip(frames.chunks_exact(8)) {
            z.re = V([pair[0], pair[1], pair[2], pair[3]]);
            z.im = V([pair[4], pair[5], pair[6], pair[7]]);
        }
        self.finish(w)
    }

    /// `sums` for four pixels stored as bytes: lane `l` of frame `i` is `rows[i * stride + off + l]`
    /// (`rows` holds `t` rows), converted to float32 on the way in.
    pub fn sums_u8(&self, rows: &[u8], stride: usize, off: usize, w: &mut Work) -> [f64; 4] {
        let row = |i: usize| {
            let r = &rows[i * stride + off..i * stride + off + 4];
            V([f32::from(r[0]), f32::from(r[1]), f32::from(r[2]), f32::from(r[3])])
        };
        for (n, z) in w.a.iter_mut().enumerate() {
            z.re = row(2 * n);
            z.im = row(2 * n + 1);
        }
        self.finish(w)
    }

    /// The transform of the first `t / 2` points of `w.a` and the noise sums over its bins.
    fn finish(&self, w: &mut Work) -> [f64; 4] {
        let Work { a, b, terms } = w;
        let m = self.m;
        let zs = {
            let _s = prof::span("bfft_passes");
            self.fft(a, b)
        };
        let _s = prof::span("bfft_tail");
        let inv_t = V::splat(1.0 / self.t as f32);
        let eps = V::splat(f32::EPSILON);
        let half = V::splat(0.5);
        // the arguments of the logarithms first, so that the logarithm itself runs as a
        // plain loop over a slice (which vectorises), then the in-order sums
        for ((k, &(wr, wi)), out) in (self.lo..self.hi).zip(&self.post).zip(terms.chunks_exact_mut(4)) {
            let (a, b) = (zs[k], zs[m - k]); // Z[k], Z[m - k]
            let (wr, wi) = (V::splat(wr), V::splat(wi));
            // E = (Z[k] + conj Z[m - k]) / 2, O = (Z[k] - conj Z[m - k]) / (2 i), X = E + w O
            let er = half * (a.re + b.re);
            let ei = half * (a.im - b.im);
            let or = half * (a.im + b.im);
            let oi = half * (b.re - a.re);
            let xr = er + (wr * or - wi * oi);
            let xi = ei + (wr * oi + wi * or);
            out.copy_from_slice(&((xr * xr + xi * xi) * inv_t + eps).0);
        }
        // sum of ln v = ln 2 sum e + ln prod m per lane, v = m 2^e with m in [1, 2): the
        // mantissas of up to MANT_BINS bins are multiplied (below 2^MANT_BINS, in range) and
        // one logarithm is taken of the product, instead of one per bin
        let mut acc = [0f64; 4];
        for chunk in terms.chunks(MANT_BINS * 4) {
            let mut prod = V::splat(1.0);
            let mut exp = [0i32; 4];
            for t in chunk.chunks_exact(4) {
                for l in 0..4 {
                    let bits = t[l].to_bits();
                    exp[l] += (bits >> 23) as i32 - 127;
                    prod.0[l] *= f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
                }
            }
            for l in 0..4 {
                acc[l] += f64::from(ln_f32(prod.0[l])) + f64::from(exp[l]) * std::f64::consts::LN_2;
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use realfft::RealFftPlanner;

    fn signal(t: usize, lane: usize) -> Vec<f32> {
        let mut s = 99u64 + lane as u64;
        (0..t)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                100.0 + (s >> 56) as f32 * 0.3 + 5.0 * (i as f32 * 0.01).sin()
            })
            .collect()
    }

    #[test]
    fn matches_realfft_on_smooth_lengths() {
        for t in [3000usize, 1000, 600, 100, 90, 250, 30, 4, 6, 10, 12] {
            let (lo, hi) = (((0.06 * t as f64).round() as usize).max(1), t / 2);
            let Some(plan) = NoisePlan::new(t, lo, hi) else {
                panic!("{t}");
            };
            let sigs: Vec<Vec<f32>> = (0..4).map(|l| signal(t, l)).collect();
            let mut frames = vec![0f32; t * 4];
            for i in 0..t {
                for l in 0..4 {
                    frames[i * 4 + l] = sigs[l][i];
                }
            }
            let got = plan.sums(&frames, &mut plan.work());
            let mut planner = RealFftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(t);
            for l in 0..4 {
                let mut inp = sigs[l].clone();
                let mut spec = fft.make_output_vec();
                fft.process(&mut inp, &mut spec).unwrap();
                let want: f64 = spec[lo..hi]
                    .iter()
                    .map(|c| f64::from(((c.re * c.re + c.im * c.im) / t as f32 + f32::EPSILON).ln()))
                    .sum();
                let rel = (got[l] - want).abs() / want.abs().max(1.0);
                assert!(rel < 2e-5, "t {t} lane {l}: {} vs {want} ({rel})", got[l]);
            }
        }
    }

    #[test]
    fn refuses_other_lengths() {
        assert!(NoisePlan::new(3002, 100, 1500).is_none()); // 1501 = 19 * 79
        assert!(NoisePlan::new(3001, 100, 1500).is_none());
        assert!(NoisePlan::new(3000, 100, 1501).is_none());
    }
}
