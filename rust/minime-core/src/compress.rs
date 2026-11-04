//! Pass 2 of `minime.compress` on one tile: per-pixel mean, energy and noise, the
//! streamed randomized SVD (Halko, `n_iter` power iterations) and the lag-1
//! autocorrelation rank cut, doubling the rank while the last components still pass.
//!
//! A tile is held as `(T, P)` uint8 and used in float32 chunks of `CHUNK` frames (mean
//! removed). The products with `Y` are float32 GEMMs (`matrixmultiply`); QR, the small
//! SVD and the autocorrelations run in float64. The Gaussian test matrices come from a
//! `Normals` source, so a check can replay the draws numpy made (`ref_pass2.py`).

use crate::bfft::NoisePlan;
use crate::fastmath::ln_f32;
use crate::linalg::sym_eig;
use crate::npy;
use crate::prof;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use serde::Deserialize;

pub const N_OVERSAMPLES: usize = 10;
pub const MAX_RANK: usize = 300;
pub const ADAPT_MARGIN: usize = 10;
pub const NOISE_SEG: usize = 1000;
pub const CHUNK: usize = 512;

/// `ref_pass2.py`'s `pass2.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Pass2Params {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub tile: usize,
    pub svd_k0: usize,
    pub svd_power_iter: usize,
    pub rank_rho_min: f64,
    pub rank_z: f64,
    pub noise_freq: f64,
    pub noise_frames: usize,
}

// ---------------------------------------------------------------------------
// Gaussian draws
// ---------------------------------------------------------------------------

/// Source of standard normal float32 values, consumed in draw order.
pub trait Normals {
    fn draw(&mut self, n: usize) -> Result<Vec<f32>, String>;
}

/// Replays recorded draws (one flat array, as `ref_pass2.py` writes `omega_NNN.npy`).
pub struct Recorded {
    data: Vec<f32>,
    pos: usize,
}

impl Recorded {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data, pos: 0 }
    }
}

impl Normals for Recorded {
    fn draw(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(format!("{} recorded draws, {end} needed", self.data.len()));
        }
        let out = self.data[self.pos..end].to_vec();
        self.pos = end;
        Ok(out)
    }
}

/// xoshiro128++ with Box-Muller: Rust's own draws when nothing is replayed.
pub struct Gaussian {
    s: [u32; 4],
}

impl Gaussian {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            // splitmix64
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        let (a, b) = (next(), next());
        Self {
            s: [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32],
        }
    }

    fn next_u32(&mut self) -> u32 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(7).wrapping_add(s[0]);
        let t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(11);
        out
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        (f64::from(self.next_u32() >> 8) + 1.0) / 16_777_216.0
    }
}

impl Normals for Gaussian {
    fn draw(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let r = (-2.0 * self.uniform().ln()).sqrt();
            let th = std::f64::consts::TAU * self.uniform();
            out.push((r * th.cos()) as f32);
            out.push((r * th.sin()) as f32);
        }
        out.truncate(n);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Streamed tile matrix
// ---------------------------------------------------------------------------

/// `Y` (P x T, mean removed) of one tile, held as `(T, P)` uint8.
pub struct TileMatrix<'a> {
    raw: &'a [u8],
    pub t: usize,
    pub p: usize,
    pub mean: Vec<f32>,
    pub energy: Vec<f64>,
}

impl<'a> TileMatrix<'a> {
    pub fn new(raw: &'a [u8], t: usize, p: usize) -> Self {
        let _s = prof::span("mean_energy");
        assert_eq!(raw.len(), t * p);
        // Integer column sums are exact, so `mean` and `energy` equal the float64 sums of
        // the first implementation bit for bit; `sq` is flushed before it can overflow.
        let mut acc = vec![0u32; p]; // 255 t < 2^32 for t < 16.8 M
        let mut sq = vec![0u32; p]; // 65025 rows per flush < 2^32 / 65025 = 66051
        let mut sq64 = vec![0u64; p];
        for (n, row) in raw.chunks_exact(p).enumerate() {
            for ((a, s), &v) in acc.iter_mut().zip(sq.iter_mut()).zip(row) {
                let v = u32::from(v);
                *a += v;
                *s += v * v;
            }
            if (n + 1) % 60_000 == 0 {
                for (w, s) in sq64.iter_mut().zip(sq.iter_mut()) {
                    *w += u64::from(*s);
                    *s = 0;
                }
            }
        }
        let tf = t as f64;
        let mean = acc.iter().map(|&a| (f64::from(a) / tf) as f32).collect();
        let energy = acc
            .iter()
            .zip(sq.iter().zip(&sq64))
            .map(|(&a, (&s, &w))| {
                let a = f64::from(a);
                ((w as f64 + f64::from(s)) - a * a / tf).max(0.0)
            })
            .collect();
        Self {
            raw,
            t,
            p,
            mean,
            energy,
        }
    }

    /// Frames `c0..c0 + n` as `(n, P)` row-major float32, mean removed.
    fn block(&self, c0: usize, n: usize, buf: &mut Vec<f32>) {
        let _s = prof::span("block_u8_f32");
        buf.clear();
        for row in self.raw[c0 * self.p..(c0 + n) * self.p].chunks_exact(self.p) {
            buf.extend(row.iter().zip(&self.mean).map(|(&v, &m)| f32::from(v) - m));
        }
    }

    /// `Y^T Q`: `(T, ell)` row-major from `q` `(P, ell)` row-major.
    pub fn t_matmul(&self, q: &[f32], ell: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.t * ell];
        let mut x = Vec::new();
        for c0 in (0..self.t).step_by(CHUNK) {
            let n = CHUNK.min(self.t - c0);
            self.block(c0, n, &mut x);
            let c = &mut out[c0 * ell..(c0 + n) * ell];
            // SAFETY: the strides describe the row-major buffers above
            let _s = prof::span("gemm_YtQ");
            unsafe {
                matrixmultiply::sgemm(
                    n,
                    self.p,
                    ell,
                    1.0,
                    x.as_ptr(),
                    self.p as isize,
                    1,
                    q.as_ptr(),
                    ell as isize,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    ell as isize,
                    1,
                );
            }
        }
        out
    }

    /// `Y Z`: `(P, ell)` row-major from `z` `(T, ell)` row-major.
    pub fn matmul(&self, z: &[f32], ell: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.p * ell];
        let mut x = Vec::new();
        for c0 in (0..self.t).step_by(CHUNK) {
            let n = CHUNK.min(self.t - c0);
            self.block(c0, n, &mut x);
            let zc = &z[c0 * ell..(c0 + n) * ell];
            // SAFETY: x^T is x with swapped strides; the rest is row-major as above
            let _s = prof::span("gemm_YZ");
            unsafe {
                matrixmultiply::sgemm(
                    self.p,
                    n,
                    ell,
                    1.0,
                    x.as_ptr(),
                    1,
                    self.p as isize,
                    zc.as_ptr(),
                    ell as isize,
                    1,
                    1.0,
                    out.as_mut_ptr(),
                    ell as isize,
                    1,
                );
            }
        }
        out
    }

    /// Per-pixel noise std from the power spectrum above `noise_freq` (`TileMatrix.noise`):
    /// the square root of the geometric mean of `|rfft(y)|^2 / T` over bins
    /// `[round(noise_freq T), round(T / 2))`, in float32 as numpy does it. With
    /// `frames > 0` (and fewer than `T`) the bins are pooled over `frames / NOISE_SEG`
    /// segments spread over the tile.
    pub fn noise(&self, noise_freq: f64, frames: usize) -> Vec<f64> {
        let _s = prof::span("noise");
        let n_seg = frames / NOISE_SEG;
        if n_seg >= 1 && n_seg * NOISE_SEG < self.t {
            return self.noise_segments(noise_freq, n_seg);
        }
        let lo = ((noise_freq * self.t as f64).round_ties_even() as usize).max(1);
        let hi = (0.5 * self.t as f64).round_ties_even() as usize;
        if hi <= lo {
            return self.noise_mad();
        }
        if let Some(plan) = NoisePlan::new(self.t, lo, hi) {
            return self.noise_batched(&plan, hi - lo);
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.t);
        let mut spec = fft.make_output_vec();
        let mut scratch = fft.make_scratch_vec();
        let (tf, nb) = (self.t as f32, hi - lo);
        let mut cols = vec![0f32; NOISE_GROUP * self.t];
        let mut terms = vec![0f32; NOISE_GROUP * nb];
        let mut out = Vec::with_capacity(self.p);
        for px0 in (0..self.p).step_by(NOISE_GROUP) {
            let n = NOISE_GROUP.min(self.p - px0);
            self.columns(px0, n, 0, self.t, &mut cols);
            for g in 0..n {
                {
                    let _s = prof::span("noise_fft");
                    fft.process_with_scratch(
                        &mut cols[g * self.t..(g + 1) * self.t],
                        &mut spec,
                        &mut scratch,
                    )
                    .expect("fft length");
                }
                let _s = prof::span("noise_log");
                log_terms(&spec[lo..hi], tf, &mut terms[g * nb..(g + 1) * nb]);
            }
            let _s = prof::span("noise_sum");
            // numpy sums the bins of a pixel in order, in float32: four pixels at a time
            // keeps that order per pixel without waiting on one dependent add chain
            let sums = seq_sums_f32(&terms[..n * nb], nb);
            out.extend(sums.iter().map(|&s| f64::from((0.5 * (s / nb as f32)).exp())));
        }
        out
    }

    /// The noise of every pixel with `bfft`'s four-pixel FFT: groups of 16 pixels are copied
    /// out of the tile once (so each frame row is read contiguously), then run four at a time.
    fn noise_batched(&self, plan: &NoisePlan, nb: usize) -> Vec<f64> {
        let (t, p) = (self.t, self.p);
        let mut work = plan.work();
        let mut blk = vec![0u8; t * 16];
        let mut out = Vec::with_capacity(p);
        for px0 in (0..p).step_by(16) {
            let n = 16.min(p - px0);
            {
                let _s = prof::span("noise_gather");
                for (row, dst) in self.raw.chunks_exact(p).zip(blk.chunks_exact_mut(16)) {
                    dst[..n].copy_from_slice(&row[px0..px0 + n]);
                    dst[n..].fill(0);
                }
            }
            for g in (0..n).step_by(4) {
                let _s = prof::span("noise_fft");
                let sums = plan.sums_u8(&blk, 16, g, &mut work);
                out.extend(
                    sums.iter()
                        .take(n - g)
                        .map(|&s| f64::from((0.5 * (s / nb as f32)).exp())),
                );
            }
        }
        out
    }

    /// Pixels `px0..px0 + n` of frames `t0..t0 + len` as float32 rows of `len` (`dst[g * len + i]`):
    /// the transpose a per-pixel FFT needs, done for a group of pixels so that every frame
    /// row is read once, contiguously.
    fn columns(&self, px0: usize, n: usize, t0: usize, len: usize, dst: &mut [f32]) {
        let _s = prof::span("noise_gather");
        for i in 0..len {
            let row = &self.raw[(t0 + i) * self.p + px0..(t0 + i) * self.p + px0 + n];
            for (g, &v) in row.iter().enumerate() {
                dst[g * len + i] = f32::from(v);
            }
        }
    }

    fn noise_segments(&self, noise_freq: f64, n_seg: usize) -> Vec<f64> {
        let l = NOISE_SEG;
        let lo = ((noise_freq * l as f64).round_ties_even() as usize).max(1);
        let hi = (0.5 * l as f64).round_ties_even() as usize;
        let span = (self.t - l) as f64;
        let starts: Vec<usize> = (0..n_seg)
            .map(|i| match n_seg {
                1 => 0.0,
                _ if i == n_seg - 1 => span,
                _ => i as f64 * (span / (n_seg - 1) as f64),
            })
            .map(|v| v.round_ties_even() as usize)
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(l);
        let mut spec = fft.make_output_vec();
        let mut scratch = fft.make_scratch_vec();
        let (lf, nb) = (l as f32, hi - lo);
        let mut cols = vec![0f32; NOISE_GROUP * l];
        let mut terms = vec![0f32; nb];
        let mut out = vec![0f64; self.p];
        for px0 in (0..self.p).step_by(NOISE_GROUP) {
            let n = NOISE_GROUP.min(self.p - px0);
            let mut acc = [0f64; NOISE_GROUP];
            for &t0 in &starts {
                self.columns(px0, n, t0, l, &mut cols);
                for (g, a) in acc.iter_mut().enumerate().take(n) {
                    fft.process_with_scratch(&mut cols[g * l..(g + 1) * l], &mut spec, &mut scratch)
                        .expect("fft length");
                    log_terms(&spec[lo..hi], lf, &mut terms);
                    let mut s = 0f64;
                    for &v in &terms {
                        s += f64::from(v);
                    }
                    *a += s / nb as f64;
                }
            }
            for (g, a) in acc.iter().enumerate().take(n) {
                out[px0 + g] = (0.5 * a / n_seg as f64).exp();
            }
        }
        out
    }

    /// Too few frames for a spectrum: `1.4826 median |dy| / sqrt 2`.
    fn noise_mad(&self) -> Vec<f64> {
        if self.t < 2 {
            return vec![0.0; self.p];
        }
        (0..self.p)
            .map(|px| {
                let mut d: Vec<f32> = (1..self.t)
                    .map(|ti| {
                        (f32::from(self.raw[ti * self.p + px]) - f32::from(self.raw[(ti - 1) * self.p + px]))
                            .abs()
                    })
                    .collect();
                d.sort_by(f32::total_cmp);
                let n = d.len();
                let med = if n % 2 == 1 {
                    d[n / 2]
                } else {
                    (d[n / 2 - 1] + d[n / 2]) / 2.0
                };
                f64::from(1.4826f32 * med) / std::f64::consts::SQRT_2
            })
            .collect()
    }
}

/// Pixels per group of the noise stage's transposed gather.
const NOISE_GROUP: usize = 16;

/// `ln(|X|^2 / len + eps)` of each bin, float32 (`|X|^2` as `re^2 + im^2`).
fn log_terms(spec: &[Complex<f32>], len: f32, out: &mut [f32]) {
    for (o, c) in out.iter_mut().zip(spec) {
        *o = ln_f32((c.re * c.re + c.im * c.im) / len + f32::EPSILON);
    }
}

/// In-order float32 sum of each row of `terms` (`(n, nb)`), `n <= NOISE_GROUP`.
fn seq_sums_f32(terms: &[f32], nb: usize) -> Vec<f32> {
    let n = terms.len() / nb;
    let mut out = vec![0f32; n];
    let mut g = 0;
    while g + 4 <= n {
        let r = |i: usize| &terms[(g + i) * nb..(g + i + 1) * nb];
        let (a, b, c, d) = (r(0), r(1), r(2), r(3));
        let mut s = [0f32; 4];
        for i in 0..nb {
            s[0] += a[i];
            s[1] += b[i];
            s[2] += c[i];
            s[3] += d[i];
        }
        out[g..g + 4].copy_from_slice(&s);
        g += 4;
    }
    for g in g..n {
        out[g] = terms[g * nb..(g + 1) * nb].iter().fold(0f32, |s, &v| s + v);
    }
    out
}

// ---------------------------------------------------------------------------
// Randomized SVD and rank cut
// ---------------------------------------------------------------------------

/// Columns of a register block in the QR's trailing updates.
const QR_W: usize = 8;

/// `acc[k] = sum_{i > j} v_i t[i][col0 + k]` for `k < W` over the row-major `(m, n)` matrix
/// `t`: one pass over the rows with `W` sums in registers. Every sum runs in row order
/// from zero with unfused products, so it equals a plain dot product, bit for bit.
#[inline(always)]
fn dot_cols<const W: usize>(t: &[f64], n: usize, j: usize, v: &[f64], col0: usize, acc: &mut [f64]) {
    let mut s = [-0.0f64; W]; // `Iterator::sum` starts at -0.0
    for (row, &vi) in t[(j + 1) * n..].chunks_exact(n).zip(v) {
        let x = &row[col0..col0 + W];
        for k in 0..W {
            s[k] += vi * x[k];
        }
    }
    acc[..W].copy_from_slice(&s);
}

/// `t[i][col0 + k] -= f[k] * v_i` for `k < W` and the rows `i > j`.
#[inline(always)]
fn axpy_cols<const W: usize>(t: &mut [f64], n: usize, j: usize, v: &[f64], col0: usize, f: &[f64]) {
    let mut fv = [0f64; W];
    fv.copy_from_slice(&f[..W]);
    for (row, &vi) in t[(j + 1) * n..].chunks_exact_mut(n).zip(v) {
        let x = &mut row[col0..col0 + W];
        for k in 0..W {
            x[k] -= fv[k] * vi;
        }
    }
}

/// Apply the reflector `H = I - tau [1; v] [1; v]^T` (row `j` is its implicit 1) to the
/// columns `cols` of the row-major `(m, n)` matrix `t`: `d = t[j][k] + v . t[j + 1.., k]`,
/// `f = tau d`, `t[j][k] -= f`, `t[i][k] -= f v_i`.
fn reflect(t: &mut [f64], n: usize, j: usize, v: &[f64], tau: f64, cols: std::ops::Range<usize>) {
    let mut acc = [0f64; QR_W];
    let mut col0 = cols.start;
    while col0 < cols.end {
        let rem = cols.end - col0;
        let w = match rem {
            8.. => 8,
            4..=7 => 4,
            2..=3 => 2,
            _ => 1,
        };
        match w {
            8 => dot_cols::<8>(t, n, j, v, col0, &mut acc),
            4 => dot_cols::<4>(t, n, j, v, col0, &mut acc),
            2 => dot_cols::<2>(t, n, j, v, col0, &mut acc),
            _ => dot_cols::<1>(t, n, j, v, col0, &mut acc),
        }
        for k in 0..w {
            let d = t[j * n + col0 + k] + acc[k];
            acc[k] = tau * d; // f
            t[j * n + col0 + k] -= acc[k];
        }
        match w {
            8 => axpy_cols::<8>(t, n, j, v, col0, &acc),
            4 => axpy_cols::<4>(t, n, j, v, col0, &acc),
            2 => axpy_cols::<2>(t, n, j, v, col0, &acc),
            _ => axpy_cols::<1>(t, n, j, v, col0, &acc),
        }
        col0 += w;
    }
}

/// Orthonormal basis (economic Householder QR, LAPACK's sign convention) of the
/// columns of `a` `(m, n)` row-major, as `(m, n)` row-major float32. Float64 throughout;
/// the operations and their order are those of a column-by-column implementation, with
/// the sums of all trailing columns running side by side.
pub fn qr_q(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let _s = prof::span("qr");
    let mut c: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect(); // (m, n) row-major
    let r = n.min(m);
    let mut tau = vec![0f64; n];
    let mut v = vec![0f64; m];
    for j in 0..r {
        let alpha = c[j * n + j];
        let xnorm = (j + 1..m)
            .map(|i| c[i * n + j])
            .fold(-0.0, |s, x| s + x * x)
            .sqrt();
        if xnorm == 0.0 {
            continue; // H = I
        }
        let beta = -alpha.signum() * alpha.hypot(xnorm);
        let scale = 1.0 / (alpha - beta);
        for (i, vi) in v[..m - j - 1].iter_mut().enumerate() {
            let x = &mut c[(j + 1 + i) * n + j];
            *x *= scale;
            *vi = *x;
        }
        tau[j] = (beta - alpha) / beta;
        c[j * n + j] = beta;
        reflect(&mut c, n, j, &v[..m - j - 1], tau[j], j + 1..n);
    }
    // Q = H_0 ... H_{n-1} I[:, :n], applied from the last reflector back
    let mut q = vec![0f64; m * n];
    for j in 0..r {
        q[j * n + j] = 1.0;
    }
    for j in (0..r).rev() {
        if tau[j] == 0.0 {
            continue;
        }
        for (i, vi) in v[..m - j - 1].iter_mut().enumerate() {
            *vi = c[(j + 1 + i) * n + j];
        }
        reflect(&mut q, n, j, &v[..m - j - 1], tau[j], j..n);
    }
    q.iter().map(|&x| x as f32).collect()
}

/// Rank-`k` factors of a randomized SVD: `u` `(P, k)`, `s` `(k)`, `vt` `(k, T)`.
pub struct Svd {
    pub k: usize,
    pub u: Vec<f32>,
    pub s: Vec<f64>,
    pub vt: Vec<f32>,
}

/// `compress.rsvd_streamed`: Halko randomized SVD with every product streamed over chunks.
pub fn rsvd_streamed(y: &TileMatrix, k: usize, n_iter: usize, rng: &mut dyn Normals) -> Result<Svd, String> {
    let (p, t) = (y.p, y.t);
    let ell = k + N_OVERSAMPLES;
    let omega = rng.draw(t * ell)?;
    let mut q = qr_q(&y.matmul(&omega, ell), p, ell);
    for _ in 0..n_iter {
        let z = qr_q(&y.t_matmul(&q, ell), t, ell);
        q = qr_q(&y.matmul(&z, ell), p, ell);
    }
    let bt = y.t_matmul(&q, ell); // (T, ell) = (Q^T Y)^T
    let _s = prof::span("small_svd");
    // SVD of B = bt^T from the eigenvectors of B B^T, in float64
    let mut g = vec![0f64; ell * ell];
    let mut rowf = vec![0f64; ell];
    for row in bt.chunks_exact(ell) {
        for (d, &v) in rowf.iter_mut().zip(row) {
            *d = f64::from(v);
        }
        for a in 0..ell {
            let ra = rowf[a];
            for (gv, &rb) in g[a * ell + a..(a + 1) * ell].iter_mut().zip(&rowf[a..]) {
                *gv += ra * rb;
            }
        }
    }
    for a in 0..ell {
        for b in 0..a {
            g[a * ell + b] = g[b * ell + a];
        }
    }
    let (evals, evecs) = sym_eig(&g, ell);
    let order: Vec<usize> = (0..ell).rev().take(k).collect(); // descending
    let s: Vec<f64> = order.iter().map(|&j| evals[j].max(0.0).sqrt()).collect();
    // the kept eigenvectors as (ell, k); every output is a sum over `a` in order, the `k`
    // outputs of a row side by side
    let mut e2 = vec![0f64; ell * k];
    for a in 0..ell {
        for (i, &j) in order.iter().enumerate() {
            e2[a * k + i] = evecs[a * ell + j];
        }
    }
    let project = |row: &[f32], d: &mut [f64]| {
        d.fill(-0.0);
        for (&ra, e) in row.iter().zip(e2.chunks_exact(k)) {
            let ra = f64::from(ra);
            for (dv, &ev) in d.iter_mut().zip(e) {
                *dv += ra * ev;
            }
        }
    };
    let mut d = vec![0f64; k];
    let mut vt = vec![0f32; k * t];
    for (ti, row) in bt.chunks_exact(ell).enumerate() {
        project(row, &mut d);
        for (i, &dv) in d.iter().enumerate() {
            if s[i] != 0.0 {
                vt[i * t + ti] = (dv / s[i]) as f32;
            }
        }
    }
    let mut u = vec![0f32; p * k];
    for (qr, urow) in q.chunks_exact(ell).zip(u.chunks_exact_mut(k)) {
        project(qr, &mut d);
        for (o, &dv) in urow.iter_mut().zip(&d) {
            *o = dv as f32;
        }
    }
    Ok(Svd { k, u, s, vt })
}

/// Lag-1 autocorrelation of each row of `vt` `(k, T)`.
pub fn lag1_autocorr(vt: &[f32], t: usize) -> Vec<f64> {
    vt.chunks_exact(t)
        .map(|row| {
            let m = row.iter().map(|&v| f64::from(v)).sum::<f64>() / t as f64;
            let v: Vec<f64> = row.iter().map(|&x| f64::from(x) - m).collect();
            let num: f64 = v[1..].iter().zip(&v[..t - 1]).map(|(a, b)| a * b).sum();
            let den: f64 = v.iter().map(|a| a * a).sum();
            num / den.max(1e-30)
        })
        .collect()
}

/// `compress.autocorr_keep`: components whose time course is not (mostly) white noise.
pub fn autocorr_keep(vt: &[f32], t: usize, z: f64, rho_min: f64) -> Vec<bool> {
    let thr = (z / (t as f64).sqrt()).max(rho_min);
    lag1_autocorr(vt, t).iter().map(|&r| r > thr).collect()
}

// ---------------------------------------------------------------------------
// One tile
// ---------------------------------------------------------------------------

/// `factors.TileFactors` of one tile, plus the last rSVD rank `k`.
#[derive(Clone)]
pub struct TileFactors {
    pub w: Vec<f32>, // (P, r) row-major: left factors scaled by singular values
    pub v: Vec<f32>, // (T, r) row-major: unit-norm temporal components
    pub s: Vec<f32>, // (r) kept singular values
    pub mean: Vec<f32>,
    pub sigma: Vec<f32>,
    pub energy: Vec<f64>,
    pub rank: usize,
    pub k: usize,
}

/// `compress.compress_tile` on `raw` `(T, P)` uint8: start at rank `svd_k0` and double it
/// (up to `min(MAX_RANK, P - 1, T - 1)`) while any of the last `ADAPT_MARGIN` components
/// still passes the autocorrelation cut; keep component 0 if none passes.
pub fn compress_tile(
    raw: &[u8],
    t: usize,
    p: usize,
    pr: &Pass2Params,
    rng: &mut dyn Normals,
) -> Result<TileFactors, String> {
    let y = TileMatrix::new(raw, t, p);
    let sigma: Vec<f32> = y
        .noise(pr.noise_freq, pr.noise_frames)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let k_cap = MAX_RANK.min(p.saturating_sub(1)).min(t.saturating_sub(1)).max(1);
    let mut k = pr.svd_k0.min(k_cap);
    let (svd, mut keep) = loop {
        let svd = rsvd_streamed(&y, k, pr.svd_power_iter, rng)?;
        let keep = autocorr_keep(&svd.vt, t, pr.rank_z, pr.rank_rho_min);
        if k >= k_cap || !keep[k.saturating_sub(ADAPT_MARGIN)..].iter().any(|&b| b) {
            break (svd, keep);
        }
        k = (2 * k).min(k_cap);
    };
    if !keep.iter().any(|&b| b) {
        keep[0] = true;
    }
    let idx: Vec<usize> = (0..k).filter(|&i| keep[i]).collect();
    let r = idx.len();
    let mut w = vec![0f32; p * r];
    for px in 0..p {
        for (c, &i) in idx.iter().enumerate() {
            w[px * r + c] = (f64::from(svd.u[px * k + i]) * svd.s[i]) as f32;
        }
    }
    let mut v = vec![0f32; t * r];
    for ti in 0..t {
        for (c, &i) in idx.iter().enumerate() {
            v[ti * r + c] = svd.vt[i * t + ti];
        }
    }
    Ok(TileFactors {
        w,
        v,
        s: idx.iter().map(|&i| svd.s[i] as f32).collect(),
        mean: y.mean,
        sigma,
        energy: y.energy,
        rank: r,
        k,
    })
}

/// The tiles' factors as `.npy` files named like `ref_pass2.py`'s `ref/` (plus
/// `summary.json`): flat concatenations in grid order, and `rank` and `k` per tile.
pub fn outputs(tiles: &[TileFactors], summary: serde_json::Value) -> Vec<(String, Vec<u8>)> {
    fn flat<T: npy::NpyElem>(tiles: &[TileFactors], f: impl Fn(&TileFactors) -> &[T]) -> Vec<T> {
        tiles.iter().flat_map(|tf| f(tf).iter().copied()).collect()
    }
    let mut out = Vec::new();
    let w = flat(tiles, |tf| &tf.w);
    out.push(("W.npy".into(), npy::write(&[w.len()], &w)));
    let v = flat(tiles, |tf| &tf.v);
    out.push(("V.npy".into(), npy::write(&[v.len()], &v)));
    let s = flat(tiles, |tf| &tf.s);
    out.push(("s.npy".into(), npy::write(&[s.len()], &s)));
    let m = flat(tiles, |tf| &tf.mean);
    out.push(("mean.npy".into(), npy::write(&[m.len()], &m)));
    let sg = flat(tiles, |tf| &tf.sigma);
    out.push(("sigma.npy".into(), npy::write(&[sg.len()], &sg)));
    let e = flat(tiles, |tf| &tf.energy);
    out.push(("energy.npy".into(), npy::write(&[e.len()], &e)));
    let rank: Vec<i64> = tiles.iter().map(|tf| tf.rank as i64).collect();
    out.push(("rank.npy".into(), npy::write(&[rank.len()], &rank)));
    let k: Vec<i64> = tiles.iter().map(|tf| tf.k as i64).collect();
    out.push(("k.npy".into(), npy::write(&[k.len()], &k)));
    out.push((
        "summary.json".into(),
        serde_json::to_vec_pretty(&summary).unwrap(),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(k0: usize) -> Pass2Params {
        Pass2Params {
            h: 0,
            w: 0,
            t: 0,
            tile: 64,
            svd_k0: k0,
            svd_power_iter: 1,
            rank_rho_min: 0.2,
            rank_z: 4.0,
            noise_freq: 0.06,
            noise_frames: 0,
        }
    }

    #[test]
    fn qr_is_orthonormal_and_spans() {
        let (m, n) = (40, 6);
        let a = Gaussian::new(1).draw(m * n).unwrap();
        let q = qr_q(&a, m, n);
        for i in 0..n {
            for j in 0..n {
                let d: f64 = (0..m)
                    .map(|r| f64::from(q[r * n + i]) * f64::from(q[r * n + j]))
                    .sum();
                assert!((d - f64::from(u8::from(i == j))).abs() < 1e-6);
            }
        }
        // a = Q (Q^T a)
        for j in 0..n {
            let c: Vec<f64> = (0..n)
                .map(|i| (0..m).map(|r| f64::from(q[r * n + i] * a[r * n + j])).sum())
                .collect();
            for r in 0..m {
                let rec: f64 = (0..n).map(|i| f64::from(q[r * n + i]) * c[i]).sum();
                assert!((rec - f64::from(a[r * n + j])).abs() < 1e-4);
            }
        }
    }

    /// Slow low-rank signal plus white noise: the kept rank is the signal's, and the
    /// rank doubles when `k0` is too small for it.
    #[test]
    fn keeps_the_slow_components() {
        let (t, p, r_true) = (600, 120, 12);
        let mut g = Gaussian::new(7);
        let noise = g.draw(t * p).unwrap();
        let spatial = g.draw(r_true * p).unwrap();
        let mut raw = vec![0u8; t * p];
        for ti in 0..t {
            for px in 0..p {
                let mut v = 100.0 + 2.0 * f64::from(noise[ti * p + px]);
                for c in 0..r_true {
                    let cycles = 2.0 * (c + 1) as f64; // orthogonal, below the noise band
                    let ph = (std::f64::consts::TAU * cycles * ti as f64 / t as f64).sin();
                    v += 8.0 * ph * f64::from(spatial[c * p + px]);
                }
                raw[ti * p + px] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        let tf = compress_tile(&raw, t, p, &params(4), &mut Gaussian::new(0)).unwrap();
        assert_eq!(tf.rank, r_true);
        assert!(tf.k >= 16);
        let y = TileMatrix::new(&raw, t, p);
        let sigma = y.noise(0.06, 0);
        let mean_sigma = sigma.iter().sum::<f64>() / p as f64;
        assert!((mean_sigma / 2.0 - 0.75).abs() < 0.1, "{mean_sigma}"); // 0.75 sigma scale
    }

    /// The faster steps against the code they replaced (`compress_ref`, `--features
    /// reference`): QR bit for bit on random matrices, whole tiles on a slow low-rank
    /// signal plus noise.
    #[cfg(feature = "reference")]
    mod against_reference {
        use super::*;
        use crate::compress_ref;

        #[test]
        fn qr_is_bit_identical() {
            for (m, n) in [(40, 6), (300, 50), (4096, 50), (17, 17), (9, 12), (64, 1)] {
                let a = Gaussian::new(3).draw(m * n).unwrap();
                let (q0, q1) = (compress_ref::qr_q(&a, m, n), qr_q(&a, m, n));
                assert!(
                    q0.iter().zip(&q1).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{m}x{n}"
                );
            }
        }

        fn synthetic(t: usize, p: usize, r_true: usize, seed: u64) -> Vec<u8> {
            let mut g = Gaussian::new(seed);
            let noise = g.draw(t * p).unwrap();
            let spatial = g.draw(r_true * p).unwrap();
            let mut raw = vec![0u8; t * p];
            for ti in 0..t {
                for px in 0..p {
                    let mut v = 100.0 + 2.0 * f64::from(noise[ti * p + px]);
                    for c in 0..r_true {
                        let ph = (std::f64::consts::TAU * (2 * (c + 1)) as f64 * ti as f64 / t as f64).sin();
                        v += 8.0 * ph * f64::from(spatial[c * p + px]);
                    }
                    raw[ti * p + px] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
            raw
        }

        #[test]
        fn tile_matches_reference() {
            for (t, p, r, k0, seed) in [
                (600, 120, 12, 4, 7),
                (1000, 64 * 24, 5, 10, 8),
                (300, 77, 3, 20, 9),
            ] {
                let mut raw = synthetic(t, p, r, seed);
                for ti in 0..t {
                    raw[ti * p] = 100; // a constant pixel: its spectrum is rounding noise or zero
                }
                let pr = params(k0);
                let a = compress_ref::compress_tile(&raw, t, p, &pr, &mut Gaussian::new(1)).unwrap();
                let b = compress_tile(&raw, t, p, &pr, &mut Gaussian::new(1)).unwrap();
                assert_eq!((a.rank, a.k), (b.rank, b.k));
                assert_eq!(a.mean, b.mean);
                assert_eq!(a.energy, b.energy);
                for (i, (x, y)) in a.sigma.iter().zip(&b.sigma).enumerate() {
                    // the constant pixel (0) is a sum of identical logarithms in float32, whose
                    // rounding the two sums (and the rounding noise of two FFTs) do not share
                    let tol = if i == 0 { 1e-4 } else { 1e-5 };
                    assert!((x - y).abs() <= tol * x.abs(), "sigma {i}: {x} {y}");
                }
                let rel = |u: &[f32], v: &[f32]| {
                    let num: f64 = u.iter().zip(v).map(|(&x, &y)| f64::from(x - y).powi(2)).sum();
                    (num / u.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()).sqrt()
                };
                assert!(rel(&a.w, &b.w) < 1e-5, "w {}", rel(&a.w, &b.w));
                assert!(rel(&a.v, &b.v) < 1e-5, "v {}", rel(&a.v, &b.v));
            }
        }
    }
}
