//! Pass 2 of `minime.compress` on one tile: per-pixel mean, energy and noise, the
//! streamed randomized SVD (Halko, `n_iter` power iterations) and the lag-1
//! autocorrelation rank cut, doubling the rank while the last components still pass.
//!
//! A tile is held as `(T, P)` uint8 and used in float32 chunks of `CHUNK` frames (mean
//! removed). The products with `Y` are float32 GEMMs (`matrixmultiply`); QR, the small
//! SVD and the autocorrelations run in float64. The Gaussian test matrices come from a
//! `Normals` source, so a check can replay the draws numpy made (`ref_pass2.py`).

use crate::linalg::sym_eig;
use crate::npy;
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
        assert_eq!(raw.len(), t * p);
        let mut acc = vec![0f64; p];
        let mut sq = vec![0f64; p];
        for row in raw.chunks_exact(p) {
            for ((a, s), &v) in acc.iter_mut().zip(sq.iter_mut()).zip(row) {
                let v = f64::from(v);
                *a += v;
                *s += v * v;
            }
        }
        let tf = t as f64;
        let mean = acc.iter().map(|&a| (a / tf) as f32).collect();
        let energy = acc
            .iter()
            .zip(&sq)
            .map(|(&a, &s)| (s - a * a / tf).max(0.0))
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
        let n_seg = frames / NOISE_SEG;
        if n_seg >= 1 && n_seg * NOISE_SEG < self.t {
            return self.noise_segments(noise_freq, n_seg);
        }
        let lo = ((noise_freq * self.t as f64).round_ties_even() as usize).max(1);
        let hi = (0.5 * self.t as f64).round_ties_even() as usize;
        if hi <= lo {
            return self.noise_mad();
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.t);
        let mut inp = fft.make_input_vec();
        let mut spec = fft.make_output_vec();
        let tf = self.t as f32;
        let nb = (hi - lo) as f32;
        (0..self.p)
            .map(|px| {
                for (ti, v) in inp.iter_mut().enumerate() {
                    *v = f32::from(self.raw[ti * self.p + px]);
                }
                fft.process(&mut inp, &mut spec).expect("fft length");
                let mut s = 0f32;
                for c in &spec[lo..hi] {
                    let a = c.re.hypot(c.im);
                    s += (a * a / tf + f32::EPSILON).ln();
                }
                f64::from((0.5 * (s / nb)).exp())
            })
            .collect()
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
        let mut inp = fft.make_input_vec();
        let mut spec = fft.make_output_vec();
        let lf = l as f32;
        let nb = (hi - lo) as f64;
        (0..self.p)
            .map(|px| {
                let mut acc = 0f64;
                for &t0 in &starts {
                    for (ti, v) in inp.iter_mut().enumerate() {
                        *v = f32::from(self.raw[(t0 + ti) * self.p + px]);
                    }
                    fft.process(&mut inp, &mut spec).expect("fft length");
                    let mut s = 0f64;
                    for c in &spec[lo..hi] {
                        let a = c.re.hypot(c.im);
                        s += f64::from((a * a / lf + f32::EPSILON).ln());
                    }
                    acc += s / nb;
                }
                (0.5 * acc / n_seg as f64).exp()
            })
            .collect()
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

// ---------------------------------------------------------------------------
// Randomized SVD and rank cut
// ---------------------------------------------------------------------------

/// Orthonormal basis (economic Householder QR, LAPACK's sign convention) of the
/// columns of `a` `(m, n)` row-major, as `(m, n)` row-major float32.
pub fn qr_q(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut c: Vec<f64> = vec![0.0; m * n]; // column-major
    for i in 0..m {
        for j in 0..n {
            c[j * m + i] = f64::from(a[i * n + j]);
        }
    }
    let mut tau = vec![0f64; n];
    for j in 0..n.min(m) {
        let (head, rest) = c.split_at_mut((j + 1) * m);
        let col = &mut head[j * m..];
        let alpha = col[j];
        let xnorm = col[j + 1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if xnorm == 0.0 {
            continue; // H = I
        }
        let beta = -alpha.signum() * alpha.hypot(xnorm);
        let scale = 1.0 / (alpha - beta);
        for v in &mut col[j + 1..] {
            *v *= scale;
        }
        tau[j] = (beta - alpha) / beta;
        col[j] = beta;
        let v = &col[j..];
        for k in 0..n - j - 1 {
            let ck = &mut rest[k * m + j..(k + 1) * m];
            let d = ck[0] + v[1..].iter().zip(&ck[1..]).map(|(a, b)| a * b).sum::<f64>();
            let f = tau[j] * d;
            ck[0] -= f;
            for (x, &vi) in ck[1..].iter_mut().zip(&v[1..]) {
                *x -= f * vi;
            }
        }
    }
    // Q = H_0 ... H_{n-1} I[:, :n], applied from the last reflector back
    let mut q = vec![0f64; m * n];
    for j in 0..n.min(m) {
        q[j * m + j] = 1.0;
    }
    for j in (0..n.min(m)).rev() {
        if tau[j] == 0.0 {
            continue;
        }
        let v = &c[j * m + j..(j + 1) * m];
        for k in j..n {
            let qk = &mut q[k * m + j..(k + 1) * m];
            let d = qk[0] + v[1..].iter().zip(&qk[1..]).map(|(a, b)| a * b).sum::<f64>();
            let f = tau[j] * d;
            qk[0] -= f;
            for (x, &vi) in qk[1..].iter_mut().zip(&v[1..]) {
                *x -= f * vi;
            }
        }
    }
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = q[j * m + i] as f32;
        }
    }
    out
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
                                  // SVD of B = bt^T from the eigenvectors of B B^T, in float64
    let mut g = vec![0f64; ell * ell];
    for row in bt.chunks_exact(ell) {
        for a in 0..ell {
            let ra = f64::from(row[a]);
            for b in a..ell {
                g[a * ell + b] += ra * f64::from(row[b]);
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
    let mut vt = vec![0f32; k * t];
    for (i, &j) in order.iter().enumerate() {
        if s[i] == 0.0 {
            continue;
        }
        for (ti, row) in bt.chunks_exact(ell).enumerate() {
            let d: f64 = (0..ell).map(|a| f64::from(row[a]) * evecs[a * ell + j]).sum();
            vt[i * t + ti] = (d / s[i]) as f32;
        }
    }
    let mut u = vec![0f32; p * k];
    for px in 0..p {
        let qr = &q[px * ell..(px + 1) * ell];
        for (i, &j) in order.iter().enumerate() {
            let d: f64 = (0..ell).map(|a| f64::from(qr[a]) * evecs[a * ell + j]).sum();
            u[px * k + i] = d as f32;
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
}
