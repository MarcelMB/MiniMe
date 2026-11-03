//! `minime.gmm`: a 1-D Gaussian mixture fitted as sklearn's `GaussianMixture(random_state=0)`
//! does it (k-means++ start from numpy's `RandomState(0)`, Lloyd's k-means, EM stopped once
//! the mean log-likelihood changes by less than `TOL`).
//!
//! The draws come from MT19937 seeded as numpy's legacy `RandomState`, so nothing needs
//! replaying. Sums follow numpy's orders: `np.sum` / `mean` / `var` of a 1-D array are
//! pairwise (`pairwise_sum`), `cumsum` and sums over axis 0 of an `(n, k)` array are
//! sequential. Only `resp.T @ x` (a BLAS gemv in numpy) runs in another order.

pub const TOL: f64 = 1e-3;
pub const REG_COVAR: f64 = 1e-6;
pub const MAX_ITER: usize = 100;
pub const KMEANS_TOL: f64 = 1e-4;
pub const KMEANS_MAX_ITER: usize = 300;
pub const SEED: u32 = 0;

// ---------------------------------------------------------------------------
// numpy's MT19937 and sums
// ---------------------------------------------------------------------------

/// numpy's legacy `RandomState(seed)`: MT19937 with `init_genrand`, `random_sample` from
/// two 32-bit outputs (53 bits).
pub struct Mt19937 {
    mt: [u32; 624],
    pos: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, pos: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[i] = v;
            }
            self.pos = 0;
        }
        let mut y = self.mt[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `RandomState.random_sample()`: uniform in [0, 1).
    pub fn random_sample(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (f64::from(a) * 67_108_864.0 + f64::from(b)) / 9_007_199_254_740_992.0
    }
}

/// `np.sum` of a contiguous float64 array: numpy's pairwise summation (blocks of 128 with
/// eight accumulators).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(-0.0, |s, &v| s + v)
    } else if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let m = n - n % 8;
        for c in a[8..m].as_chunks::<8>().0 {
            for j in 0..8 {
                r[j] += c[j];
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[m..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

fn sq(v: f64) -> f64 {
    v * v
}

fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `np.var` (ddof 0): pairwise sums of the values and of the squared deviations.
fn var(a: &[f64]) -> f64 {
    let m = mean(a);
    let d: Vec<f64> = a.iter().map(|&v| sq(v - m)).collect();
    pairwise_sum(&d) / a.len() as f64
}

/// First index of the smallest value (`argmin`).
fn argmin(a: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in a.iter().enumerate() {
        if v < a[best] {
            best = i;
        }
    }
    best
}

/// First index of the largest value (`argmax`).
pub fn argmax(a: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in a.iter().enumerate() {
        if v > a[best] {
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// k-means start
// ---------------------------------------------------------------------------

/// sklearn's greedy k-means++ centres (`2 + log k` candidates per centre).
fn kmeans_plusplus(x: &[f64], k: usize, rng: &mut Mt19937) -> Vec<f64> {
    let n = x.len();
    // rng.choice(n, p=uniform): one draw against the normalised cumulative sum
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for _ in 0..n {
        acc += 1.0 / n as f64;
        cdf.push(acc);
    }
    let last = cdf[n - 1];
    for c in &mut cdf {
        *c /= last;
    }
    let u = rng.random_sample();
    let first = cdf.partition_point(|&c| c <= u).min(n - 1);
    let mut centres = vec![x[first]];
    let mut closest: Vec<f64> = x.iter().map(|&v| sq(v - centres[0])).collect();
    let mut pot = pairwise_sum(&closest);
    let n_trials = 2 + (k as f64).ln() as usize;
    for _ in 1..k {
        let mut cum = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &c in &closest {
            acc += c;
            cum.push(acc);
        }
        let ids: Vec<usize> = (0..n_trials)
            .map(|_| {
                let v = rng.random_sample() * pot;
                cum.partition_point(|&c| c < v).min(n - 1)
            })
            .collect();
        let dist: Vec<Vec<f64>> = ids
            .iter()
            .map(|&i| {
                closest
                    .iter()
                    .zip(x)
                    .map(|(&c, &v)| c.min(sq(v - x[i])))
                    .collect()
            })
            .collect();
        let sums: Vec<f64> = dist.iter().map(|d| pairwise_sum(d)).collect();
        let best = argmin(&sums);
        pot = sums[best];
        closest.clone_from(&dist[best]);
        centres.push(x[ids[best]]);
    }
    centres
}

fn nearest(x: &[f64], centres: &[f64]) -> Vec<usize> {
    x.iter()
        .map(|&v| {
            let d: Vec<f64> = centres.iter().map(|&c| (v - c).abs()).collect();
            argmin(&d)
        })
        .collect()
}

/// Labels of sklearn's `KMeans(k, n_init=1, random_state=0)` on 1-D `x`.
fn kmeans_1d(x: &[f64], k: usize) -> Vec<usize> {
    let m = mean(x);
    let x: Vec<f64> = x.iter().map(|&v| v - m).collect();
    let mut centres = kmeans_plusplus(&x, k, &mut Mt19937::new(SEED));
    let tol = var(&x) * KMEANS_TOL;
    let mut old: Option<Vec<usize>> = None;
    for _ in 0..KMEANS_MAX_ITER {
        let labels = nearest(&x, &centres);
        let new: Vec<f64> = (0..k)
            .map(|j| {
                let sel: Vec<f64> = x
                    .iter()
                    .zip(&labels)
                    .filter(|(_, &l)| l == j)
                    .map(|(&v, _)| v)
                    .collect();
                if sel.is_empty() {
                    centres[j]
                } else {
                    mean(&sel)
                }
            })
            .collect();
        let d: Vec<f64> = new.iter().zip(&centres).map(|(a, b)| sq(a - b)).collect();
        let shift = pairwise_sum(&d);
        centres = new;
        if old.as_ref() == Some(&labels) {
            return labels;
        }
        if shift <= tol {
            break;
        }
        old = Some(labels);
    }
    nearest(&x, &centres)
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

/// Fitted mixture: component `weights`, `means` and `variances`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gmm1D {
    pub weights: Vec<f64>,
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub iterations: usize,
}

impl Gmm1D {
    /// `(n, k)` row-major `log w_k + log N(x | mu_k, var_k)`.
    fn weighted_log_prob(&self, x: &[f64]) -> Vec<f64> {
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let lv: Vec<f64> = self.variances.iter().map(|v| v.ln()).collect();
        let lw: Vec<f64> = self.weights.iter().map(|w| w.ln()).collect();
        let k = self.means.len();
        let mut out = Vec::with_capacity(x.len() * k);
        for &v in x {
            for j in 0..k {
                let d2 = sq(v - self.means[j]) / self.variances[j];
                out.push(-0.5 * (ln2pi + lv[j] + d2) + lw[j]);
            }
        }
        out
    }

    /// Most likely component of each value.
    pub fn predict(&self, x: &[f64]) -> Vec<usize> {
        let k = self.means.len();
        self.weighted_log_prob(x).chunks_exact(k).map(argmax).collect()
    }
}

/// Weights, means and variances from `(n, k)` row-major responsibilities.
fn m_step(x: &[f64], resp: &[f64], k: usize) -> Gmm1D {
    let eps10 = 10.0 * f64::EPSILON;
    let mut nk = vec![-0.0; k];
    let mut sx = vec![0.0; k];
    for (row, &v) in resp.chunks_exact(k).zip(x) {
        for j in 0..k {
            nk[j] += row[j];
            sx[j] += row[j] * v;
        }
    }
    for n in &mut nk {
        *n += eps10;
    }
    let means: Vec<f64> = sx.iter().zip(&nk).map(|(s, n)| s / n).collect();
    let mut sv = vec![-0.0; k];
    for (row, &v) in resp.chunks_exact(k).zip(x) {
        for j in 0..k {
            sv[j] += row[j] * sq(v - means[j]);
        }
    }
    let variances = sv.iter().zip(&nk).map(|(s, n)| s / n + REG_COVAR).collect();
    let tot = pairwise_sum(&nk);
    Gmm1D {
        weights: nk.iter().map(|n| n / tot).collect(),
        means,
        variances,
        iterations: 0,
    }
}

/// scipy's `logsumexp` of one row: the largest term split off, `log1p` of the rest.
fn logsumexp(row: &[f64]) -> f64 {
    let i = argmax(row);
    let m = row[i];
    let s: Vec<f64> = row
        .iter()
        .enumerate()
        .map(|(j, &v)| if j == i { 0.0 } else { (v - m).exp() })
        .collect();
    let s = pairwise_sum(&s);
    s.ln_1p() + m
}

/// Fit a `k`-component Gaussian mixture to `x`, as sklearn's defaults.
pub fn fit_gmm_1d(x: &[f64], k: usize) -> Result<Gmm1D, String> {
    if x.len() < k {
        return Err(format!("{} values for a {k}-component mixture", x.len()));
    }
    let labels = kmeans_1d(x, k);
    let mut resp = vec![0.0; x.len() * k];
    for (i, &l) in labels.iter().enumerate() {
        resp[i * k + l] = 1.0;
    }
    let mut g = m_step(x, &resp, k);
    let mut lower = f64::NEG_INFINITY;
    for it in 0..MAX_ITER {
        let wlp = g.weighted_log_prob(x);
        let norm: Vec<f64> = wlp.chunks_exact(k).map(logsumexp).collect();
        for (row, (w, &nm)) in resp.chunks_exact_mut(k).zip(wlp.chunks_exact(k).zip(&norm)) {
            for j in 0..k {
                row[j] = (w[j] - nm).exp();
            }
        }
        g = m_step(x, &resp, k);
        g.iterations = it + 1;
        let prev = lower;
        lower = mean(&norm);
        if (lower - prev).abs() < TOL {
            break;
        }
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_numpy() {
        // np.random.RandomState(0).random_sample(3)
        let mut r = Mt19937::new(0);
        let want = [0.5488135039273248, 0.7151893663724195, 0.6027633760716439];
        for w in want {
            assert_eq!(r.random_sample(), w);
        }
    }

    #[test]
    fn pairwise_sum_small_and_large() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin() * 1e3).collect();
        let naive: f64 = a.iter().sum();
        assert!((pairwise_sum(&a) - naive).abs() < 1e-9);
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn separates_two_clusters() {
        let mut r = Mt19937::new(5);
        let x: Vec<f64> = (0..400)
            .map(|i| r.random_sample() + if i % 4 == 0 { 10.0 } else { 0.0 })
            .collect();
        let g = fit_gmm_1d(&x, 2).unwrap();
        let hi = argmax(&g.means);
        let lab = g.predict(&x);
        for (i, &l) in lab.iter().enumerate() {
            assert_eq!(l == hi, i % 4 == 0);
        }
        assert!((g.weights[hi] - 0.25).abs() < 1e-6);
    }
}
