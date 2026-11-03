//! `minime.corr.FactorCorr`: correlations of low-passed pixel traces as bilinear forms on
//! the tile factors, and the seed merge (`brightest_parent`) built on them.
//!
//! With `G_ab = V_lp,a^T V_lp,b`: `num(i, j) = W_i G_ab W_j^T`, `||i||^2 = W_i G_aa W_i^T +
//! f_lp resid_i`, `corr = num / sqrt(||i||^2 ||j||^2)` (1 for a pixel with itself). The
//! low-pass is a brick wall on the FFT (DC and bins from `int(noise_freq * T)` removed).
//! Everything runs in float64 as in Python, so values differ from it by rounding only.

use crate::factors::Factors;
use realfft::RealFftPlanner;
use std::collections::HashMap;

/// `corr.lowpass` of the columns of `v` (`(t, r)` row-major float32): float64, same layout.
pub fn lowpass_cols(v: &[f32], t: usize, r: usize, noise_freq: f64) -> Vec<f64> {
    let cut = (noise_freq * t as f64) as usize;
    let mut planner = RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(t);
    let inv = planner.plan_fft_inverse(t);
    let mut x = fwd.make_input_vec();
    let mut spec = fwd.make_output_vec();
    let mut out = vec![0f64; t * r];
    for c in 0..r {
        for (ti, xv) in x.iter_mut().enumerate() {
            *xv = f64::from(v[ti * r + c]);
        }
        fwd.process(&mut x, &mut spec).expect("fft length");
        spec[0] = 0.0.into();
        for s in spec.iter_mut().skip(cut) {
            *s = 0.0.into();
        }
        if t.is_multiple_of(2) {
            // irfft ignores the imaginary part at the Nyquist bin
            spec[t / 2].im = 0.0;
        }
        inv.process(&mut spec, &mut x).expect("fft length");
        for (ti, xv) in x.iter().enumerate() {
            out[ti * r + c] = xv / t as f64;
        }
    }
    out
}

/// `a^T b` of `(t, ra)` and `(t, rb)` row-major float64 matrices: `(ra, rb)`.
fn gram(a: &[f64], ra: usize, b: &[f64], rb: usize) -> Vec<f64> {
    let mut g = vec![0f64; ra * rb];
    for (ar, br) in a.chunks_exact(ra.max(1)).zip(b.chunks_exact(rb.max(1))) {
        for i in 0..ra {
            for j in 0..rb {
                g[i * rb + j] += ar[i] * br[j];
            }
        }
    }
    g
}

/// `x G y^T` for row vectors `x` (ra), `y` (rb) and `G` (ra, rb).
fn bilinear(x: &[f32], g: &[f64], y: &[f32]) -> f64 {
    let rb = y.len();
    let mut s = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        let row = &g[i * rb..(i + 1) * rb];
        let u: f64 = row.iter().zip(y).map(|(&gv, &yv)| gv * f64::from(yv)).sum();
        s += f64::from(xi) * u;
    }
    s
}

pub struct FactorCorr {
    pub noise_freq: f64,
    vlp: Vec<Vec<f32>>,  // per tile (t, r): low-passed V, float32 as Python keeps it
    gaa: Vec<Vec<f64>>,  // per tile (r, r), from the float64 low-pass
    pub norm2: Vec<f64>, // (P,) per-pixel squared norm
    gab: HashMap<(usize, usize), Vec<f64>>,
}

impl FactorCorr {
    pub fn new(f: &Factors, noise_freq: f64) -> Self {
        let t = f.t;
        let nr = (noise_freq * t as f64) as usize;
        let f_lp = 2.0 * (nr as f64 - 1.0) / t as f64;
        let mut vlp = Vec::with_capacity(f.tiles.len());
        let mut gaa = Vec::with_capacity(f.tiles.len());
        let mut norm2 = vec![0f64; f.p()];
        for (g, tf) in f.grid.iter().zip(&f.tiles) {
            let r = tf.rank;
            let v = lowpass_cols(&tf.v, t, r, noise_freq);
            let ga = gram(&v, r, &v, r);
            let mut loc = 0;
            for y in g.y0..g.y1 {
                for x in g.x0..g.x1 {
                    let w = &tf.w[loc * r..(loc + 1) * r];
                    let low = bilinear(w, &ga, w);
                    let wsq: f64 = w.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
                    let resid = (tf.energy[loc] - wsq).max(0.0);
                    norm2[y * f.w + x] = low + f_lp * resid;
                    loc += 1;
                }
            }
            vlp.push(v.iter().map(|&a| a as f32).collect());
            gaa.push(ga);
        }
        Self {
            noise_freq,
            vlp,
            gaa,
            norm2,
            gab: HashMap::new(),
        }
    }

    fn g(&mut self, f: &Factors, a: usize, b: usize) -> &[f64] {
        if a == b {
            return &self.gaa[a];
        }
        let (ra, rb) = (f.tiles[a].rank, f.tiles[b].rank);
        let (va, vb) = (&self.vlp[a], &self.vlp[b]);
        self.gab.entry((a, b)).or_insert_with(|| {
            let da: Vec<f64> = va.iter().map(|&v| f64::from(v)).collect();
            let db: Vec<f64> = vb.iter().map(|&v| f64::from(v)).collect();
            gram(&da, ra, &db, rb)
        })
    }

    /// Correlation of the flat pixels `src` and `dst`, in [-1, 1].
    pub fn pair(&mut self, f: &Factors, src: usize, dst: usize) -> f64 {
        let ((a, la), (b, lb)) = (f.locate(src), f.locate(dst));
        let den = (self.norm2[src] * self.norm2[dst]).sqrt();
        if den <= 0.0 {
            return 0.0;
        }
        if src == dst {
            return 1.0;
        }
        let (ra, rb) = (f.tiles[a].rank, f.tiles[b].rank);
        let x = &f.tiles[a].w[la * ra..(la + 1) * ra];
        let y = &f.tiles[b].w[lb * rb..(lb + 1) * rb];
        let num = bilinear(x, self.g(f, a, b), y);
        (num / den).clamp(-1.0, 1.0)
    }
}

/// Seed pairs `(i, j)`, `i < j`, at most `dist` apart (`cKDTree.query_pairs`), in
/// lexicographic order. `hw` must be sorted by row (seeds come in row-major order).
pub fn close_pairs(hw: &[[i64; 2]], dist: f64) -> Vec<[usize; 2]> {
    let d2 = dist * dist;
    let mut out = Vec::new();
    for i in 0..hw.len() {
        for j in i + 1..hw.len() {
            let dh = (hw[j][0] - hw[i][0]) as f64;
            if dh > dist {
                break;
            }
            let dw = (hw[j][1] - hw[i][1]) as f64;
            if dh * dh + dw * dw <= d2 {
                out.push([i, j]);
            }
        }
    }
    out
}

/// `corr.seeds_merge`: keep each seed that has no brighter linked neighbour (`max_proj`,
/// ties to the lower index), linked = within `dist` and correlated above `thres`. Returns
/// the keep mask, the close pairs and their correlations.
pub fn seeds_merge(
    fc: &mut FactorCorr,
    f: &Factors,
    hw: &[[i64; 2]],
    dist: f64,
    thres: f64,
) -> (Vec<bool>, Vec<[usize; 2]>, Vec<f64>) {
    let n = hw.len();
    let px: Vec<usize> = hw.iter().map(|&[h, w]| h as usize * f.w + w as usize).collect();
    let bright: Vec<u8> = px.iter().map(|&p| f.max_proj[p]).collect();
    let brighter = |a: usize, b: usize| bright[a] > bright[b] || (bright[a] == bright[b] && a < b);
    let pairs = if n > 1 { close_pairs(hw, dist) } else { Vec::new() };
    let corr: Vec<f64> = pairs.iter().map(|&[i, j]| fc.pair(f, px[i], px[j])).collect();
    let mut keep = vec![true; n];
    for (&[i, j], &c) in pairs.iter().zip(&corr) {
        if c > thres {
            if brighter(i, j) {
                keep[j] = false;
            } else {
                keep[i] = false;
            }
        }
    }
    (keep, pairs, corr)
}
