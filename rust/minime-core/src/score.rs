//! Port of `minime.autotune.score`: each unit is good or flagged with its reasons, the score is
//! `good - flagged - lost`, and `pair_stats` shows what `dup_corr` and `min_dist_frac` act on.

use serde::{Deserialize, Serialize};

use crate::init::{centroids, Csr};
use crate::seeds::ptp_q;

pub const SIZE_FRAC: f32 = 0.1; // footprint size = pixels above 10 % of peak
pub const REASONS: [&str; 4] = ["low_pnr", "in_pieces", "bad_size", "duplicate"];
pub const MIN_UNITS_FOR_ESTIMATE: usize = 10;

/// Thresholds of the four unit checks; sizes scale with the cell `diameter` (px).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Checks {
    pub diameter: f64,
    #[serde(default = "d_pnr_min")]
    pub pnr_min: f64, // trace_pnr scale: pure white noise scores ~6.2
    #[serde(default = "d_piece_frac")]
    pub piece_frac: f64,
    #[serde(default = "d_size_lo")]
    pub size_lo: f64, // x cell area
    #[serde(default = "d_size_hi")]
    pub size_hi: f64,
    #[serde(default = "d_dup_corr")]
    pub dup_corr: f64,
    #[serde(default = "d_keep_dist")]
    pub keep_dist: f64,
}
fn d_pnr_min() -> f64 {
    12.0
}
fn d_piece_frac() -> f64 {
    0.9
}
fn d_size_lo() -> f64 {
    0.3
}
fn d_size_hi() -> f64 {
    3.0
}
fn d_dup_corr() -> f64 {
    0.6
}
fn d_keep_dist() -> f64 {
    0.5
}

impl Checks {
    pub fn new(diameter: f64) -> Result<Self, String> {
        if !(diameter.is_finite() && diameter > 0.0) {
            return Err(format!("diameter must be a number of px > 0, got {diameter}"));
        }
        Ok(Self {
            diameter,
            pnr_min: d_pnr_min(),
            piece_frac: d_piece_frac(),
            size_lo: d_size_lo(),
            size_hi: d_size_hi(),
            dup_corr: d_dup_corr(),
            keep_dist: d_keep_dist(),
        })
    }
    pub fn cell_area(&self) -> f64 {
        std::f64::consts::PI * (self.diameter / 2.0).powi(2)
    }
}

// ---------------------------------------------------------------------------
// Footprint and trace metrics
// ---------------------------------------------------------------------------

/// `(K,)` pixels of each footprint above `frac` of its own maximum.
pub fn size_above(a: &Csr, frac: f32) -> Vec<usize> {
    (0..a.k)
        .map(|i| {
            let (_, v) = a.row(i);
            let peak = v.iter().copied().fold(0f32, f32::max); // scipy's max of an empty row is 0
            v.iter().filter(|&&x| x > frac * peak).count()
        })
        .collect()
}

/// `numpy.percentile(sorted, q)` with `method="linear"`.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    (v[(n - 1) / 2] + v[n / 2]) / 2.0
}

/// `(K,)` peak-to-noise of traces `(K, T)`: (99.9th - 0.1th percentile) / noise std from diffs.
/// Diffs between two frames that are both 0 are left out of the noise (`YrA` is clipped at 0).
/// A flat trace scores 0; a trace with range but no diff noise scores inf.
pub fn trace_pnr(c: &[f32], k: usize, t: usize) -> Vec<f64> {
    (0..k)
        .map(|i| {
            let row = &c[i * t..(i + 1) * t];
            let mut sorted = row.to_vec();
            let range = ptp_q(&mut sorted, (0.1, 99.9));
            let mut d: Vec<f64> = row
                .windows(2)
                .filter(|p| !(p[0] == 0.0 && p[1] == 0.0))
                .map(|p| (f64::from(p[1]) - f64::from(p[0])).abs())
                .collect();
            let med = if d.is_empty() { 0.0 } else { median(&mut d) }; // all-nan row: no noise, as flat
            let noise = med / 0.6745 / 2f64.sqrt();
            if noise > 0.0 {
                range / noise
            } else if range > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        })
        .collect()
}

/// `(K,)` share of the weight above `frac` of peak in the largest 8-connected piece (0 for a
/// unit with no pixels), as `cv2.connectedComponents` labels it.
pub fn main_piece_frac(a: &Csr, _h: usize, w: usize, frac: f32) -> Vec<f64> {
    (0..a.k)
        .map(|i| {
            let (idx, val) = a.row(i);
            let peak = val.iter().copied().fold(0f32, f32::max);
            if peak <= 0.0 {
                return 0.0;
            }
            let px: Vec<(usize, usize, f64)> = idx
                .iter()
                .zip(val)
                .filter(|(_, &v)| v > frac * peak)
                .map(|(&p, &v)| (p / w, p % w, f64::from(v)))
                .collect();
            if px.is_empty() {
                return 0.0;
            }
            let (y0, y1) = (
                px.iter().map(|p| p.0).min().unwrap(),
                px.iter().map(|p| p.0).max().unwrap(),
            );
            let (x0, x1) = (
                px.iter().map(|p| p.1).min().unwrap(),
                px.iter().map(|p| p.1).max().unwrap(),
            );
            let (bh, bw) = (y1 - y0 + 1, x1 - x0 + 1);
            let mut img = vec![0f64; bh * bw];
            for &(y, x, v) in &px {
                img[(y - y0) * bw + (x - x0)] = v;
            }
            let mut seen = vec![false; bh * bw];
            let (mut best, mut total) = (0.0, 0.0);
            for start in 0..bh * bw {
                if img[start] == 0.0 || seen[start] {
                    continue;
                }
                let mut stack = vec![start];
                seen[start] = true;
                let mut sum = 0.0;
                while let Some(p) = stack.pop() {
                    sum += img[p];
                    let (y, x) = (p / bw, p % bw);
                    for yy in y.saturating_sub(1)..(y + 2).min(bh) {
                        for xx in x.saturating_sub(1)..(x + 2).min(bw) {
                            let q = yy * bw + xx;
                            if img[q] != 0.0 && !seen[q] {
                                seen[q] = true;
                                stack.push(q);
                            }
                        }
                    }
                }
                total += sum;
                best = f64::max(best, sum);
            }
            best / total
        })
        .collect()
}

/// Rows of `traces` centred and unit-normed (flat rows -> zeros), float64 from the float32 values.
fn centred_rows(traces: &[f32], k: usize, t: usize) -> Vec<f64> {
    let mut z = vec![0.0; k * t];
    for i in 0..k {
        let row = &traces[i * t..(i + 1) * t];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / t.max(1) as f64;
        let out = &mut z[i * t..(i + 1) * t];
        for (o, &v) in out.iter_mut().zip(row) {
            *o = f64::from(v) - mean;
        }
        let nrm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for o in out.iter_mut() {
                *o /= nrm;
            }
        } else {
            out.fill(0.0);
        }
    }
    z
}

fn dot(z: &[f64], t: usize, i: usize, j: usize) -> f64 {
    z[i * t..(i + 1) * t]
        .iter()
        .zip(&z[j * t..(j + 1) * t])
        .map(|(a, b)| a * b)
        .sum()
}

fn finite(cent: &[[f64; 2]]) -> Vec<usize> {
    (0..cent.len())
        .filter(|&i| cent[i][0].is_finite() && cent[i][1].is_finite())
        .collect()
}

/// `(K,)` True where a unit closer than `radius` with trace corr >= `thres` has a higher PNR.
/// Equal PNR: the higher index is flagged. Units with a NaN centroid are never paired.
pub fn duplicates(
    cent: &[[f64; 2]],
    traces: &[f32],
    k: usize,
    t: usize,
    pnr: &[f64],
    radius: f64,
    thres: f64,
) -> Vec<bool> {
    let mut dup = vec![false; k];
    let ok = finite(cent);
    if ok.len() < 2 {
        return dup;
    }
    let z = centred_rows(traces, k, t);
    for (a, &i) in ok.iter().enumerate() {
        for &j in &ok[a + 1..] {
            let d = ((cent[i][0] - cent[j][0]).powi(2) + (cent[i][1] - cent[j][1]).powi(2)).sqrt();
            if d > radius || dot(&z, t, i, j) < thres {
                continue;
            }
            let (lo, hi) = (i.min(j), i.max(j));
            dup[if pnr[lo] >= pnr[hi] { hi } else { lo }] = true;
        }
    }
    dup
}

// ---------------------------------------------------------------------------
// Checks and score
// ---------------------------------------------------------------------------

pub struct UnitFlags {
    pub pnr: Vec<f64>,
    pub size10: Vec<usize>,
    pub main_piece: Vec<f64>,
    pub low_pnr: Vec<bool>,
    pub in_pieces: Vec<bool>,
    pub bad_size: Vec<bool>,
    pub duplicate: Vec<bool>,
    pub good: Vec<bool>,
    pub cent: Vec<[f64; 2]>, // (y, x), NaN for an empty footprint
}

/// One row per unit: metrics, one flag per reason in `REASONS`, and `good`.
pub fn unit_flags(a: &Csr, traces: &[f32], h: usize, w: usize, checks: &Checks) -> Result<UnitFlags, String> {
    let k = a.k;
    let t = traces.len().checked_div(k).unwrap_or(0);
    if traces.len() != k * t {
        return Err(format!("{} trace values for {k} footprints", traces.len()));
    }
    let pnr = trace_pnr(traces, k, t);
    let size10 = size_above(a, SIZE_FRAC);
    let cent = centroids(a, w);
    let area = checks.cell_area();
    let main_piece = main_piece_frac(a, h, w, SIZE_FRAC);
    let low_pnr: Vec<bool> = pnr.iter().map(|&p| p < checks.pnr_min).collect();
    let in_pieces: Vec<bool> = main_piece.iter().map(|&m| m < checks.piece_frac).collect();
    let bad_size: Vec<bool> = size10
        .iter()
        .map(|&s| (s as f64) < checks.size_lo * area || (s as f64) > checks.size_hi * area)
        .collect();
    let duplicate = duplicates(&cent, traces, k, t, &pnr, checks.diameter, checks.dup_corr);
    let good = (0..k)
        .map(|i| !(low_pnr[i] || in_pieces[i] || bad_size[i] || duplicate[i]))
        .collect();
    Ok(UnitFlags {
        pnr,
        size10,
        main_piece,
        low_pnr,
        in_pieces,
        bad_size,
        duplicate,
        good,
        cent,
    })
}

/// Centroids of the good units with a finite centroid.
pub fn good_centroids(f: &UnitFlags) -> Vec<[f64; 2]> {
    finite(&f.cent)
        .into_iter()
        .filter(|&i| f.good[i])
        .map(|i| f.cent[i])
        .collect()
}

/// How many points of `reference` (the run's good units) have no unit of `f` (good or flagged)
/// within `radius` px.
pub fn lost_units(reference: &[[f64; 2]], f: &UnitFlags, radius: f64) -> usize {
    let c: Vec<[f64; 2]> = finite(&f.cent).into_iter().map(|i| f.cent[i]).collect();
    if reference.is_empty() {
        return 0;
    }
    if c.is_empty() {
        return reference.len();
    }
    reference
        .iter()
        .filter(|r| {
            c.iter()
                .map(|p| ((p[0] - r[0]).powi(2) + (p[1] - r[1]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
                > radius
        })
        .count()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub good: i64,
    pub flagged: i64,
    pub lost: i64,
    pub score: i64,
    pub low_pnr: i64,
    pub in_pieces: i64,
    pub bad_size: i64,
    pub duplicate: i64,
}

pub fn summarize(f: &UnitFlags, lost: usize) -> Summary {
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count() as i64;
    let good = count(&f.good);
    let flagged = f.good.len() as i64 - good;
    Summary {
        good,
        flagged,
        lost: lost as i64,
        score: good - flagged - lost as i64,
        low_pnr: count(&f.low_pnr),
        in_pieces: count(&f.in_pieces),
        bad_size: count(&f.bad_size),
        duplicate: count(&f.duplicate),
    }
}

pub struct PairStats {
    pub near_corr: Vec<f64>,
    pub far_corr: Vec<f64>,
    pub nn_dist: Vec<f64>, // in units of D
    pub n_near: usize,
    pub n_near_above: usize,
    pub n_closer: usize,
    pub far_p99: Option<f64>,
}

/// What the two thresholds act on: trace correlations of unit pairs closer than D (`near_corr`),
/// of pairs farther than 2 D as the null (`far_corr`), each unit's nearest-neighbour distance
/// in units of D, and the counts on the far side of `checks.dup_corr` and `min_dist_frac`.
#[allow(clippy::too_many_arguments)]
pub fn pair_stats(
    a: &Csr,
    traces: &[f32],
    k: usize,
    t: usize,
    _h: usize,
    w: usize,
    checks: &Checks,
    min_dist_frac: f64,
) -> PairStats {
    let d = checks.diameter;
    let cent = centroids(a, w);
    let ok = finite(&cent);
    let empty = PairStats {
        near_corr: vec![],
        far_corr: vec![],
        nn_dist: vec![],
        n_near: 0,
        n_near_above: 0,
        n_closer: 0,
        far_p99: None,
    };
    if ok.len() < 2 {
        return empty;
    }
    let z = centred_rows(traces, k, t);
    let mut nn = vec![f64::INFINITY; ok.len()];
    let (mut near, mut far) = (Vec::new(), Vec::new());
    for (a_, &i) in ok.iter().enumerate() {
        for (b_, &j) in ok.iter().enumerate().skip(a_ + 1) {
            let dist = ((cent[i][0] - cent[j][0]).powi(2) + (cent[i][1] - cent[j][1]).powi(2)).sqrt();
            let corr = dot(&z, t, i, j);
            nn[a_] = nn[a_].min(dist);
            nn[b_] = nn[b_].min(dist);
            if dist < d {
                near.push(corr);
            } else if dist > 2.0 * d {
                far.push(corr);
            }
        }
    }
    let mut sorted = far.clone();
    sorted.sort_by(f64::total_cmp);
    let nn_dist: Vec<f64> = nn.iter().map(|&v| v / d).collect();
    PairStats {
        n_near: near.len(),
        n_near_above: near.iter().filter(|&&c| c >= checks.dup_corr).count(),
        n_closer: nn.iter().filter(|&&v| v < min_dist_frac * d).count(),
        far_p99: if sorted.is_empty() {
            None
        } else {
            Some(percentile(&sorted, 99.0))
        },
        near_corr: near,
        far_corr: far,
        nn_dist,
    }
}

/// Cell diameter (px) from the run's final footprints: the equivalent diameter of the median
/// footprint size above 10 % of peak, one decimal. `Err` with fewer than 10 units.
pub fn estimate_diameter(a: &Csr) -> Result<f64, String> {
    if a.k < MIN_UNITS_FOR_ESTIMATE {
        return Err(format!(
            "only {} units: too few to estimate the cell diameter, pass --diameter",
            a.k
        ));
    }
    let mut sizes: Vec<f64> = size_above(a, SIZE_FRAC).into_iter().map(|s| s as f64).collect();
    let med = median(&mut sizes);
    Ok((2.0 * (med / std::f64::consts::PI).sqrt() * 10.0).round() / 10.0)
}

// ---------------------------------------------------------------------------
// Documents: unit_flags.csv, the flags and pairs replies of the Tune page
// ---------------------------------------------------------------------------

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// A float as pandas' `to_csv` prints a float64 (`repr`): the shortest round-trip text, `.0` kept.
fn py_float(v: f64) -> String {
    if v.is_nan() {
        return String::new();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// `unit_flags.csv` as `autotune.run` writes it (pandas `to_csv(index=False)`).
pub fn flags_csv(f: &UnitFlags) -> String {
    let mut out =
        String::from("unit_id,pnr,size10,main_piece,low_pnr,in_pieces,bad_size,duplicate,good,y,x\n");
    for i in 0..f.good.len() {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            i,
            py_float(f.pnr[i]),
            f.size10[i],
            py_float(f.main_piece[i]),
            py_bool(f.low_pnr[i]),
            py_bool(f.in_pieces[i]),
            py_bool(f.bad_size[i]),
            py_bool(f.duplicate[i]),
            py_bool(f.good[i]),
            py_float(f.cent[i][0]),
            py_float(f.cent[i][1]),
        ));
    }
    out
}

/// `serve.tune.compute_flags`'s document: the summary, the unit ids per reason, the checks.
pub fn flags_doc(f: &UnitFlags, checks: &Checks) -> serde_json::Value {
    let ids = |v: &[bool]| -> Vec<usize> { (0..v.len()).filter(|&i| v[i]).collect() };
    let mut doc = serde_json::to_value(summarize(f, 0)).unwrap();
    doc["reasons"] = serde_json::json!({
        "low_pnr": ids(&f.low_pnr),
        "in_pieces": ids(&f.in_pieces),
        "bad_size": ids(&f.bad_size),
        "duplicate": ids(&f.duplicate),
    });
    doc["checks"] = serde_json::to_value(checks).unwrap();
    doc
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

/// `np.histogram(values, bins)`: the last bin includes its right edge.
fn histogram(values: &[f64], bins: &[f64]) -> Vec<i64> {
    let n = bins.len() - 1;
    let mut out = vec![0i64; n];
    for &v in values {
        if v < bins[0] || v > bins[n] {
            continue;
        }
        let mut b = bins[1..].iter().position(|&e| v < e).unwrap_or(n - 1);
        if b >= n {
            b = n - 1;
        }
        out[b] += 1;
    }
    out
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// `serve.tune.compute_pairs`'s document: histograms of what the two thresholds act on.
pub fn pairs_doc(s: &PairStats, checks: &Checks, min_dist_frac: f64) -> serde_json::Value {
    let corr_bins = linspace(-0.4, 1.0, 29);
    let nn_bins = linspace(0.0, 2.0, 41);
    let nn: Vec<f64> = s.nn_dist.iter().map(|&v| v.min(2.0)).collect();
    serde_json::json!({
        "corr_bins": corr_bins.iter().map(|&v| round3(v)).collect::<Vec<_>>(),
        "near_corr": histogram(&s.near_corr, &corr_bins),
        "far_corr": histogram(&s.far_corr, &corr_bins),
        "nn_bins": nn_bins.iter().map(|&v| round3(v)).collect::<Vec<_>>(),
        "nn_dist": histogram(&nn, &nn_bins),
        "dup_corr": checks.dup_corr,
        "min_dist_frac": min_dist_frac,
        "n_near": s.n_near,
        "n_near_above": s.n_near_above,
        "n_closer": s.n_closer,
        "far_p99": s.far_p99,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::Csr;

    const H: usize = 50;
    const W: usize = 50;

    /// One row per unit: the union of square blobs `(row, col, half-side)`, value 1 (the Python test's `_footprints`).
    fn footprints(blobs: &[&[(usize, usize, usize)]]) -> Csr {
        let (mut indptr, mut indices, mut data) = (vec![0usize], Vec::new(), Vec::new());
        for unit in blobs {
            let mut img = vec![0f32; H * W];
            for &(r, c, s) in *unit {
                for y in r - s..=r + s {
                    for x in c - s..=c + s {
                        img[y * W + x] = 1.0;
                    }
                }
            }
            for (p, &v) in img.iter().enumerate() {
                if v > 0.0 {
                    indices.push(p);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        Csr {
            k: blobs.len(),
            p: H * W,
            indptr,
            indices,
            data,
        }
    }

    /// Deterministic LCG normals (Box–Muller) so the tests need no rand crate.
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// `_transient`: 10 spikes of 20 through an AR(1) with 0.9.
    fn transient(rng: &mut Rng, t: usize) -> Vec<f64> {
        let mut s = vec![0.0; t];
        for _ in 0..10 {
            let i = (rng.uniform() * t as f64) as usize;
            s[i.min(t - 1)] = 20.0;
        }
        let mut out = vec![0.0; t];
        for i in 0..t {
            out[i] = s[i] + if i > 0 { 0.9 * out[i - 1] } else { 0.0 };
        }
        out
    }

    fn good_traces(k: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(k * 2000);
        for _ in 0..k {
            let tr = transient(&mut rng, 2000);
            out.extend(tr.iter().map(|&v| (v + rng.normal()) as f32));
        }
        out
    }

    fn checks() -> Checks {
        Checks::new(6.0).unwrap() // cell area 28.3 px -> size 8.5-84.8 px
    }

    /// Flags with only `good` and `cent` set (the other fields are zero-filled).
    fn flags_with(good: Vec<bool>, cent: Vec<[f64; 2]>) -> UnitFlags {
        let n = good.len();
        UnitFlags {
            pnr: vec![0.0; n],
            size10: vec![0; n],
            main_piece: vec![0.0; n],
            low_pnr: vec![false; n],
            in_pieces: vec![false; n],
            bad_size: vec![false; n],
            duplicate: vec![false; n],
            good,
            cent,
        }
    }

    #[test]
    fn pnr_flags_noise_and_flat_traces() {
        let a = footprints(&[&[(10, 10, 2)], &[(10, 30, 2)], &[(30, 10, 2)]]);
        let mut rng = Rng(1);
        let mut traces = good_traces(1, 0);
        traces.extend((0..2000).map(|_| rng.normal() as f32));
        traces.extend(std::iter::repeat_n(0f32, 2000));
        let f = unit_flags(&a, &traces, H, W, &checks()).unwrap();
        assert_eq!(f.low_pnr, vec![false, true, true]);
        assert_eq!(trace_pnr(&vec![0f32; 100], 1, 100)[0], 0.0); // flat: 0, not inf
    }

    #[test]
    fn pnr_of_clipped_noise_is_finite_and_low() {
        // YrA is clipped at 0: a noise trace at 0 most of the time must not score inf
        let mut rng = Rng(2);
        let mut c = Vec::new();
        for mean in [-1.0, -0.5, 0.0] {
            c.extend((0..4000).map(|_| ((mean + rng.normal()) as f32).max(0.0)));
        }
        let pnr = trace_pnr(&c, 3, 4000);
        assert!(
            pnr.iter()
                .all(|v| v.is_finite() && *v < Checks::new(8.0).unwrap().pnr_min),
            "{pnr:?}"
        );
    }

    #[test]
    fn size_and_pieces() {
        let a = footprints(&[
            &[(10, 10, 2)],              // 25 px: fine
            &[(10, 30, 0)],              // 1 px: too small
            &[(30, 25, 6)],              // 169 px: too large
            &[(40, 5, 2), (40, 30, 2)],  // two equal pieces
            &[(20, 40, 2), (20, 46, 0)], // one piece + a 1-px speck (25 / 26 > 0.9)
        ]);
        let f = unit_flags(&a, &good_traces(5, 0), H, W, &checks()).unwrap();
        assert_eq!(f.bad_size, vec![false, true, true, false, false]);
        assert_eq!(f.in_pieces, vec![false, false, false, true, false]);
        assert!((f.main_piece[3] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn duplicate_flags_the_weaker_correlated_neighbour() {
        let a = footprints(&[&[(10, 10, 2)], &[(10, 10, 2)], &[(10, 14, 2)], &[(40, 40, 2)]]);
        let mut rng = Rng(2);
        let t0 = transient(&mut rng, 2000);
        let other = transient(&mut rng, 2000);
        let mut traces = Vec::new();
        traces.extend(t0.iter().map(|&v| (v + 0.5 * rng.normal()) as f32));
        traces.extend(t0.iter().map(|&v| (0.5 * v + 0.5 * rng.normal()) as f32)); // same cell, weaker
        traces.extend(other.iter().map(|&v| (v + 0.5 * rng.normal()) as f32)); // neighbour, other cell
        traces.extend(t0.iter().map(|&v| (v + 0.5 * rng.normal()) as f32)); // same trace but far away
        let f = unit_flags(&a, &traces, H, W, &checks()).unwrap();
        assert_eq!(f.duplicate, vec![false, true, false, false]);
    }

    #[test]
    fn empty_and_degenerate_units() {
        let empty = Csr {
            k: 0,
            p: H * W,
            indptr: vec![0],
            indices: vec![],
            data: vec![],
        };
        let f = unit_flags(&empty, &[], H, W, &checks()).unwrap();
        assert_eq!(summarize(&f, 0), Summary::default());
        let mut a = footprints(&[&[(10, 10, 2)]]);
        a.k = 2;
        a.indptr.push(a.indices.len()); // second row has no pixels
        let f = unit_flags(&a, &good_traces(2, 0), H, W, &checks()).unwrap();
        assert_eq!(f.good, vec![true, false]);
    }

    #[test]
    fn summarize_counts_reasons() {
        let mut f = flags_with(vec![true, false, false], vec![[0.0, 0.0]; 3]);
        f.low_pnr = vec![false, true, true];
        f.in_pieces = vec![false, false, true];
        let s = summarize(&f, 0);
        assert_eq!(
            (s.good, s.flagged, s.lost, s.score, s.low_pnr, s.in_pieces),
            (1, 2, 0, -1, 2, 1)
        );
        assert_eq!(summarize(&f, 3).score, -4); // good - flagged - lost
    }

    #[test]
    fn lost_units_counts_the_runs_good_units_without_a_unit_nearby() {
        let reference = [[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]];
        let cent = vec![[11.0, 10.0], [30.0, 34.0], [f64::NAN, 0.0]];
        let f = flags_with(vec![true, false, true], cent.clone());
        assert_eq!(lost_units(&reference, &f, 3.0), 2); // kept by a flagged unit too; NaN ignored
        assert_eq!(lost_units(&reference, &f, 5.0), 1);
        assert_eq!(lost_units(&[], &f, 3.0), 0);
        let none = flags_with(vec![], vec![]);
        assert_eq!(lost_units(&reference, &none, 3.0), 3);
        assert_eq!(
            good_centroids(&flags_with(vec![true, false, true], cent)),
            vec![[11.0, 10.0]]
        );
    }

    #[test]
    fn pair_stats_counts_and_percentile() {
        let a = footprints(&[&[(10, 10, 2)], &[(10, 14, 2)], &[(40, 40, 2)]]);
        let s = pair_stats(&a, &good_traces(3, 1), 3, 2000, H, W, &checks(), 0.4);
        assert_eq!(s.nn_dist.len(), 3);
        assert_eq!(s.n_near, s.near_corr.len());
        assert_eq!(s.n_near, 1); // units 0 and 1 are 4 px apart (< D = 6)
        assert_eq!(s.n_closer, s.nn_dist.iter().filter(|&&d| d < 0.4).count());
        let one = footprints(&[&[(10, 10, 2)]]);
        assert_eq!(
            pair_stats(&one, &good_traces(1, 1), 1, 2000, H, W, &checks(), 0.4).n_near,
            0
        );
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0], 99.0) - 3.97).abs() < 1e-12);
    }

    #[test]
    fn estimate_diameter_needs_ten_units() {
        let a = footprints(&[&[(10, 10, 2)]]);
        assert!(estimate_diameter(&a).unwrap_err().contains("too few"));
        let blobs: Vec<&[(usize, usize, usize)]> = vec![&[(10, 10, 2)]; 10];
        let d = estimate_diameter(&footprints(&blobs)).unwrap();
        assert!((d - 5.6).abs() < 1e-9); // 25 px -> 2 sqrt(25 / pi) = 5.64 -> 5.6
    }

    #[test]
    fn flags_csv_has_pandas_columns_and_bools() {
        let a = footprints(&[&[(10, 10, 2)]]);
        let csv = flags_csv(&unit_flags(&a, &good_traces(1, 0), H, W, &checks()).unwrap());
        let mut lines = csv.lines();
        assert_eq!(
            lines.next().unwrap(),
            "unit_id,pnr,size10,main_piece,low_pnr,in_pieces,bad_size,duplicate,good,y,x"
        );
        let row = lines.next().unwrap();
        assert!(
            row.starts_with("0,") && row.contains(",False,") && row.ends_with(",10.0,10.0"),
            "{row}"
        );
    }
}
