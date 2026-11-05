//! The seed stage of a window run (`pipeline.seeds_unmerged` + `seeds_merged`): Minian's
//! rolling-window local maxima on pass 1's block maxima (`seeds.seeds_init`), the
//! peak-to-noise filter with its 2-component GMM threshold (`pnr_refine`), the opt-in
//! `gmm_refine`, and the merge of close, correlated seeds (`corr.seeds_merge`).
//!
//! The local maxima are exact (integer images). `local_max_union` uses that the disks are
//! nested: a pixel that is the maximum of the radius-`r` disk is also the maximum of every
//! smaller one, and its contrast `max - min` only grows with `r`, so the union over the
//! kernels `k0..k1` is "maximum of the `k0` disk, and the contrast at the largest radius
//! where it is still the maximum exceeds `diff`". That visits a few rings per pixel instead
//! of a dilation and an erosion per kernel.

use crate::corr::{self, FactorCorr};
use crate::factors::Factors;
use crate::gmm::{self, argmax, fit_gmm_1d, Gmm1D};
use crate::npy;
use serde::Deserialize;

/// `ref_seeds.py`'s `seeds.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct SeedParams {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub tile: usize,
    pub seeds_wnd: usize,
    pub seeds_stp: usize,
    pub seeds_max_wnd: usize,
    pub seeds_diff_thres: f64,
    pub noise_freq: f64,
    pub pnr_thres: serde_json::Value, // "auto" or a number
    pub gmm_refine: bool,
    pub merge_dist: f64,
    pub merge_corr: f64,
}

pub const PNR_Q: (f64, f64) = (0.1, 99.9);

// ---------------------------------------------------------------------------
// Windows and local maxima
// ---------------------------------------------------------------------------

/// Minian's rolling windows `[start, stop)`: `ceil(t / stp) + 1` centres from
/// `np.linspace(0, t - 1)`, half width `ceil(wnd / 2)`, clipped to `[0, t]`.
pub fn rolling_windows(t: usize, wnd: usize, stp: usize) -> Vec<(usize, usize)> {
    let nstp = t.div_ceil(stp) + 1;
    let hwnd = (wnd as f64 / 2.0).ceil();
    let step = (t as f64 - 1.0) / (nstp - 1) as f64;
    (0..nstp)
        .map(|i| {
            // np.linspace sets the last point to the stop exactly
            let c = if i == nstp - 1 {
                t as f64 - 1.0
            } else {
                i as f64 * step
            };
            (
                (c - hwnd).floor().max(0.0) as usize,
                ((c + hwnd).ceil() as usize).min(t),
            )
        })
        .collect()
}

/// `[i0, i1)` block range of each window; every window bound must be a block edge.
pub fn window_block_ranges(
    edges: &[usize],
    windows: &[(usize, usize)],
) -> Result<Vec<(usize, usize)>, String> {
    windows
        .iter()
        .map(
            |&(a, b)| match (edges.binary_search(&a), edges.binary_search(&b)) {
                (Ok(i0), Ok(i1)) => Ok((i0, i1)),
                _ => Err(format!(
                    "window {a}-{b}: edge missing from the stored block edges"
                )),
            },
        )
        .collect()
}

/// Offsets `(dy, dx)` by ring: ring `r` holds `(r - 1)^2 < dy^2 + dx^2 <= r^2` (ring 0 is
/// the centre), so the disk of radius `r` (`seeds.disk`) is rings `0..=r`.
pub(crate) fn rings(rmax: usize) -> Vec<Vec<(isize, isize)>> {
    let r = rmax as isize;
    let mut out = vec![Vec::new(); rmax + 1];
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dy * dy + dx * dx;
            if d2 <= r * r {
                let ring = (0..=r).find(|&k| d2 <= k * k).unwrap_or(0) as usize;
                out[ring].push((dy, dx));
            }
        }
    }
    out
}

/// Union over disk kernels of radius `k0..k1` of Minian's `local_extreme(fm, k, "max",
/// diff)`: `fm == dilate(fm, k)` and `dilate - erode > diff` (pixels outside the frame are
/// ignored, as OpenCV's default morphology border does).
pub fn local_max_union(fm: &[u8], h: usize, w: usize, k0: usize, k1: usize, diff: f64) -> Vec<bool> {
    let mut out = vec![false; h * w];
    if k1 <= k0 {
        return out;
    }
    let rg = rings(k1 - 1);
    let ring_range = |y: usize, x: usize, ring: &[(isize, isize)]| {
        let (mut mx, mut mn) = (0u8, 255u8);
        for &(dy, dx) in ring {
            let (yy, xx) = (y as isize + dy, x as isize + dx);
            if yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w {
                let v = fm[yy as usize * w + xx as usize];
                mx = mx.max(v);
                mn = mn.min(v);
            }
        }
        (mx, mn)
    };
    for y in 0..h {
        for x in 0..w {
            let v = fm[y * w + x];
            let mut mn = v;
            let mut is_max = true;
            for ring in &rg[1..=k0] {
                let (rmx, rmn) = ring_range(y, x, ring);
                if rmx > v {
                    is_max = false;
                    break;
                }
                mn = mn.min(rmn);
            }
            if !is_max {
                continue;
            }
            for ring in &rg[k0 + 1..k1] {
                let (rmx, rmn) = ring_range(y, x, ring);
                if rmx > v {
                    break;
                }
                mn = mn.min(rmn);
            }
            out[y * w + x] = f64::from(v - mn) > diff;
        }
    }
    out
}

/// `seeds._local_max_roll`: one pixel per 8-connected component of `mask`, at the per-axis
/// median of its rows and of its columns (`int(np.median)`: the floor of the mean of the two
/// middle values for an even count).
pub fn component_medians(mask: &[bool], h: usize, w: usize) -> Vec<(usize, usize)> {
    let mut seen = vec![false; h * w];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..h * w {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        while let Some(p) = stack.pop() {
            let (y, x) = (p / w, p % w);
            rows.push(y);
            cols.push(x);
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    let q = yy * w + xx;
                    if mask[q] && !seen[q] {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        rows.sort_unstable();
        cols.sort_unstable();
        let n = rows.len();
        let med = |s: &[usize]| (s[(n - 1) / 2] + s[n / 2]) / 2;
        out.push((med(&rows), med(&cols)));
    }
    out
}

/// `seeds.seeds_init`: `(height, width, count)` of every pixel that is a seed in at least
/// one rolling window, row-major.
pub fn seeds_init(f: &Factors, p: &SeedParams) -> Result<Vec<[i64; 3]>, String> {
    let windows: Vec<(usize, usize)> = rolling_windows(f.t, p.seeds_wnd, p.seeds_stp);
    if windows.is_empty() {
        return Err(format!(
            "no {}-frame seed window fits in {} frames",
            p.seeds_wnd, f.t
        ));
    }
    let ranges = window_block_ranges(&f.edges, &windows)?;
    let (h, w) = (f.h, f.w);
    let mut count = vec![0i64; h * w];
    let mut fm = vec![0u8; h * w];
    for (i0, i1) in ranges {
        fm.copy_from_slice(&f.block_max[i0 * h * w..(i0 + 1) * h * w]);
        for b in i0 + 1..i1 {
            for (m, &v) in fm.iter_mut().zip(&f.block_max[b * h * w..(b + 1) * h * w]) {
                *m = (*m).max(v);
            }
        }
        let union = local_max_union(&fm, h, w, 2, p.seeds_max_wnd, p.seeds_diff_thres);
        for (y, x) in component_medians(&union, h, w) {
            count[y * w + x] += 1;
        }
    }
    Ok((0..h * w)
        .filter(|&i| count[i] > 0)
        .map(|i| [(i / w) as i64, (i % w) as i64, count[i]])
        .collect())
}

// ---------------------------------------------------------------------------
// PNR and GMM filters
// ---------------------------------------------------------------------------

/// `np.percentile(x, [q0, q1])` peak to peak (`method="linear"`): exact order statistics,
/// the difference of the neighbours in float32 and the interpolation in float64, as numpy's
/// `_lerp` on a float32 array.
pub fn ptp_q(x: &mut [f32], q: (f64, f64)) -> f64 {
    x.sort_unstable_by(f32::total_cmp);
    let n = x.len();
    let at = |qq: f64| {
        let vi = (n - 1) as f64 * (qq / 100.0);
        let (i, g) = if vi >= (n - 1) as f64 {
            (n - 1, 0.0)
        } else if vi < 0.0 {
            (0, 0.0)
        } else {
            let fl = vi.floor();
            (fl as usize, vi - fl)
        };
        let (a, b) = (x[i], x[(i + 1).min(n - 1)]);
        let d = f64::from(b - a);
        if g >= 0.5 {
            f64::from(b) - d * (1.0 - g)
        } else {
            f64::from(a) + d * g
        }
    };
    at(q.1) - at(q.0)
}

/// `seeds.seed_ptp` of flat `pixels`.
pub fn seed_ptp(f: &Factors, pixels: &[usize], q: (f64, f64)) -> Vec<f64> {
    let mut tr = vec![0f32; f.t];
    pixels
        .iter()
        .map(|&px| {
            f.trace(px, &mut tr);
            ptp_q(&mut tr, q)
        })
        .collect()
}

/// `seeds.seed_pnr`: ptp over `6.18 sigma` (a non-positive sigma replaced by the median).
pub fn seed_pnr(f: &Factors, pixels: &[usize], ptp: &[f64]) -> Vec<f64> {
    let mut med: Option<f64> = None;
    pixels
        .iter()
        .zip(ptp)
        .map(|(&px, &v)| {
            let mut s = f64::from(f.sigma(px));
            if s <= 0.0 {
                s = *med.get_or_insert_with(|| f64::from(f.sigma_median()));
            }
            v / (6.18 * s)
        })
        .collect()
}

/// `pnr_refine`'s mask: the upper component of a 2-GMM on log PNR (`"auto"`), or `pnr >
/// thres`. The mixture comes back for `"auto"`.
pub fn pnr_mask(pnr: &[f64], thres: &serde_json::Value) -> Result<(Vec<bool>, Option<Gmm1D>), String> {
    match thres {
        serde_json::Value::String(s) if s == "auto" => {
            let dat: Vec<f64> = pnr.iter().map(|&v| v.max(f64::MIN_POSITIVE).ln()).collect();
            let g = fit_gmm_1d(&dat, 2)?;
            let top = argmax(&g.means);
            Ok((g.predict(&dat).iter().map(|&l| l == top).collect(), Some(g)))
        }
        serde_json::Value::Number(n) => {
            let t = n.as_f64().ok_or("pnr_thres")?;
            Ok((pnr.iter().map(|&v| v > t).collect(), None))
        }
        v => Err(format!("unknown pnr threshold {v}")),
    }
}

/// `seeds.gmm_refine` (2 components, the top one valid, above the lowest mean) on `ptp`.
pub fn gmm_mask(ptp: &[f64]) -> Result<(Vec<bool>, Gmm1D), String> {
    let g = fit_gmm_1d(ptp, 2)?;
    // np.argsort(means)[-1]: the larger mean, the second on a tie
    let top = if g.means[1] >= g.means[0] { 1 } else { 0 };
    let lo = g.means.iter().copied().fold(f64::INFINITY, f64::min);
    let mask = g
        .predict(ptp)
        .iter()
        .zip(ptp)
        .map(|(&l, &v)| l == top && v > lo)
        .collect();
    Ok((mask, g))
}

// ---------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------

/// Everything the stage produced (the arrays `ref_seeds.py` saves).
pub struct SeedsResult {
    pub init: Vec<[i64; 3]>,
    pub pnr: Vec<f64>,
    pub mask_pnr: Vec<bool>,
    pub gmm_pnr: Option<Gmm1D>,
    pub mask_mrg: Vec<bool>, // over the seeds the merge saw
    pub merge_pairs: Vec<[usize; 2]>,
    pub merge_corr: Vec<f64>,
    pub merged: Vec<[i64; 2]>,
    pub fc: FactorCorr,      // kept for init (its `norm2` is an output)
    pub mask_gmm: Vec<bool>, // opt-in gmm_refine on the PNR survivors
    pub gmm_refine: Gmm1D,
    pub seconds: Vec<(&'static str, f64)>,
}

/// Run the seed stage on `f` (`now()` gives seconds, for the per-step timings).
pub fn run(f: &Factors, p: &SeedParams, now: &dyn Fn() -> f64) -> Result<SeedsResult, String> {
    let mut seconds = Vec::new();
    let mut t = now();
    let mut lap = |name: &'static str, seconds: &mut Vec<(&'static str, f64)>| {
        let n = now();
        seconds.push((name, n - t));
        t = n;
    };
    let mut fc = FactorCorr::new(f, p.noise_freq);
    lap("corr_setup", &mut seconds);
    let init = seeds_init(f, p)?;
    lap("seeds_init", &mut seconds);
    let px: Vec<usize> = init.iter().map(|s| s[0] as usize * f.w + s[1] as usize).collect();
    let ptp = seed_ptp(f, &px, PNR_Q);
    let pnr = seed_pnr(f, &px, &ptp);
    let (mask_pnr, gmm_pnr) = pnr_mask(&pnr, &p.pnr_thres)?;
    if !mask_pnr.iter().any(|&m| m) {
        return Err("no seeds survived pnr_refine".into());
    }
    let kept: Vec<usize> = (0..init.len()).filter(|&i| mask_pnr[i]).collect();
    let kept_ptp: Vec<f64> = kept.iter().map(|&i| ptp[i]).collect();
    lap("pnr_refine", &mut seconds);
    let (mask_gmm, gmm_refine) = gmm_mask(&kept_ptp)?;
    lap("gmm_refine", &mut seconds);
    let unmerged: Vec<usize> = if p.gmm_refine {
        if !mask_gmm.iter().any(|&m| m) {
            return Err("no seeds survived gmm_refine".into());
        }
        kept.iter()
            .zip(&mask_gmm)
            .filter(|(_, &m)| m)
            .map(|(&i, _)| i)
            .collect()
    } else {
        kept
    };
    let hw: Vec<[i64; 2]> = unmerged.iter().map(|&i| [init[i][0], init[i][1]]).collect();
    let (mask_mrg, merge_pairs, merge_corr) = corr::seeds_merge(&mut fc, f, &hw, p.merge_dist, p.merge_corr);
    let merged = hw
        .iter()
        .zip(&mask_mrg)
        .filter(|(_, &m)| m)
        .map(|(&s, _)| s)
        .collect();
    lap("seeds_merge", &mut seconds);
    Ok(SeedsResult {
        init,
        pnr,
        mask_pnr,
        gmm_pnr,
        mask_mrg,
        merge_pairs,
        merge_corr,
        merged,
        fc,
        mask_gmm,
        gmm_refine,
        seconds,
    })
}

fn gmm_rows(g: &Gmm1D) -> Vec<f64> {
    [&g.weights, &g.means, &g.variances]
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect()
}

/// The stage's results as `.npy` files named like `ref_seeds.py`'s `ref/` (plus
/// `summary.json`).
pub fn outputs(r: &SeedsResult, mut summary: serde_json::Value) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let init: Vec<i64> = r.init.iter().flatten().copied().collect();
    out.push(("seeds_init.npy".into(), npy::write(&[r.init.len(), 3], &init)));
    out.push(("pnr.npy".into(), npy::write(&[r.pnr.len()], &r.pnr)));
    out.push((
        "mask_pnr.npy".into(),
        npy::write(&[r.mask_pnr.len()], &r.mask_pnr),
    ));
    if let Some(g) = &r.gmm_pnr {
        out.push((
            "gmm_pnr.npy".into(),
            npy::write(&[3, g.means.len()], &gmm_rows(g)),
        ));
    }
    out.push((
        "mask_mrg.npy".into(),
        npy::write(&[r.mask_mrg.len()], &r.mask_mrg),
    ));
    let pairs: Vec<i64> = r.merge_pairs.iter().flatten().map(|&v| v as i64).collect();
    out.push((
        "merge_pairs.npy".into(),
        npy::write(&[r.merge_pairs.len(), 2], &pairs),
    ));
    out.push((
        "merge_corr.npy".into(),
        npy::write(&[r.merge_corr.len()], &r.merge_corr),
    ));
    let merged: Vec<i64> = r.merged.iter().flatten().copied().collect();
    out.push((
        "seeds_merged.npy".into(),
        npy::write(&[r.merged.len(), 2], &merged),
    ));
    out.push((
        "corr_norm2.npy".into(),
        npy::write(&[r.fc.norm2.len()], &r.fc.norm2),
    ));
    out.push((
        "mask_gmm.npy".into(),
        npy::write(&[r.mask_gmm.len()], &r.mask_gmm),
    ));
    let g = &r.gmm_refine;
    out.push((
        "gmm_refine.npy".into(),
        npy::write(&[3, g.means.len()], &gmm_rows(g)),
    ));
    let mut rng = gmm::Mt19937::new(gmm::SEED);
    let draws: Vec<f64> = (0..3).map(|_| rng.random_sample()).collect();
    out.push(("gmm_draws.npy".into(), npy::write(&[3], &draws)));
    summary["seeds_init"] = r.init.len().into();
    summary["seeds_pnr"] = r.mask_pnr.iter().filter(|&&m| m).count().into();
    summary["seeds_merged"] = r.merged.len().into();
    summary["merge_pairs"] = r.merge_pairs.len().into();
    summary["gmm_refine_optin_kept"] = r.mask_gmm.iter().filter(|&&m| m).count().into();
    if let Some(g) = &r.gmm_pnr {
        summary["gmm_pnr_iterations"] = g.iterations.into();
    }
    summary["stages"] = r
        .seconds
        .iter()
        .map(|&(n, s)| (n.to_string(), serde_json::Value::from(s)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    out.push((
        "summary.json".into(),
        serde_json::to_vec_pretty(&summary).unwrap(),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_make_disks() {
        let rg = rings(3);
        let n: Vec<usize> = (0..=3).map(|r| rg[..=r].iter().map(Vec::len).sum()).collect();
        // seeds.disk(r).sum() for r = 0..3
        assert_eq!(n, vec![1, 5, 13, 29]);
    }

    #[test]
    fn plateau_and_contrast() {
        let (h, w) = (9, 9);
        let mut fm = vec![10u8; h * w];
        fm[4 * w + 4] = 20;
        fm[4 * w + 5] = 20; // a two-pixel plateau: both are maxima
        let u = local_max_union(&fm, h, w, 2, 4, 3.0);
        assert!(u[4 * w + 4] && u[4 * w + 5]);
        assert_eq!(u.iter().filter(|&&b| b).count(), 2);
        assert_eq!(component_medians(&u, h, w), vec![(4, 4)]);
        // contrast 10 is not above diff 10
        assert!(!local_max_union(&fm, h, w, 2, 4, 10.0).iter().any(|&b| b));
    }

    #[test]
    fn percentile_matches_numpy() {
        // np.percentile(np.arange(3000, dtype=np.float32), [0.1, 99.9]) -> 2.999, 2996.001
        let mut x: Vec<f32> = (0..3000).rev().map(|v| v as f32).collect();
        let v = ptp_q(&mut x, PNR_Q);
        assert!((v - (2996.001 - 2.999)).abs() < 1e-9);
    }
}
