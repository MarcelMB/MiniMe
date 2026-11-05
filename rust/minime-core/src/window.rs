//! One window's seeds → init → CNMF from its saved factors (`factors/wNN/`), for window-2
//! reruns and the auto-tune search: `Window2Session` + `Sweep._init`/`_cnmf` of the Python app,
//! with the stage results cached by the parameters that shape them (`Sweep._pnr_of`,
//! `LocalSearch._stage`).

use serde::Serialize;

use crate::cnmf::{self, CnmfParams};
use crate::corr::{seeds_merge, FactorCorr};
use crate::factors::Factors;
use crate::init::{self, Csr, InitParams};
use crate::pass1::{seed_block_edges, tile_grid};
use crate::score::{self, Checks};
use crate::seeds::{gmm_mask, pnr_mask, seed_pnr, seed_ptp, seeds_init, SeedParams, PNR_Q};
use crate::stitch::window_inputs;
use crate::tune;

/// The run's own seed counts, the reference of the cap (`Sweep._cap`); 0 = no cap.
pub struct Caps {
    pub seeds_pnr: usize,
    pub seeds_merged: usize,
}

#[derive(Serialize, Debug)]
pub struct TrialOut {
    pub counts: serde_json::Map<String, serde_json::Value>,
    pub score: Option<score::Summary>,
    pub error: Option<String>,
    pub size10: Vec<usize>,
    pub seconds: [f64; 3], // seeds, init, cnmf
}

struct SeedsStage {
    key: String,
    unmerged: Vec<[i64; 2]>,
    seeds_init: usize,
}

struct InitStage {
    key: String,
    a: Csr,
    c: Vec<f32>,
    units_init_a: usize,
    seeds_merged: usize,
}

pub struct WindowSession {
    pub t0: usize,
    pub t1: usize,
    pub f: Factors,
    seeds_wnd: usize,
    seeds_stp: usize,
    fc: Option<FactorCorr>,
    pnr: Vec<f64>, // per pixel, NaN = not read yet (Sweep._pnr_of)
    seeds: Option<SeedsStage>,
    init: Option<InitStage>,
    reference: Vec<[f64; 2]>,
    last: Option<(Csr, Vec<f64>)>, // final A and the peak of each YrA, for the stitch
}

fn obj<'a>(
    s: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    s[key]
        .as_object()
        .ok_or_else(|| format!("session.json: no \"{key}\" object"))
}

fn usize_of(m: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<usize, String> {
    m.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("session.json: {key} is not a whole number"))
}

/// The refusal for a window folder that lacks a file only newer runs save.
fn missing(name: &str) -> String {
    format!("this run was made before auto-tune: run it again to tune or rerun it (factors/wNN/{name} is missing)")
}

impl WindowSession {
    /// The window `[t0, t1)` of the run whose `session.json` is `run_session`, from the files of
    /// its `factors/wNN/` folder (`get(name)`: `W.npy`, `V.npy`, `mean.npy`, `sigma.npy`,
    /// `rank.npy`, `block_max.npy`, `max_proj.npy`, `energy.npy`).
    pub fn new(
        run_session: &[u8],
        t0: usize,
        t1: usize,
        get: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<Self, String> {
        let s: serde_json::Value =
            serde_json::from_slice(run_session).map_err(|e| format!("session.json: {e}"))?;
        let p1 = obj(&s, "pass1")?;
        let (h, w, tile) = (usize_of(p1, "H")?, usize_of(p1, "W")?, usize_of(p1, "tile")?);
        let (seeds_wnd, seeds_stp) = (usize_of(p1, "seeds_wnd")?, usize_of(p1, "seeds_stp")?);
        if t1 <= t0 {
            return Err(format!("window {t0}..{t1} has no frames"));
        }
        let t = t1 - t0;
        let grid = tile_grid(h, w, tile);
        let wf = crate::npy::read_f32(&get("W.npy")?)?.data;
        let vf = crate::npy::read_f32(&get("V.npy")?)?.data;
        let mean = crate::npy::read_f32(&get("mean.npy")?)?.data;
        let sigma = crate::npy::read_f32(&get("sigma.npy").map_err(|_| missing("sigma.npy"))?)?.data;
        let rank: Vec<usize> = crate::npy::read_i64(&get("rank.npy")?)?
            .data
            .iter()
            .map(|&r| r as usize)
            .collect();
        let energy = crate::npy::read_f64(&get("energy.npy").map_err(|_| missing("energy.npy"))?)?.data;
        let tiles = Factors::split_flat(&grid, t, &wf, &vf, &mean, &sigma, &energy, &rank)?;
        let max_proj = crate::npy::read_u8(&get("max_proj.npy")?)?.data;
        let block_max = crate::npy::read_u8(&get("block_max.npy")?)?.data;
        let edges = seed_block_edges(t, seeds_wnd, seeds_stp);
        let f = Factors::new(h, w, t, tile, tiles, max_proj, block_max, edges)?;
        Ok(Self {
            t0,
            t1,
            f,
            seeds_wnd,
            seeds_stp,
            fc: None,
            pnr: vec![f64::NAN; h * w],
            seeds: None,
            init: None,
            reference: Vec::new(),
            last: None,
        })
    }

    fn pnr_of(&mut self, pixels: &[usize]) -> Vec<f64> {
        let new: Vec<usize> = pixels
            .iter()
            .copied()
            .filter(|&px| self.pnr[px].is_nan())
            .collect();
        if !new.is_empty() {
            let ptp = seed_ptp(&self.f, &new, PNR_Q);
            for (px, v) in new.iter().zip(seed_pnr(&self.f, &new, &ptp)) {
                self.pnr[*px] = v;
            }
        }
        pixels.iter().map(|&px| self.pnr[px]).collect()
    }

    /// `Sweep._cap`: a count over `factor` x the run's own is an error (no reference, no cap).
    fn cap(what: &str, n: usize, caps: Option<&Caps>, factor: f64) -> Result<(), String> {
        let r = match (what, caps) {
            ("seeds_pnr", Some(c)) => c.seeds_pnr,
            ("seeds_merged", Some(c)) => c.seeds_merged,
            _ => return Ok(()),
        };
        if r > 0 && n as f64 > factor * r as f64 {
            return Err(format!(
                "ValueError: too many {what}: {n} > {factor} x the run's {r}"
            ));
        }
        Ok(())
    }

    /// The three stage parameter sets of `session` and the cache keys of the seed stage (before
    /// the merge, which does not see `merge_dist` / `merge_corr`) and of the init stage.
    fn stage_params(
        &self,
        session: &[u8],
    ) -> Result<(SeedParams, InitParams, CnmfParams, String, String), String> {
        let files = window_inputs(session, &[], self.t0, self.t1)?;
        let get = |name: &str| {
            files
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, b)| b.as_slice())
                .ok_or_else(|| format!("no {name}"))
        };
        let sp: SeedParams =
            serde_json::from_slice(get("seeds.json")?).map_err(|e| format!("seeds.json: {e}"))?;
        let ip: InitParams =
            serde_json::from_slice(get("init.json")?).map_err(|e| format!("init.json: {e}"))?;
        let cp: CnmfParams =
            serde_json::from_slice(get("cnmf.json")?).map_err(|e| format!("cnmf.json: {e}"))?;
        let mut sv: serde_json::Value =
            serde_json::from_slice(get("seeds.json")?).map_err(|e| format!("seeds.json: {e}"))?;
        sv["merge_dist"] = serde_json::Value::Null;
        sv["merge_corr"] = serde_json::Value::Null;
        let seeds_key = sv.to_string();
        let init_key = format!(
            "{}{}",
            String::from_utf8_lossy(get("seeds.json")?),
            String::from_utf8_lossy(get("init.json")?)
        );
        Ok((sp, ip, cp, seeds_key, init_key))
    }

    /// Seeds before the merge: `seeds_init`, the PNR filter (PNR cached per pixel) and the
    /// opt-in `gmm_refine`.
    fn seed_stage(&mut self, sp: &SeedParams, key: &str) -> Result<(), String> {
        let init_seeds = seeds_init(&self.f, sp)?;
        if init_seeds.is_empty() {
            return Err("ValueError: no seeds survived pnr_refine".into());
        }
        let px: Vec<usize> = init_seeds
            .iter()
            .map(|s| s[0] as usize * self.f.w + s[1] as usize)
            .collect();
        let pnr = self.pnr_of(&px);
        let (mask, _) = pnr_mask(&pnr, &sp.pnr_thres)?;
        if !mask.iter().any(|&m| m) {
            return Err("ValueError: no seeds survived pnr_refine".into());
        }
        let mut kept: Vec<usize> = (0..init_seeds.len()).filter(|&i| mask[i]).collect();
        if sp.gmm_refine {
            let kpx: Vec<usize> = kept.iter().map(|&i| px[i]).collect();
            let (gm, _) = gmm_mask(&seed_ptp(&self.f, &kpx, PNR_Q))?;
            if !gm.iter().any(|&m| m) {
                return Err("ValueError: no seeds survived gmm_refine".into());
            }
            kept = kept
                .iter()
                .zip(&gm)
                .filter(|(_, &m)| m)
                .map(|(&i, _)| i)
                .collect();
        }
        let unmerged = kept
            .iter()
            .map(|&i| [init_seeds[i][0], init_seeds[i][1]])
            .collect();
        self.seeds = Some(SeedsStage {
            key: key.to_string(),
            unmerged,
            seeds_init: init_seeds.len(),
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        session: &[u8],
        caps: Option<&Caps>,
        factor: f64,
        now: &dyn Fn() -> f64,
        counts: &mut serde_json::Map<String, serde_json::Value>,
        seconds: &mut [f64; 3],
    ) -> Result<(Csr, Vec<f32>), String> {
        let (sp, ip, cp, seeds_key, init_key) = self.stage_params(session)?;
        if sp.seeds_wnd != self.seeds_wnd || sp.seeds_stp != self.seeds_stp {
            return Err(format!(
                "ValueError: seeds_wnd {} / seeds_stp {}: the seed windows do not fall on the block edges pass 1 stored (built with seeds_wnd {}, seeds_stp {}); a change needs a full rerun: minime run",
                sp.seeds_wnd, sp.seeds_stp, self.seeds_wnd, self.seeds_stp
            ));
        }
        if self.fc.as_ref().map(|fc| fc.noise_freq) != Some(sp.noise_freq) {
            self.fc = None;
            self.fc = Some(FactorCorr::new(&self.f, sp.noise_freq));
        }
        let mut t = now();
        let mut lap = |i: usize, seconds: &mut [f64; 3]| {
            let n = now();
            seconds[i] += n - t;
            t = n;
        };
        // seeds before the merge, cached by their parameters
        if self.seeds.as_ref().map(|s| s.key.as_str()) != Some(seeds_key.as_str()) {
            self.seeds = None;
            self.init = None;
            self.seed_stage(&sp, &seeds_key)?;
        }
        let st = self.seeds.as_ref().unwrap();
        counts.insert("seeds_init".into(), st.seeds_init.into());
        counts.insert("seeds_pnr".into(), st.unmerged.len().into());
        Self::cap("seeds_pnr", st.unmerged.len(), caps, factor)?;
        // merge and init, cached by the seed and init parameters
        if self.init.as_ref().map(|s| s.key.as_str()) != Some(init_key.as_str()) {
            self.init = None;
            let fc = self.fc.as_mut().unwrap();
            let unmerged = &self.seeds.as_ref().unwrap().unmerged;
            let (keep, _, _) = seeds_merge(fc, &self.f, unmerged, sp.merge_dist, sp.merge_corr);
            let merged: Vec<[i64; 2]> = unmerged
                .iter()
                .zip(&keep)
                .filter(|(_, &m)| m)
                .map(|(&s, _)| s)
                .collect();
            counts.insert("seeds_merged".into(), merged.len().into());
            Self::cap("seeds_merged", merged.len(), caps, factor)?;
            lap(0, seconds);
            let r = init::run(fc, &self.f, &merged, &ip, now)?;
            self.init = Some(InitStage {
                key: init_key,
                a: r.a,
                c: r.c,
                units_init_a: r.seeds,
                seeds_merged: merged.len(),
            });
            lap(1, seconds);
        } else {
            lap(0, seconds);
            let n = self.init.as_ref().unwrap().seeds_merged;
            counts.insert("seeds_merged".into(), n.into());
            Self::cap("seeds_merged", n, caps, factor)?;
        }
        let ist = self.init.as_ref().unwrap();
        counts.insert("units_initA".into(), ist.units_init_a.into());
        counts.insert("units_init_merge".into(), ist.a.k.into());
        let r = cnmf::run(&self.f, &ist.a, &ist.c, &cp, now)?;
        lap(2, seconds);
        let last = r.last();
        let a = last.merge.a.clone();
        // the loop merges YrA as `c` and C as `extra` (see `cnmf::outputs`)
        let yra = last.merge.c.clone();
        counts.insert("units_final".into(), a.k.into());
        let t_ = self.f.t;
        let bright: Vec<f64> = (0..a.k)
            .map(|i| {
                f64::from(
                    yra[i * t_..(i + 1) * t_]
                        .iter()
                        .fold(f32::NEG_INFINITY, |m, &v| m.max(v)),
                )
            })
            .collect();
        self.last = Some((a.clone(), bright));
        Ok((a, yra))
    }

    /// One setting on this window, scored (`Sweep._try`): `reference` makes it the trial whose
    /// good units later ones are charged for losing. A failed setting carries the Python error
    /// text and no score.
    pub fn trial(
        &mut self,
        session: &[u8],
        checks: &Checks,
        reference: bool,
        caps: Option<&Caps>,
        cap_factor: f64,
        now: &dyn Fn() -> f64,
    ) -> TrialOut {
        let mut counts = serde_json::Map::new();
        let mut seconds = [0.0; 3];
        let failed = |counts, seconds, e: String| {
            let error = if e.starts_with("ValueError: ") || e.starts_with("RuntimeError: ") {
                e
            } else {
                format!("RuntimeError: {e}")
            };
            TrialOut {
                counts,
                score: None,
                error: Some(error),
                size10: vec![],
                seconds,
            }
        };
        match self.run(session, caps, cap_factor, now, &mut counts, &mut seconds) {
            Err(e) => failed(counts, seconds, e),
            Ok((a, yra)) => {
                let flags = match score::unit_flags(&a, &yra, self.f.h, self.f.w, checks) {
                    Ok(f) => f,
                    Err(e) => return failed(counts, seconds, e),
                };
                if reference {
                    self.reference = score::good_centroids(&flags);
                }
                let lost = score::lost_units(&self.reference, &flags, checks.keep_dist * checks.diameter);
                TrialOut {
                    counts,
                    score: Some(score::summarize(&flags, lost)),
                    error: None,
                    size10: flags.size10.clone(),
                    seconds,
                }
            }
        }
    }

    /// The last successful trial's final units as the stitch takes them.
    pub fn window_bytes(&self) -> Result<Vec<u8>, String> {
        let (a, bright) = self.last.as_ref().ok_or("no finished trial on this window")?;
        Ok(crate::poolio::window_to_bytes(a, bright))
    }

    /// `sweep.start_diff_thres`'s histogram on this window's block maxima: `n_windows` of its
    /// seed windows, spread over it.
    pub fn contrast_hist(
        &self,
        seeds_wnd: usize,
        seeds_stp: usize,
        seeds_max_wnd: usize,
        n_windows: usize,
    ) -> Result<Vec<i64>, String> {
        let ranges = tune::seed_window_ranges(&self.f.edges, self.f.t, seeds_wnd, seeds_stp)?;
        let ranges = tune::sample_ranges(&ranges, n_windows);
        Ok(tune::contrast_hist(
            &self.f.block_max,
            self.f.h,
            self.f.w,
            &ranges,
            2,
            seeds_max_wnd,
        ))
    }

    /// Bytes of the factors held (for the worker-count budget).
    pub fn bytes(&self) -> usize {
        self.f
            .tiles
            .iter()
            .map(|t| 4 * (t.w.len() + t.v.len() + t.mean.len() + t.sigma.len()))
            .sum::<usize>()
            + self.f.block_max.len()
            + self.f.max_proj.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32x32, 120-frame window of two Gaussian cells with transients, through pass 2's
    /// `compress_tile` (one 32-px tile) and pass 1's block maxima, as the browser saves them.
    fn tiny() -> (Vec<u8>, usize, usize, std::collections::HashMap<String, Vec<u8>>) {
        use crate::compress::{compress_tile, Gaussian, Pass2Params};
        let (h, w, t) = (32usize, 32usize, 120usize);
        let mut raw = vec![0u8; t * h * w];
        let mut s = 11u32;
        for fr in 0..t {
            for y in 0..h {
                for x in 0..w {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    let noise = ((s >> 16) % 5) as f64;
                    let mut v = 20.0 + noise;
                    for (cy, cx, phase) in [(10.0, 10.0, 0usize), (22.0, 20.0, 60usize)] {
                        let g = (-(((y as f64 - cy).powi(2) + (x as f64 - cx).powi(2)) / 8.0)).exp();
                        let amp = if (fr + phase) % 40 < 8 { 120.0 } else { 0.0 };
                        v += amp * g;
                    }
                    raw[fr * h * w + y * w + x] = v.min(255.0) as u8;
                }
            }
        }
        let p = Pass2Params {
            h,
            w,
            t,
            tile: 32,
            svd_k0: 8,
            svd_power_iter: 1,
            rank_rho_min: 0.2,
            rank_z: 4.0,
            noise_freq: 0.06,
            noise_frames: 0,
        };
        let tf = compress_tile(&raw, t, h * w, &p, &mut Gaussian::new(0)).unwrap();
        let edges = crate::pass1::seed_block_edges(t, 40, 20);
        let mut block_max = vec![0u8; (edges.len() - 1) * h * w];
        let mut max_proj = vec![0u8; h * w];
        for b in 0..edges.len() - 1 {
            for fr in edges[b]..edges[b + 1] {
                for px in 0..h * w {
                    let v = raw[fr * h * w + px];
                    block_max[b * h * w + px] = block_max[b * h * w + px].max(v);
                    max_proj[px] = max_proj[px].max(v);
                }
            }
        }
        let mut files = std::collections::HashMap::new();
        files.insert("W.npy".into(), crate::npy::write(&[tf.w.len()], &tf.w));
        files.insert("V.npy".into(), crate::npy::write(&[tf.v.len()], &tf.v));
        files.insert("s.npy".into(), crate::npy::write(&[tf.s.len()], &tf.s));
        files.insert("mean.npy".into(), crate::npy::write(&[tf.mean.len()], &tf.mean));
        files.insert(
            "sigma.npy".into(),
            crate::npy::write(&[tf.sigma.len()], &tf.sigma),
        );
        files.insert(
            "energy.npy".into(),
            crate::npy::write(&[tf.energy.len()], &tf.energy),
        );
        files.insert("rank.npy".into(), crate::npy::write(&[1], &[tf.rank as i64]));
        files.insert("k.npy".into(), crate::npy::write(&[1], &[tf.k as i64]));
        files.insert(
            "block_max.npy".into(),
            crate::npy::write(&[edges.len() - 1, h, w], &block_max),
        );
        files.insert("max_proj.npy".into(), crate::npy::write(&[h, w], &max_proj));
        let session = serde_json::json!({
            "window_frames": t,
            "pass1": {"H": h, "W": w, "median_ksize": 7, "tophat_wnd": 15, "motion_max_shift": 20, "motion_down": 2, "template_frames": 200, "motion_before_tophat": true, "motion_bandpass": [1.0, 6.0], "tile": 32, "seeds_wnd": 40, "seeds_stp": 20},
            "pass2": {"H": h, "W": w, "tile": 32, "svd_k0": 8, "svd_power_iter": 1, "rank_rho_min": 0.2, "rank_z": 4.0, "noise_freq": 0.06, "noise_frames": 0},
            "seeds": {"H": h, "W": w, "tile": 32, "seeds_wnd": 40, "seeds_stp": 20, "seeds_max_wnd": 6, "seeds_diff_thres": 3, "noise_freq": 0.06, "pnr_thres": 3.0, "gmm_refine": false, "merge_dist": 6, "merge_corr": 0.8},
            "init": {"H": h, "W": w, "tile": 32, "noise_freq": 0.06, "inita_corr": 0.8, "inita_wnd": 6, "init_merge_corr": 0.8, "merge_min_dist": 0.0},
            "cnmf": {"H": h, "W": w, "tile": 32, "noise_freq": 0.06, "dl_wnd": 4, "sparse_snr": 1.0, "size_thres": [4, null], "n_iter": 2, "background_clip": false, "loop_merge_corr": 0.9, "merge_min_dist": 0.0},
            "stitch": {"cos_thres": 0.7}
        });
        (serde_json::to_vec(&session).unwrap(), 0, t, files)
    }

    fn with(session: &[u8], key: &str, k2: &str, v: serde_json::Value) -> Vec<u8> {
        let mut s: serde_json::Value = serde_json::from_slice(session).unwrap();
        s[key][k2] = v;
        serde_json::to_vec(&s).unwrap()
    }

    #[test]
    fn a_trial_finds_the_two_cells_and_caches_stages() {
        let (session, t0, t1, files) = tiny();
        let mut ws = WindowSession::new(&session, t0, t1, &mut |n| {
            files.get(n).cloned().ok_or(format!("no {n}"))
        })
        .unwrap();
        let checks = crate::score::Checks::new(5.0).unwrap();
        let now = || 0.0;
        let r = ws.trial(&session, &checks, true, None, 5.0, &now);
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.counts["units_final"], 2);
        assert_eq!(r.score.as_ref().unwrap().lost, 0);
        // the same seed parameters again: the seed stage comes from the cache (its lap is 0)
        let r2 = ws.trial(
            &with(&session, "cnmf", "sparse_snr", 0.7.into()),
            &checks,
            false,
            None,
            5.0,
            &now,
        );
        assert!(r2.error.is_none());
        assert_eq!(r2.counts["seeds_merged"], r.counts["seeds_merged"]);
        assert!(ws.window_bytes().is_ok());
        // a cap far below the run's own seed count fails fast with Python's text
        let n = r.counts["seeds_pnr"].as_u64().unwrap() as usize;
        let caps = Caps {
            seeds_pnr: n,
            seeds_merged: n,
        };
        let r3 = ws.trial(&session, &checks, false, Some(&caps), 0.1, &now);
        assert!(
            r3.error
                .as_deref()
                .unwrap()
                .starts_with("ValueError: too many seeds_pnr"),
            "{:?}",
            r3.error
        );
    }

    #[test]
    fn a_setting_with_no_seeds_is_a_failed_trial_not_a_panic() {
        let (session, t0, t1, files) = tiny();
        let mut ws = WindowSession::new(&session, t0, t1, &mut |n| {
            files.get(n).cloned().ok_or(format!("no {n}"))
        })
        .unwrap();
        let checks = crate::score::Checks::new(5.0).unwrap();
        let r = ws.trial(
            &with(&session, "seeds", "seeds_diff_thres", 250.into()),
            &checks,
            false,
            None,
            5.0,
            &|| 0.0,
        );
        assert!(
            r.score.is_none() && r.error.as_deref().unwrap().contains("no seeds"),
            "{:?}",
            r.error
        );
    }

    #[test]
    fn contrast_hist_has_a_tail_at_the_cells() {
        let (session, t0, t1, files) = tiny();
        let ws = WindowSession::new(&session, t0, t1, &mut |n| {
            files.get(n).cloned().ok_or(format!("no {n}"))
        })
        .unwrap();
        let hist = ws.contrast_hist(40, 20, 6, 16).unwrap();
        assert!(hist.iter().sum::<i64>() > 0 && hist.len() > 20);
    }

    #[test]
    fn a_window_without_sigma_or_energy_is_refused() {
        for name in ["sigma.npy", "energy.npy"] {
            let (session, t0, t1, mut files) = tiny();
            files.remove(name);
            let e = WindowSession::new(&session, t0, t1, &mut |n| {
                files.get(n).cloned().ok_or(format!("no {n}"))
            })
            .err()
            .unwrap();
            assert!(
                e.starts_with("this run was made before auto-tune")
                    && e.contains(&format!("factors/wNN/{name} is missing")),
                "{e}"
            );
        }
    }
}
