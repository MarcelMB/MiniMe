//! Bounded-memory streaming MiniMe core, shared by the native check harness and the
//! browser (wasm) build. Python (`src/minime`, `experiments/streaming/`) is the reference.

pub mod avi;
pub mod butter;
pub mod filters;
pub mod linalg;
pub mod motion;
pub mod nnqp;
pub mod npy;
pub mod preprocess;
pub mod stream;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use stream::{Csr, Params, StreamPass};

/// Everything one streaming pass reads besides the raw frames (`ref_dump.py`'s `inputs/`).
pub struct Inputs {
    pub params: Params,
    pub a: Csr,
    pub shifts: Vec<f32>, // (T, 2) dy, dx
    pub ymean: Vec<f64>,
    pub sn: Vec<f64>,
}

impl Inputs {
    /// Build from file contents by name (`params.json`, `A_data.npy`, ...).
    pub fn load(get: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>) -> Result<Self, String> {
        let params: Params =
            serde_json::from_slice(&get("params.json")?).map_err(|e| format!("params.json: {e}"))?;
        let p = params.h * params.w;
        let data = npy::read_f64(&get("A_data.npy")?)?.data;
        let indices: Vec<usize> = npy::read_i64(&get("A_indices.npy")?)?
            .data
            .iter()
            .map(|&v| v as usize)
            .collect();
        let indptr: Vec<usize> = npy::read_i64(&get("A_indptr.npy")?)?
            .data
            .iter()
            .map(|&v| v as usize)
            .collect();
        let a = Csr {
            nrows: indptr.len() - 1,
            ncols: p,
            indptr,
            indices,
            data,
        };
        let shifts = npy::read_f32(&get("shifts.npy")?)?.data;
        let ymean = npy::read_f64(&get("ymean.npy")?)?.data;
        let sn = npy::read_f64(&get("sn.npy")?)?.data;
        if ymean.len() != p || sn.len() != p {
            return Err(format!(
                "ymean/sn hold {}/{} pixels, expected {p}",
                ymean.len(),
                sn.len()
            ));
        }
        if shifts.len() < 2 * params.t {
            return Err(format!("{} shifts for {} frames", shifts.len() / 2, params.t));
        }
        Ok(Self {
            params,
            a,
            shifts,
            ymean,
            sn,
        })
    }

    pub fn prep(&self) -> preprocess::Prep {
        preprocess::Prep {
            median_ksize: self.params.median_ksize,
            tophat_wnd: self.params.tophat_wnd,
            motion_before_tophat: self.params.motion_before_tophat,
        }
    }
}

fn csr_files(out: &mut Vec<(String, Vec<u8>)>, name: &str, a: &Csr) {
    let data: Vec<f32> = a.data.iter().map(|&v| v as f32).collect();
    let indices: Vec<i32> = a.indices.iter().map(|&v| v as i32).collect();
    let indptr: Vec<i64> = a.indptr.iter().map(|&v| v as i64).collect();
    out.push((format!("{name}_data.npy"), npy::write(&[data.len()], &data)));
    out.push((
        format!("{name}_indices.npy"),
        npy::write(&[indices.len()], &indices),
    ));
    out.push((format!("{name}_indptr.npy"), npy::write(&[indptr.len()], &indptr)));
}

/// The pass's results as `.npy` files named like `ref_dump.py`'s `ref/` (plus
/// `summary.json`), so one comparison script checks native and browser runs alike.
pub fn finish(
    pass: &StreamPass,
    inp: &Inputs,
    timing: serde_json::Value,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let k = pass.model.k;
    let t = pass.t;
    let p = pass.params.h * pass.params.w;
    let mut out = Vec::new();
    let mut c_kt = vec![0f32; k * t];
    for ti in 0..t {
        for i in 0..k {
            c_kt[i * t + ti] = pass.c[ti * k + i];
        }
    }
    out.push(("C.npy".into(), npy::write(&[k, t], &c_kt)));
    out.push(("f.npy".into(), npy::write(&[t], &pass.f)));
    out.push(("b.npy".into(), npy::write(&[p], &pass.model.b)));
    let sup = &pass.support;
    let pix: Vec<i64> = sup.pix.iter().map(|&v| v as i64).collect();
    let unit: Vec<i64> = sup.unit.iter().map(|&v| v as i64).collect();
    out.push(("support_pix.npy".into(), npy::write(&[pix.len()], &pix)));
    out.push(("support_unit.npy".into(), npy::write(&[unit.len()], &unit)));
    let tot = pass.total();
    out.push(("stats_n.npy".into(), npy::write(&[1], &[tot.n as i64])));
    out.push(("stats_L.npy".into(), npy::write(&[tot.l.len()], &tot.l)));
    out.push(("stats_M.npy".into(), npy::write(&[k, k], &tot.m)));
    out.push(("stats_csum.npy".into(), npy::write(&[k], &tot.csum)));
    out.push(("stats_ysum.npy".into(), npy::write(&[p], &tot.ysum)));
    out.push(("stats_dsq.npy".into(), npy::write(&[p], &tot.dsq)));
    out.push(("stats_Mlp.npy".into(), npy::write(&[k, k], &tot.mlp)));
    out.push(("stats_lsum.npy".into(), npy::write(&[k], &tot.lsum)));
    let sos: Vec<f64> = pass.lowpass.sos().iter().flatten().copied().collect();
    out.push(("sos.npy".into(), npy::write(&[sos.len() / 6, 6], &sos)));
    let pr = &inp.params;
    let sp = stream::spatial_from_stats(k, p, sup, &tot, &inp.sn, pr.sparse_snr, pr.size_thres)?;
    csr_files(&mut out, "Asp", &sp.a);
    out.push(("mask.npy".into(), npy::write(&[sp.mask.len()], &sp.mask)));
    out.push((
        "norm_fac.npy".into(),
        npy::write(&[sp.norm_fac.len()], &sp.norm_fac),
    ));
    let merged = stream::merge_from_stats(&sp.a, &tot, k, &sp.mask, pr.loop_merge_corr);
    csr_files(&mut out, "Amerged", &merged);
    let summary = serde_json::json!({
        "T": t,
        "K": k,
        "kept": sp.mask.iter().filter(|&&m| m).count(),
        "merged": merged.nrows,
        "state_mb": pass.state_bytes() as f64 / 1048576.0,
        "trace_solver": if pass.model.solver.is_cholesky() { "cholesky" } else { "pinv" },
        "timing": timing,
    });
    out.push((
        "summary.json".into(),
        serde_json::to_vec_pretty(&summary).unwrap(),
    ));
    Ok(out)
}
