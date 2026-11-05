//! wasm API for the GPU path of the per-frame stages (`web/src/gpu/`, on the pool's frame workers):
//! each stage says which kernels it needs (or why the GPU cannot give its bits), and takes the GPU's
//! results through the same code the CPU path runs after its own kernels (see `gpuplan`).

use crate::gpuplan::{kernels, motion_kernels, reference};
use crate::motion::{registration_image, Correlator, MotionParams};
use crate::npy;
use crate::pool::{StreamSolver, TemplateWorker};
use crate::wasm::Pass1Engine;
use js_sys::{Map, Uint8Array};
use wasm_bindgen::prelude::*;

fn json(k: Result<crate::gpuplan::Kernels, String>) -> Result<String, JsError> {
    let k = k.map_err(|e| JsError::new(&e))?;
    serde_json::to_string(&k).map_err(|e| JsError::new(&e.to_string()))
}

/// `n` downsampled images (`px` values each) of `all`, checked.
fn frames(all: &[f32], n: usize, px: usize) -> Result<std::slice::ChunksExact<'_, f32>, JsError> {
    if all.len() != n * px {
        return Err(JsError::new(&format!(
            "{} values for {n} images of {px}",
            all.len()
        )));
    }
    Ok(all.chunks_exact(px))
}

#[wasm_bindgen]
impl Pass1Engine {
    /// The GPU kernels this pass needs (`gpuplan::Kernels` as JSON), or an error saying why the GPU
    /// cannot give its bits.
    pub fn gpu_kernels(&self) -> Result<String, JsError> {
        json(motion_kernels(&self.p.motion))
    }
    /// The given shifts of window frames `a..a + n`, `(n, 2)` float32; empty when the pass estimates them.
    pub fn given_shifts(&self, a: usize, n: usize) -> Vec<f32> {
        self.shifts
            .as_ref()
            .map(|s| s[2 * a..2 * (a + n)].to_vec())
            .unwrap_or_default()
    }
    /// Estimate `n` frames' shifts from their downsampled registration images (`n` x the template's
    /// pixels): `(n, 2)` float64, as `push_raw` estimates them.
    pub fn estimate_down(&self, n: usize, down: &[f32]) -> Result<Vec<f64>, JsError> {
        let m = self.pass.motion().ok_or(JsError::new("no template"))?;
        let (ph, pw) = m.prep_shape();
        Ok(frames(down, n, ph * pw)?
            .flat_map(|f| m.estimate_down(f.to_vec()))
            .collect())
    }
    /// Fold in `n` frames the GPU corrected: `blk` their shifted, top-hatted pixels (`n * H * W`),
    /// `warped_down` their shifted registration images downsampled (for the quality), `est` the
    /// shifts `estimate_down` gave (empty when they were given) and `used` the shifts applied,
    /// float32. The same as `push_raw` on the frames.
    pub fn push_corrected(
        &mut self,
        n: usize,
        blk: &[u8],
        warped_down: &[f32],
        est: &[f64],
        used: &[f32],
    ) -> Result<(), JsError> {
        let px = self.p.motion.h * self.p.motion.w;
        if n > self.pass.room()
            || blk.len() != n * px
            || used.len() != 2 * n
            || !(est.is_empty() || est.len() == 2 * n)
        {
            return Err(JsError::new(&format!("{n} corrected frames do not fit the pass")));
        }
        let m = self.pass.motion().ok_or(JsError::new("no template"))?;
        let (ph, pw) = m.prep_shape();
        let quality: Vec<f32> = frames(warped_down, n, ph * pw)?
            .map(|f| m.quality_down(f.to_vec()))
            .collect();
        for i in 0..n {
            if !est.is_empty() {
                self.pass.push_estimate([est[2 * i], est[2 * i + 1]]);
            }
            self.pass.fold_corrected(
                &blk[i * px..(i + 1) * px],
                Some(quality[i]),
                [used[2 * i], used[2 * i + 1]],
            );
        }
        Ok(())
    }
}

#[wasm_bindgen]
impl TemplateWorker {
    pub fn gpu_kernels(&self) -> Result<String, JsError> {
        json(motion_kernels(&self.p))
    }
    /// `n` downsampled registration images (round 1), or shifted ones (round 2), prepared as
    /// `round1` / `round2` prepare theirs.
    pub fn prep_down(&self, n: usize, down: &[f32]) -> Result<Vec<f32>, JsError> {
        let (ph, pw) = self.corr.prep_shape();
        Ok(frames(down, n, ph * pw)?
            .flat_map(|f| self.corr.prep_down(f.to_vec()))
            .collect())
    }
    /// Round 2's shifts of `n` frames against the first template, from their downsampled
    /// registration images: `(n, 2)` float64.
    pub fn estimate_first_down(&self, n: usize, down: &[f32]) -> Result<Vec<f64>, JsError> {
        let first = self.first.as_ref().ok_or(JsError::new("set_first first"))?;
        let (ph, pw) = self.corr.prep_shape();
        Ok(frames(down, n, ph * pw)?
            .flat_map(|f| first.estimate_down(f.to_vec()))
            .collect())
    }
}

#[wasm_bindgen]
impl StreamSolver {
    pub fn gpu_kernels(&self) -> Result<String, JsError> {
        let p = &self.inp.params;
        json(kernels(
            p.h,
            p.w,
            p.median_ksize,
            p.tophat_wnd,
            p.motion_before_tophat,
            None,
        ))
    }
}

fn motion_params(params_json: &[u8]) -> Result<MotionParams, JsError> {
    serde_json::from_slice(params_json).map_err(|e| JsError::new(&format!("params.json: {e}")))
}

/// The GPU kernels a session's motion parameters (`params_json`, a stage's `params.json`) need, or
/// why the GPU cannot give their bits.
#[wasm_bindgen]
pub fn gpu_kernels(params_json: &[u8]) -> Result<String, JsError> {
    json(motion_kernels(&motion_params(params_json)?))
}

/// The CPU's result of every GPU kernel on one raw frame and shift (the self-test): `den`, `down`,
/// `blk`, `warped_down` (see `gpuplan::reference`).
#[wasm_bindgen]
pub fn gpu_reference(raw: &[u8], params_json: &[u8], dy: f32, dx: f32) -> Result<Map, JsError> {
    let out = reference(raw, &motion_params(params_json)?, dy, dx).map_err(|e| JsError::new(&e))?;
    let m = Map::new();
    for (name, bytes) in out {
        m.set(&name.into(), &Uint8Array::from(bytes.as_slice()));
    }
    Ok(m)
}

/// A template (`template.npy`) for a speed test of pass 1: the prepared registration image of one
/// raw frame (the estimate costs the same against any template).
#[wasm_bindgen]
pub fn bench_template(params_json: &[u8], raw: &[u8]) -> Result<Vec<u8>, JsError> {
    let p = motion_params(params_json)?;
    let corr = Correlator::new(p.h, p.w, p.motion_down).map_err(|e| JsError::new(&e))?;
    let img = corr.prep(&registration_image(raw, &p));
    let (ph, pw) = corr.prep_shape();
    Ok(npy::write(&[ph, pw], &img))
}
