//! wasm API for the worker pool (`web/src/stream/pool.ts`): the frame-parallel stages run on
//! several wasm instances, each in its own Web Worker, and one instance puts the results
//! together in the order a single worker would have produced them.
//!
//! - pass 1: a worker runs `Pass1Engine` on a chunk of the window (`start_at`), writes its own
//!   part of every tile file, and hands back its max projection, block maxima, shifts and
//!   quality (`chunk_*`); the window's engine folds them in (`merge_chunk`). Maxima are exact in
//!   any order, so the merged engine equals the one-worker engine bit for bit;
//! - streaming pass: a worker preprocesses and solves frames (`StreamSolver`: the same
//!   `prep().frame` and `TraceModel::solve` as `Engine::push_raw`); the accumulating engine folds
//!   the solved frames in frame order (`Engine::push_solved`), so every sum, the low-pass state
//!   and the traces are the one-worker pass's, bit for bit;
//! - the motion template: workers prepare the template frames (`TemplateWorker`: median, band-pass,
//!   and in round 2 the alignment to round 1's template), the motion engine adds the prepared
//!   frames in frame order (`push_template_prepped`);
//! - pass 2: a worker compresses any tile of the window's tile store (`compress_at`), the
//!   window's engine takes the factors in tile order (`set_tile`).

use crate::motion::{registration_image, Correlator, Motion, MotionParams};
use crate::npy;
use crate::stream::{lane_ranges, FoldLane, LaneRange, Support, TraceModel};
use crate::wasm::{map_bytes, unpad, Engine, MotionEngine, Pass1Engine, Pass2Engine};
use crate::{compress::TileFactors, Inputs};
use js_sys::Map;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl Pass1Engine {
    /// Begin at frame `a` of the window (a chunk worker): `push_raw` then takes the shifts
    /// given for frame `a`, `a + 1`, ... or estimates them.
    pub fn start_at(&mut self, a: usize) {
        self.pass.start_at(a);
    }
    pub fn chunk_max_proj(&self) -> Vec<u8> {
        self.pass.max_proj.clone()
    }
    pub fn chunk_block_max(&self) -> Vec<u8> {
        self.pass.block_max.clone()
    }
    /// The chunk's shifts, `(n, 2)` float32 (flat).
    pub fn chunk_shifts(&self) -> Vec<f32> {
        self.pass.shifts.clone()
    }
    /// The chunk's estimated shifts before the cast to float32, `(n, 2)` (flat); empty when
    /// they were given.
    pub fn chunk_shifts64(&self) -> Vec<f64> {
        self.pass.shifts64.clone()
    }
    pub fn chunk_quality(&self) -> Vec<f32> {
        self.pass.quality.clone()
    }

    /// Fold in a chunk another engine ran on frames `a..a + n` of this window.
    #[allow(clippy::too_many_arguments)]
    pub fn merge_chunk(
        &mut self,
        a: usize,
        n: usize,
        max_proj: &[u8],
        block_max: &[u8],
        shifts: &[f32],
        shifts64: &[f64],
        quality: &[f32],
    ) -> Result<(), JsError> {
        self.pass
            .merge_chunk(a, n, max_proj, block_max, shifts, shifts64, quality)
            .map_err(|e| JsError::new(&e))
    }
}

/// The streaming pass's per-frame work on a pool worker: median, shift and top-hat of a raw
/// frame, then the least squares on `[A; b]`. `solve_frame` each frame, then `take_*`.
#[wasm_bindgen]
pub struct StreamSolver {
    inp: Inputs,
    model: TraceModel,
    stride: usize,
    sol: Vec<f64>,
    y8: Vec<u8>,
    cc: Vec<f64>,
    f: Vec<f32>,
}

#[wasm_bindgen]
impl StreamSolver {
    /// `files`: the streaming pass's inputs (`params.json`, `A_*.npy`, `shifts.npy`, `ymean.npy`, `sn.npy`).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<StreamSolver, JsError> {
        let inp = Inputs::load(&mut |name| map_bytes(files, name)).map_err(|e| JsError::new(&e))?;
        let model = TraceModel::build(&inp.a, &inp.ymean);
        let stride = inp.params.w;
        Ok(StreamSolver {
            inp,
            model,
            stride,
            sol: Vec::new(),
            y8: Vec::new(),
            cc: Vec::new(),
            f: Vec::new(),
        })
    }
    pub fn units(&self) -> usize {
        self.model.k
    }
    pub fn set_stride(&mut self, stride: usize) {
        self.stride = stride;
    }

    /// Solve frame `t` of the session from its raw bytes.
    pub fn solve_frame(&mut self, t: usize, chunk: &[u8]) -> Result<(), JsError> {
        let (h, w) = (self.inp.params.h, self.inp.params.w);
        if t >= self.inp.params.t {
            return Err(JsError::new("frame past the inputs' T"));
        }
        let raw = unpad(chunk, h, w, self.stride);
        let y8 = self
            .inp
            .prep()
            .frame(&raw, h, w, self.inp.shifts[2 * t], self.inp.shifts[2 * t + 1]);
        let y: Vec<f32> = y8.iter().map(|&v| f32::from(v)).collect();
        self.model.solve(&y, &mut self.sol);
        let k = self.model.k;
        self.cc.extend(self.sol[..k].iter().map(|v| v.max(0.0)));
        self.f.push(self.sol[k] as f32);
        self.y8.extend_from_slice(&y8);
        Ok(())
    }
    /// The solved frames' preprocessed pixels (`n * H * W` uint8).
    pub fn take_y8(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.y8)
    }
    /// Their clipped traces, `(n, K)` float64 frame-major.
    pub fn take_cc(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.cc)
    }
    /// Their background traces (float32).
    pub fn take_f(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.f)
    }
}

#[wasm_bindgen]
impl Engine {
    /// Fold in frames a `StreamSolver` solved, in frame order: their preprocessed pixels `y8`
    /// (`n * H * W`), clipped traces `cc` (`n * K` float64) and background traces `f`.
    pub fn push_solved(&mut self, y8: &[u8], cc: &[f64], f: &[f32]) -> Result<(), JsError> {
        let p = self.inp.params.h * self.inp.params.w;
        let k = self.pass.model.k;
        let n = f.len();
        if y8.len() != n * p || cc.len() != n * k {
            return Err(JsError::new(&format!(
                "{n} solved frames come with {} pixels and {} traces, expected {} and {}",
                y8.len(),
                cc.len(),
                n * p,
                n * k
            )));
        }
        if self.pass.t + n > self.inp.params.t {
            return Err(JsError::new("more frames than the inputs' T"));
        }
        let mut y = vec![0f32; p];
        for i in 0..n {
            for (d, &s) in y.iter_mut().zip(&y8[i * p..(i + 1) * p]) {
                *d = f32::from(s);
            }
            self.pass.push_solved(&y, &cc[i * k..(i + 1) * k], f[i]);
        }
        Ok(())
    }
}

#[wasm_bindgen]
impl Pass2Engine {
    /// Compress tile `k` of the window (`(T, th, tw)` uint8 bytes) without taking it in; returns
    /// its factors as bytes for `set_tile` of the window's own engine.
    pub fn compress_at(&self, k: usize, raw: &[u8]) -> Result<Vec<u8>, JsError> {
        Ok(self.compress_tile_k(k, raw)?.to_bytes())
    }

    /// Pixels of tile `k` (its bytes per frame in the tile store).
    pub fn tile_pixels(&self, k: usize) -> usize {
        self.grid[k].pixels()
    }

    /// Take tile `k`'s factors (from `compress_at`); the tiles must come in order. Returns the rank.
    pub fn set_tile(&mut self, k: usize, factors: &[u8]) -> Result<usize, JsError> {
        if k != self.done.len() {
            return Err(JsError::new(&format!(
                "tile {k} given after {} tiles",
                self.done.len()
            )));
        }
        let tf = TileFactors::from_bytes(factors).map_err(|e| JsError::new(&e))?;
        let r = tf.rank;
        self.done.push(tf);
        Ok(r)
    }
}

/// `(n, 2)` float64 `.npy` of flat `shifts` (the session's shifts file).
#[wasm_bindgen]
pub fn npy_shifts(shifts: &[f64]) -> Vec<u8> {
    npy::write(&[shifts.len() / 2, 2], shifts)
}

/// `(n,)` float32 `.npy`.
#[wasm_bindgen]
pub fn npy_f32(data: &[f32]) -> Vec<u8> {
    npy::write(&[data.len()], data)
}

#[wasm_bindgen]
impl MotionEngine {
    /// Add the next template frame prepared by a `TemplateWorker` (round 1 or 2, in the order
    /// `push_template_raw` takes them).
    pub fn push_template_prepped(&mut self, prep: &[f32]) -> Result<(), JsError> {
        let b = self
            .builder
            .as_mut()
            .ok_or(JsError::new("template already built"))?;
        b.accumulate(prep);
        Ok(())
    }

    /// Round 1's mean image once the first round is in (what round 2 aligns to), else empty.
    pub fn first_image(&self) -> Vec<f32> {
        self.builder
            .as_ref()
            .and_then(|b| b.first_image())
            .map(<[f32]>::to_vec)
            .unwrap_or_default()
    }
}

/// A pool worker's share of the motion template: `round1` prepares a frame as the first round
/// averages it, `round2` (after `set_first`) as the second does, aligned to the first template.
#[wasm_bindgen]
pub struct TemplateWorker {
    p: MotionParams,
    corr: Correlator,
    first: Option<Motion>,
    stride: usize,
}

#[wasm_bindgen]
impl TemplateWorker {
    /// `params_json`: the stage's `params.json` (the motion parameters; extra keys are ignored).
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &[u8]) -> Result<TemplateWorker, JsError> {
        let p: MotionParams =
            serde_json::from_slice(params_json).map_err(|e| JsError::new(&format!("params.json: {e}")))?;
        crate::preprocess::check_median_ksize(p.median_ksize).map_err(|e| JsError::new(&e))?;
        let corr = Correlator::new(p.h, p.w, p.motion_down).map_err(|e| JsError::new(&e))?;
        Ok(TemplateWorker {
            stride: p.w,
            p,
            corr,
            first: None,
        })
    }
    pub fn set_stride(&mut self, stride: usize) {
        self.stride = stride;
    }
    /// The first template's source image (`MotionEngine::first_image`).
    pub fn set_first(&mut self, image: &[f32]) -> Result<(), JsError> {
        let corr = Correlator::new(self.p.h, self.p.w, self.p.motion_down).map_err(|e| JsError::new(&e))?;
        if image.len() != corr.prep_shape().0 * corr.prep_shape().1 {
            return Err(JsError::new("the first template has the wrong size"));
        }
        self.first = Some(Motion::from_image(corr, image.to_vec(), self.p.motion_max_shift));
        Ok(())
    }
    /// Round 1: the frame's registration image, prepared.
    pub fn round1(&self, chunk: &[u8]) -> Vec<f32> {
        let raw = unpad(chunk, self.p.h, self.p.w, self.stride);
        self.corr.prep(&registration_image(&raw, &self.p))
    }
    /// Round 2: the frame's registration image aligned to the first template, prepared.
    pub fn round2(&self, chunk: &[u8]) -> Result<Vec<f32>, JsError> {
        let first = self.first.as_ref().ok_or(JsError::new("set_first first"))?;
        let raw = unpad(chunk, self.p.h, self.p.w, self.stride);
        let reg = registration_image(&raw, &self.p);
        let [dy, dx] = first.estimate(&reg);
        Ok(self
            .corr
            .prep(&reg.shifted(self.p.h, self.p.w, dy as f32, dx as f32)))
    }
}

/// One lane of the streaming pass's fold on a pool worker (`stream::FoldLane`): lane `lane` of
/// `lanes`, built from the pass's inputs. `fold` the solved frames in frame order, then hand the
/// sums to the pass (`Engine::put_lane`).
#[wasm_bindgen]
pub struct StreamLane {
    lane: FoldLane,
    all: Vec<LaneRange>,
}

#[wasm_bindgen]
impl StreamLane {
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map, lane: usize, lanes: usize) -> Result<StreamLane, JsError> {
        let inp = Inputs::load(&mut |name| map_bytes(files, name)).map_err(|e| JsError::new(&e))?;
        let (h, w) = (inp.params.h, inp.params.w);
        let sup = Support::build(&inp.a, h, w, inp.params.dl_wnd);
        let all = lane_ranges(inp.a.nrows, h * w, &sup, lanes);
        let r = *all.get(lane).ok_or_else(|| JsError::new("no such lane"))?;
        Ok(StreamLane {
            lane: FoldLane::new(&inp.params, &sup, inp.a.nrows, r),
            all,
        })
    }
    /// Every lane's first pixel, and the end (`lanes + 1` values): how to cut a frame for them.
    pub fn pixel_bounds(&self) -> Vec<u32> {
        let mut b: Vec<u32> = self.all.iter().map(|r| r.pix[0] as u32).collect();
        b.push(self.all.last().map_or(0, |r| r.pix[1]) as u32);
        b
    }
    /// Fold `n` frames: `y8` their pixels of this lane (`n * npix`), `cc` all their traces (`n * K`).
    pub fn fold(&mut self, n: usize, y8: &[u8], cc: &[f64]) -> Result<(), JsError> {
        let np = self.lane.range.pix[1] - self.lane.range.pix[0];
        if n == 0 || y8.len() != n * np || cc.len() % n != 0 {
            return Err(JsError::new("a fold chunk's pixels and traces do not match"));
        }
        let k = cc.len() / n;
        let mut y = vec![0f32; np];
        for f in 0..n {
            for (d, &v) in y.iter_mut().zip(&y8[f * np..(f + 1) * np]) {
                *d = f32::from(v);
            }
            self.lane.fold(&y, &cc[f * k..(f + 1) * k]);
        }
        Ok(())
    }
    /// `[rows, pix, pairs]` as six numbers.
    pub fn range(&self) -> Vec<u32> {
        let r = self.lane.range;
        [r.rows, r.pix, r.pairs]
            .iter()
            .flatten()
            .map(|&v| v as u32)
            .collect()
    }
    pub fn done_flat(&self) -> Vec<f64> {
        self.lane.done.to_flat()
    }
    pub fn seg_flat(&self) -> Vec<f64> {
        self.lane.seg.to_flat()
    }
}

#[wasm_bindgen]
impl Engine {
    /// Store solved frames' traces without folding them (the pool's lanes fold them): `cc`
    /// `n * K` clipped traces, `f` the `n` background traces.
    pub fn push_traces(&mut self, cc: &[f64], f: &[f32]) -> Result<(), JsError> {
        let k = self.pass.model.k;
        if cc.len() != f.len() * k || self.pass.t + f.len() > self.inp.params.t {
            return Err(JsError::new("solved traces do not match the pass"));
        }
        for (i, &fi) in f.iter().enumerate() {
            self.pass.push_traces(&cc[i * k..(i + 1) * k], fi);
        }
        Ok(())
    }
    /// Put a fold lane's sums in place (`StreamLane::range`, `done_flat`, `seg_flat`).
    pub fn put_lane(&mut self, range: &[u32], done: &[f64], seg: &[f64]) -> Result<(), JsError> {
        let v: Vec<usize> = range.iter().map(|&x| x as usize).collect();
        if v.len() != 6 {
            return Err(JsError::new("a lane range has six numbers"));
        }
        let r = LaneRange {
            rows: [v[0], v[1]],
            pix: [v[2], v[3]],
            pairs: [v[4], v[5]],
        };
        self.pass.put_lane(r, done, seg).map_err(|e| JsError::new(&e))
    }
}
