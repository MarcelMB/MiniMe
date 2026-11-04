//! wasm-bindgen API for the browser worker: an AVI index over a synchronous JS read
//! callback (`FileReaderSync` on `File.slice`), the streaming pass, the motion stage and
//! pass 1, each fed one raw frame at a time, and pass 2, fed one tile of the tile store at a
//! time, and the seed and init stages and the CNMF loop on the factors those two leave; the
//! window loop's plumbing (`time_windows`, `window_inputs`) and the stitch of every window's
//! final units. Results come back as the same `.npy` files the native harness writes.

use crate::avi::{self, AviIndex, ReadAt};
use crate::cnmf::{self, CnmfParams, CnmfResult};
use crate::compress::{self, Gaussian, Normals, Pass2Params, Recorded, TileFactors};
use crate::factors::Factors;
use crate::init::{self, InitParams, InitResult};
use crate::motion::{self, Motion, MotionParams, Reg, TemplateBuilder};
use crate::npy;
use crate::pass1::{self, Pass1, Pass1Params};
use crate::preview;
use crate::seeds::{self, SeedParams, SeedsResult};
use crate::stitch::{self, StitchParams, Window};
use crate::stream::StreamPass;
use crate::{finish, Inputs};
use js_sys::{Function, Map, Uint8Array};
use wasm_bindgen::prelude::*;

struct JsReadAt<'a> {
    f: &'a Function,
    size: u64,
}

impl ReadAt for JsReadAt<'_> {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let v = self
            .f
            .call2(
                &JsValue::NULL,
                &JsValue::from_f64(offset as f64),
                &JsValue::from_f64(len as f64),
            )
            .map_err(|e| format!("read callback failed: {e:?}"))?;
        Ok(Uint8Array::new(&v).to_vec())
    }
    fn size(&self) -> u64 {
        self.size
    }
}

/// Frame index of one AVI.
#[wasm_bindgen]
pub struct AviFile {
    ix: AviIndex,
}

#[wasm_bindgen]
impl AviFile {
    /// `read_at(offset, len) -> Uint8Array` must read synchronously.
    pub fn parse(read_at: &Function, size: f64) -> Result<AviFile, JsError> {
        let mut r = JsReadAt {
            f: read_at,
            size: size as u64,
        };
        Ok(AviFile {
            ix: avi::index(&mut r).map_err(|e| JsError::new(&e))?,
        })
    }
    pub fn width(&self) -> usize {
        self.ix.width
    }
    pub fn height(&self) -> usize {
        self.ix.height
    }
    pub fn n_frames(&self) -> usize {
        self.ix.frames.len()
    }
    /// Byte offset of frame `i`.
    pub fn offset(&self, i: usize) -> f64 {
        self.ix.frames[i].0 as f64
    }
    /// Chunk size of frame `i`.
    pub fn size(&self, i: usize) -> u32 {
        self.ix.frames[i].1
    }
}

pub(crate) fn map_bytes(files: &Map, name: &str) -> Result<Vec<u8>, String> {
    let v = files.get(&JsValue::from_str(name));
    if v.is_undefined() {
        return Err(format!("missing input file {name}"));
    }
    Ok(Uint8Array::new(&v).to_vec())
}

/// `h * w` pixels of a raw chunk whose rows are `stride` bytes apart.
pub(crate) fn unpad(chunk: &[u8], h: usize, w: usize, stride: usize) -> Vec<u8> {
    if stride == w {
        return chunk[..h * w].to_vec();
    }
    (0..h)
        .flat_map(|r| chunk[r * stride..r * stride + w].iter().copied())
        .collect()
}

pub(crate) fn npy_map(files: Vec<(String, Vec<u8>)>) -> Map {
    let m = Map::new();
    for (name, bytes) in files {
        m.set(&JsValue::from_str(&name), &Uint8Array::from(bytes.as_slice()));
    }
    m
}

/// One streaming pass: `push_raw` every frame in order, then `finish`.
#[wasm_bindgen]
pub struct Engine {
    pub(crate) inp: Inputs,
    pub(crate) pass: StreamPass,
    pub(crate) stride: usize,
}

#[wasm_bindgen]
impl Engine {
    /// `files` maps input names (`params.json`, `A_data.npy`, ...) to `Uint8Array`s.
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<Engine, JsError> {
        let inp = Inputs::load(&mut |name| map_bytes(files, name)).map_err(|e| JsError::new(&e))?;
        let pass = StreamPass::new(inp.params.clone(), &inp.a, &inp.ymean);
        let stride = inp.params.w;
        Ok(Engine { inp, pass, stride })
    }
    pub fn frames(&self) -> usize {
        self.inp.params.t
    }
    pub fn height(&self) -> usize {
        self.inp.params.h
    }
    pub fn width(&self) -> usize {
        self.inp.params.w
    }
    pub fn units(&self) -> usize {
        self.pass.model.k
    }
    pub fn done(&self) -> usize {
        self.pass.t
    }
    pub fn state_mb(&self) -> f64 {
        self.pass.state_bytes() as f64 / 1048576.0
    }
    /// Row stride of the raw chunks (`width`, or padded to 4).
    pub fn set_stride(&mut self, stride: usize) {
        self.stride = stride;
    }

    /// Preprocess one raw frame chunk (uint8) and fold it in.
    pub fn push_raw(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        let (h, w) = (self.inp.params.h, self.inp.params.w);
        let t = self.pass.t;
        if t >= self.inp.params.t {
            return Err(JsError::new("more frames than the inputs' T"));
        }
        let raw = unpad(chunk, h, w, self.stride);
        let y8 = self
            .inp
            .prep()
            .frame(&raw, h, w, self.inp.shifts[2 * t], self.inp.shifts[2 * t + 1]);
        let y: Vec<f32> = y8.iter().map(|&v| f32::from(v)).collect();
        self.pass.push(&y);
        Ok(())
    }

    /// The `.npy` header of `C.npy`, (K, T) float32; the data follows unit by unit.
    pub fn traces_header(&self) -> Vec<u8> {
        npy::write_header::<f32>(&[self.pass.model.k, self.inp.params.t])
    }
    /// The traces pushed since the last call, (K, n) unit-major: row `i` goes to
    /// `C.npy`'s unit `i` from the frame after the previous call's last one.
    pub fn take_traces(&mut self) -> Vec<f32> {
        crate::unit_major(&self.pass.take_c(), self.pass.model.k)
    }

    /// Footprint update and merge; returns output name -> `.npy`/`.json` bytes (all but
    /// `C.npy`, which the worker writes from `take_traces`).
    pub fn finish(&self, timing_json: &str) -> Result<Map, JsError> {
        let timing: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        let files = finish(&self.pass, &self.inp, timing).map_err(|e| JsError::new(&e))?;
        Ok(npy_map(files))
    }
}

/// The motion stage on frames `[t0, t1)` (`minime-core motion`): push the first
/// `template_n()` frames of the session twice with `push_template_raw`, then frames
/// `t0..t1` with `push_raw`, then `finish`.
#[wasm_bindgen]
pub struct MotionEngine {
    pub(crate) p: MotionParams,
    t0: usize,
    t1: usize,
    check: Vec<usize>,
    stride: usize,
    tmpl_n: usize,
    pub(crate) builder: Option<TemplateBuilder>,
    motion: Option<Motion>,
    shifts: Vec<f64>,
    quality: Vec<f32>,
    chk_t: Vec<i64>,
    reg_chk: Vec<f32>,
    prep_chk: Vec<f32>,
}

#[wasm_bindgen]
impl MotionEngine {
    /// `files`: `params.json` (`ref_motion.py`'s inputs) and optionally `check_frames.npy`;
    /// `seg0_frames`: frames in the first segment (the template is built from those).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map, seg0_frames: usize) -> Result<MotionEngine, JsError> {
        let err = |e: String| JsError::new(&e);
        let meta: serde_json::Value = serde_json::from_slice(&map_bytes(files, "params.json").map_err(err)?)
            .map_err(|e| JsError::new(&format!("params.json: {e}")))?;
        let p: MotionParams =
            serde_json::from_value(meta.clone()).map_err(|e| JsError::new(&format!("params.json: {e}")))?;
        let get = |k: &str| meta[k].as_u64().map(|v| v as usize);
        let (t0, t1) = (
            get("t0").unwrap_or(0),
            get("t1").ok_or(JsError::new("params.json: t1"))?,
        );
        let check = match map_bytes(files, "check_frames.npy") {
            Ok(b) => npy::read_i64(&b)
                .map_err(err)?
                .data
                .iter()
                .map(|&v| v as usize)
                .collect(),
            Err(_) => Vec::new(),
        };
        let n = p.template_frames.min(seg0_frames);
        let builder = TemplateBuilder::new(&p, n).map_err(err)?;
        Ok(MotionEngine {
            stride: p.w,
            p,
            t0,
            t1,
            check,
            tmpl_n: n,
            builder: Some(builder),
            motion: None,
            shifts: Vec::new(),
            quality: Vec::new(),
            chk_t: Vec::new(),
            reg_chk: Vec::new(),
            prep_chk: Vec::new(),
        })
    }
    pub fn height(&self) -> usize {
        self.p.h
    }
    pub fn width(&self) -> usize {
        self.p.w
    }
    pub fn t0(&self) -> usize {
        self.t0
    }
    pub fn t1(&self) -> usize {
        self.t1
    }
    /// Template frames (the first of segment 0, pushed twice).
    pub fn template_n(&self) -> usize {
        self.tmpl_n
    }
    /// Frames of `t0..t1` done.
    pub fn done(&self) -> usize {
        self.quality.len()
    }
    pub fn set_stride(&mut self, stride: usize) {
        self.stride = stride;
    }

    fn reg(&self, chunk: &[u8]) -> Reg {
        motion::registration_image(&unpad(chunk, self.p.h, self.p.w, self.stride), &self.p)
    }

    /// One template frame (frames `0..n` of segment 0, twice in order).
    pub fn push_template_raw(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        let reg = self.reg(chunk);
        let b = self
            .builder
            .as_mut()
            .ok_or(JsError::new("template already built"))?;
        b.push(&reg);
        Ok(())
    }

    /// Finish the template once all its frames are in (returns its frame count).
    pub fn build_template(&mut self) -> Result<usize, JsError> {
        let b = self
            .builder
            .take()
            .ok_or(JsError::new("template already built"))?;
        let n = b.pushed() / 2;
        self.motion = Some(b.finish().map_err(|e| JsError::new(&e))?);
        Ok(n)
    }

    /// Shift and quality of the next frame of `t0..t1`.
    pub fn push_raw(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        let reg = self.reg(chunk);
        let m = self.motion.as_ref().ok_or(JsError::new("build_template first"))?;
        let t = self.t0 + self.quality.len();
        if t >= self.t1 {
            return Err(JsError::new("more frames than t1 - t0"));
        }
        let [dy, dx] = m.estimate(&reg);
        let (h, w) = (self.p.h, self.p.w);
        self.quality
            .push(m.quality(&reg.shifted(h, w, dy as f32, dx as f32)));
        self.shifts.extend([dy, dx]);
        if self.check.contains(&t) {
            self.chk_t.push(t as i64);
            let corr = motion::Correlator::new(h, w, self.p.motion_down).map_err(|e| JsError::new(&e))?;
            self.prep_chk.extend(corr.prep(&reg));
            if let Reg::F32(f) = &reg {
                self.reg_chk.extend_from_slice(f);
            }
        }
        Ok(())
    }

    /// Output name -> `.npy`/`.json` bytes (the names `minime-core motion` writes).
    pub fn finish(&self, timing_json: &str) -> Result<Map, JsError> {
        let m = self.motion.as_ref().ok_or(JsError::new("build_template first"))?;
        let n = self.quality.len();
        let (ph, pw) = m.prep_shape();
        let k = self.chk_t.len();
        let mut files = vec![
            (
                "template.npy".to_string(),
                npy::write(&[ph, pw], m.template_image()),
            ),
            // what `Pass1Engine` needs to estimate the same shifts (`template_image` is round-tripped once more)
            ("template_src.npy".into(), npy::write(&[ph, pw], m.source_image())),
            ("shifts.npy".into(), npy::write(&[n, 2], &self.shifts)),
            ("quality.npy".into(), npy::write(&[n], &self.quality)),
            ("frames_t.npy".into(), npy::write(&[k], &self.chk_t)),
            ("prep.npy".into(), npy::write(&[k, ph, pw], &self.prep_chk)),
        ];
        if !self.reg_chk.is_empty() {
            files.push((
                "reg.npy".into(),
                npy::write(&[k, self.p.h, self.p.w], &self.reg_chk),
            ));
        }
        let max_shift = self.shifts.iter().fold(0f64, |a, &v| a.max(v.abs()));
        let mut timing: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        timing["stage"] = "motion".into();
        timing["frames"] = n.into();
        timing["max_abs_shift"] = max_shift.into();
        files.push(("summary.json".into(), serde_json::to_vec_pretty(&timing).unwrap()));
        Ok(npy_map(files))
    }
}

/// Pass 1 on frames `[t0, t1)` (`minime-core pass1`): `push_raw` each frame, append
/// `take_tiles()` to the tile files whenever `buffered()` is large enough, then `finish`
/// with the CRC-32 of every tile file as stored. Without `shifts.npy` in the inputs the
/// shifts are estimated against `template.npy`.
#[wasm_bindgen]
pub struct Pass1Engine {
    pub(crate) p: Pass1Params,
    pub(crate) pass: Pass1,
    pub(crate) shifts: Option<Vec<f32>>,
    pub(crate) stride: usize,
}

#[wasm_bindgen]
impl Pass1Engine {
    /// `files`: `ref_pass1.py`'s inputs (`params.json`, `template.npy`, `shifts.npy`).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<Pass1Engine, JsError> {
        let err = |e: String| JsError::new(&e);
        let p: Pass1Params = serde_json::from_slice(&map_bytes(files, "params.json").map_err(err)?)
            .map_err(|e| JsError::new(&format!("params.json: {e}")))?;
        let template = npy::read_f32(&map_bytes(files, "template.npy").map_err(err)?)
            .map_err(err)?
            .data;
        let shifts = match map_bytes(files, "shifts.npy") {
            Ok(b) => Some(npy::read_f32(&b).map_err(err)?.data),
            Err(_) => None,
        };
        if let Some(s) = &shifts {
            if s.len() != 2 * (p.t1 - p.t0) {
                return Err(JsError::new(&format!(
                    "{} shifts for {} frames",
                    s.len() / 2,
                    p.t1 - p.t0
                )));
            }
        }
        let m = Motion::from_template_image(&p.motion, template).map_err(err)?;
        let pass = Pass1::new(&p, Some(m));
        Ok(Pass1Engine {
            stride: p.motion.w,
            p,
            pass,
            shifts,
        })
    }
    pub fn height(&self) -> usize {
        self.p.motion.h
    }
    pub fn width(&self) -> usize {
        self.p.motion.w
    }
    pub fn t0(&self) -> usize {
        self.p.t0
    }
    pub fn t1(&self) -> usize {
        self.p.t1
    }
    pub fn done(&self) -> usize {
        self.pass.t
    }
    pub fn n_tiles(&self) -> usize {
        self.pass.grid.len()
    }
    pub fn tile_pixels(&self, k: usize) -> usize {
        self.pass.grid[k].pixels()
    }
    pub fn set_stride(&mut self, stride: usize) {
        self.stride = stride;
    }
    pub fn buffered(&self) -> usize {
        self.pass.buffered()
    }

    /// Correct the next frame of `t0..t1` and fold it in.
    pub fn push_raw(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        let (h, w) = (self.p.motion.h, self.p.motion.w);
        let i = self.pass.t;
        let given = self.shifts.as_ref().map(|s| [s[2 * i], s[2 * i + 1]]);
        let raw = unpad(chunk, h, w, self.stride);
        self.pass.push(&raw, given).map_err(|e| JsError::new(&e))?;
        Ok(())
    }

    /// The buffered frames, tile after tile (`buffered() * tile_pixels(k)` bytes each).
    pub fn take_tiles(&mut self) -> Vec<u8> {
        self.pass.take_tiles().concat()
    }

    /// Output name -> `.npy`/`.json` bytes (the names `minime-core pass1` writes).
    pub fn finish(&self, timing_json: &str, tile_crc: &[u32]) -> Result<Map, JsError> {
        let mut summary: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        summary["stage"] = "pass1".into();
        summary["frames"] = self.pass.t.into();
        summary["blocks"] = self.pass.n_blocks().into();
        summary["tiles"] = self.pass.grid.len().into();
        summary["estimate"] = self.shifts.is_none().into();
        Ok(npy_map(pass1::outputs(&self.pass, tile_crc, summary)))
    }
}

impl CnmfEngine {
    /// The loop's final footprints and the peak of each unit's `YrA`: what the stitch takes.
    pub(crate) fn window(&self) -> Result<(crate::init::Csr, Vec<f64>), JsError> {
        let r = self
            .res
            .as_ref()
            .ok_or_else(|| JsError::new("run the loop first"))?;
        let last = r.last();
        let (a, t) = (last.merge.a.clone(), r.t);
        let bright = (0..a.k)
            .map(|i| {
                f64::from(
                    last.merge.c[i * t..(i + 1) * t]
                        .iter()
                        .fold(f32::NEG_INFINITY, |m, &v| m.max(v)),
                )
            })
            .collect();
        Ok((a, bright))
    }
}

/// CRC-32 of `bytes` continuing from `crc` (`zlib.crc32`).
#[wasm_bindgen]
pub fn crc32(bytes: &[u8], crc: u32) -> u32 {
    pass1::crc32(bytes, crc)
}

/// Pass 2 on the tile store pass 1 left in OPFS: `compress` every tile in grid order, then
/// `finish`. With `omega_NNN.npy` in the inputs it replays numpy's draws (`ref_pass2.py`),
/// otherwise it draws its own (seed = tile index).
#[wasm_bindgen]
pub struct Pass2Engine {
    pub(crate) p: Pass2Params,
    pub(crate) grid: Vec<pass1::Tile>,
    pub(crate) files: Map,
    pub(crate) done: Vec<TileFactors>,
}

#[wasm_bindgen]
impl Pass2Engine {
    /// `files`: `ref_pass2.py`'s `params.json` as `pass2.json`, plus its `omega_NNN.npy`.
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<Pass2Engine, JsError> {
        let p: Pass2Params =
            serde_json::from_slice(&map_bytes(files, "pass2.json").map_err(|e| JsError::new(&e))?)
                .map_err(|e| JsError::new(&format!("pass2.json: {e}")))?;
        Ok(Pass2Engine {
            grid: pass1::tile_grid(p.h, p.w, p.tile),
            p,
            files: files.clone(),
            done: Vec::new(),
        })
    }
    pub fn n_tiles(&self) -> usize {
        self.grid.len()
    }
    pub fn replayed(&self) -> bool {
        self.files.has(&JsValue::from_str("omega_000.npy"))
    }

    /// Compress the next tile (`(T, th, tw)` uint8 bytes); returns its kept rank.
    pub fn compress(&mut self, raw: &[u8]) -> Result<usize, JsError> {
        let tf = self.compress_tile_k(self.done.len(), raw)?;
        let r = tf.rank;
        self.done.push(tf);
        Ok(r)
    }

    /// Output name -> `.npy` bytes (the names `minime-core pass2` writes; the summary is
    /// `summary_pass2.json` so it sits next to pass 1's).
    pub fn finish(&self, timing_json: &str) -> Map {
        let mut summary: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        summary["stage"] = "pass2".into();
        summary["frames"] = self.p.t.into();
        summary["tiles"] = self.done.len().into();
        summary["own_rng"] = (!self.replayed()).into();
        summary["rank_total"] = self.done.iter().map(|t| t.rank).sum::<usize>().into();
        let files = compress::outputs(&self.done, summary)
            .into_iter()
            .map(|(n, b)| {
                (
                    if n == "summary.json" {
                        "summary_pass2.json".into()
                    } else {
                        n
                    },
                    b,
                )
            })
            .collect();
        npy_map(files)
    }
}

impl Pass2Engine {
    /// `compress` of tile `k` (its draws are seeded or replayed by `k`): the tile is independent of
    /// the others, so a pool worker can do any of them.
    pub(crate) fn compress_tile_k(&self, k: usize, raw: &[u8]) -> Result<TileFactors, JsError> {
        let p = self
            .grid
            .get(k)
            .ok_or_else(|| JsError::new("all tiles done"))?
            .pixels();
        if raw.len() != self.p.t * p {
            return Err(JsError::new(&format!(
                "tile {k}: {} bytes, expected {}",
                raw.len(),
                self.p.t * p
            )));
        }
        let mut rng: Box<dyn Normals> = if self.replayed() {
            let b = map_bytes(&self.files, &format!("omega_{k:03}.npy")).map_err(|e| JsError::new(&e))?;
            Box::new(Recorded::new(
                npy::read_f32(&b).map_err(|e| JsError::new(&e))?.data,
            ))
        } else {
            Box::new(Gaussian::new(k as u64))
        };
        compress::compress_tile(raw, self.p.t, p, &self.p, rng.as_mut()).map_err(|e| JsError::new(&e))
    }
}

/// The seed stage (`minime-core seeds`) on the window's own factors: pass 1's block maxima
/// and max projection, pass 2's tiles. `run`, then `finish`.
#[wasm_bindgen]
pub struct SeedsEngine {
    p: SeedParams,
    f: Factors,
    res: Option<SeedsResult>,
}

#[wasm_bindgen]
impl SeedsEngine {
    /// `files`: `ref_seeds.py`'s `seeds.json`; `pass1`, `pass2`: the engines that ran on this
    /// window (pass 2 finished).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map, pass1: &Pass1Engine, pass2: &Pass2Engine) -> Result<SeedsEngine, JsError> {
        let err = |e: String| JsError::new(&e);
        let p: SeedParams = serde_json::from_slice(&map_bytes(files, "seeds.json").map_err(err)?)
            .map_err(|e| JsError::new(&format!("seeds.json: {e}")))?;
        let p1 = &pass1.pass;
        if p1.t != p.t || pass2.p.t != p.t || pass2.done.len() != p1.grid.len() {
            return Err(JsError::new(&format!(
                "seeds.json says {} frames; pass 1 ran {}, pass 2 {} on {} of {} tiles",
                p.t,
                p1.t,
                pass2.p.t,
                pass2.done.len(),
                p1.grid.len()
            )));
        }
        let f = Factors::new(
            p.h,
            p.w,
            p.t,
            p.tile,
            pass2.done.clone(),
            p1.max_proj.clone(),
            p1.block_max.clone(),
            p1.edges.clone(),
        )
        .map_err(err)?;
        Ok(SeedsEngine { p, f, res: None })
    }

    /// Run the stage; returns the number of merged seeds.
    pub fn run(&mut self) -> Result<usize, JsError> {
        let now = || js_sys::Date::now() / 1e3;
        let r = seeds::run(&self.f, &self.p, &now).map_err(|e| JsError::new(&e))?;
        let n = r.merged.len();
        self.res = Some(r);
        Ok(n)
    }

    /// Output name -> `.npy` bytes (the names `minime-core seeds` writes; the summary is
    /// `summary_seeds.json`).
    pub fn finish(&self, timing_json: &str) -> Result<Map, JsError> {
        let r = self
            .res
            .as_ref()
            .ok_or_else(|| JsError::new("run the stage first"))?;
        let mut summary: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        summary["stage"] = "seeds".into();
        summary["frames"] = self.p.t.into();
        let files = seeds::outputs(r, summary)
            .into_iter()
            .map(|(n, b)| {
                (
                    if n == "summary.json" {
                        "summary_seeds.json".into()
                    } else {
                        n
                    },
                    b,
                )
            })
            .collect();
        Ok(npy_map(files))
    }
}

/// Init (`minime-core init`) on the seed stage's merged seeds, factors and `FactorCorr`.
/// `run` with the finished `SeedsEngine`, then `finish`.
#[wasm_bindgen]
pub struct InitEngine {
    p: InitParams,
    res: Option<InitResult>,
}

#[wasm_bindgen]
impl InitEngine {
    /// `files`: `ref_init.py`'s `init.json`.
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<InitEngine, JsError> {
        let p: InitParams =
            serde_json::from_slice(&map_bytes(files, "init.json").map_err(|e| JsError::new(&e))?)
                .map_err(|e| JsError::new(&format!("init.json: {e}")))?;
        Ok(InitEngine { p, res: None })
    }

    /// Run the stage on `seeds` (run, not yet freed); returns the units after the merge.
    pub fn run(&mut self, seeds: &mut SeedsEngine) -> Result<usize, JsError> {
        let r = seeds
            .res
            .as_mut()
            .ok_or_else(|| JsError::new("run the seed stage first"))?;
        let now = || js_sys::Date::now() / 1e3;
        let res = init::run(&mut r.fc, &seeds.f, &r.merged, &self.p, &now).map_err(|e| JsError::new(&e))?;
        let n = res.a.k;
        self.res = Some(res);
        Ok(n)
    }

    /// Output name -> `.npy` bytes (the names `minime-core init` writes; the summary is
    /// `summary_init.json`).
    pub fn finish(&self, timing_json: &str) -> Result<Map, JsError> {
        let r = self
            .res
            .as_ref()
            .ok_or_else(|| JsError::new("run the stage first"))?;
        let mut summary: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        summary["stage"] = "init".into();
        summary["frames"] = self.p.t.into();
        let files = init::outputs(r, &self.p, summary)
            .into_iter()
            .map(|(n, b)| {
                (
                    if n == "summary.json" {
                        "summary_init.json".into()
                    } else {
                        n
                    },
                    b,
                )
            })
            .collect();
        Ok(npy_map(files))
    }
}

/// The CNMF loop (`minime-core cnmf`) on init's merged units and the seed stage's factors.
/// `run` with the finished `SeedsEngine` and `InitEngine`, then `finish`.
#[wasm_bindgen]
pub struct CnmfEngine {
    p: CnmfParams,
    pub(crate) res: Option<CnmfResult>,
}

#[wasm_bindgen]
impl CnmfEngine {
    /// `files`: `ref_cnmf.py`'s `cnmf.json`.
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<CnmfEngine, JsError> {
        let p: CnmfParams =
            serde_json::from_slice(&map_bytes(files, "cnmf.json").map_err(|e| JsError::new(&e))?)
                .map_err(|e| JsError::new(&format!("cnmf.json: {e}")))?;
        Ok(CnmfEngine { p, res: None })
    }

    /// Run the loop on `init`'s units (both engines run, not yet freed); returns the final
    /// number of units.
    pub fn run(&mut self, seeds: &SeedsEngine, init: &InitEngine) -> Result<usize, JsError> {
        let r = init.res.as_ref().ok_or_else(|| JsError::new("run init first"))?;
        let now = || js_sys::Date::now() / 1e3;
        let res = cnmf::run(&seeds.f, &r.a, &r.c, &self.p, &now).map_err(|e| JsError::new(&e))?;
        let n = res.last().merge.a.k;
        self.res = Some(res);
        Ok(n)
    }

    /// Output name -> `.npy` bytes (the names `minime-core cnmf` writes; the summary is
    /// `summary_cnmf.json`).
    pub fn finish(&self, timing_json: &str) -> Result<Map, JsError> {
        let r = self
            .res
            .as_ref()
            .ok_or_else(|| JsError::new("run the loop first"))?;
        let mut summary: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        summary["stage"] = "cnmf".into();
        let files = cnmf::outputs(r, summary)
            .into_iter()
            .map(|(n, b)| {
                (
                    if n == "summary.json" {
                        "summary_cnmf.json".into()
                    } else {
                        n
                    },
                    b,
                )
            })
            .collect();
        Ok(npy_map(files))
    }
}

/// The launcher's frame preview (`preview.frame_panels`): one raw frame (`h * w` uint8) as
/// `(4, h, w)` uint8, raw, denoised, registration image and tophat, with the motion parameters
/// of `params_json` (a `session.json` `pass1` object).
#[wasm_bindgen]
pub fn preview_panels(frame: &[u8], h: usize, w: usize, params_json: &str) -> Result<Vec<u8>, JsError> {
    let p = preview::motion_params(params_json, h, w).map_err(|e| JsError::new(&e))?;
    preview::frame_panels(frame, &p).map_err(|e| JsError::new(&e))
}

/// The launcher's motion check (`preview.motion_check`) of `n` consecutive raw frames from frame
/// `t0` (`clip`, `n * h * w` uint8), as the JSON of Python's answer.
#[wasm_bindgen]
pub fn motion_check(
    clip: &[u8],
    n: usize,
    h: usize,
    w: usize,
    t0: usize,
    params_json: &str,
) -> Result<String, JsError> {
    let err = |e: String| JsError::new(&e);
    let p = preview::motion_params(params_json, h, w).map_err(err)?;
    let r = preview::motion_check(clip, n, t0, &p).map_err(err)?;
    serde_json::to_string(&r).map_err(|e| JsError::new(&e.to_string()))
}

/// The viewer's motion residual (`derived.block_residual`) of `n` block maxima (`block_max`,
/// `n * h * w` float32) against the session's max projection (`h * w` float32): `(n, 3)` flat,
/// dy, dx and correlation per block. `params_json` is the pass-1 parameter object (its
/// `motion_max_shift` is the search window).
#[wasm_bindgen]
pub fn block_residual(
    block_max: &[f32],
    n: usize,
    h: usize,
    w: usize,
    max_proj: &[f32],
    params_json: &str,
) -> Result<Vec<f32>, JsError> {
    let err = |e: String| JsError::new(&e);
    let p = preview::motion_params(params_json, h, w).map_err(err)?;
    motion::block_residual(block_max, n, max_proj, &p).map_err(err)
}

/// `stream.time_windows` on the segment starts `edges` (with the end `t`): `[t0, t1)` pairs,
/// flat.
#[wasm_bindgen]
pub fn time_windows(t: usize, length: usize, edges: &[u32]) -> Vec<u32> {
    let e: Vec<usize> = edges.iter().map(|&v| v as usize).collect();
    stitch::time_windows(t, length, &e)
        .into_iter()
        .flatten()
        .map(|v| v as u32)
        .collect()
}

/// The streaming pass's `params.json` for `t` frames from `session.json`'s `stream` object.
#[wasm_bindgen]
pub fn stream_params(session: &[u8], t: usize) -> Result<Vec<u8>, JsError> {
    stitch::stream_params(session, t).map_err(|e| JsError::new(&e))
}

/// One window's stage files (`params.json`, `pass2.json`, `seeds.json`, `init.json`,
/// `cnmf.json`, `shifts.npy`) from the session's `session.json` and `shifts.npy`.
#[wasm_bindgen]
pub fn window_inputs(session: &[u8], shifts: &[u8], t0: usize, t1: usize) -> Result<Map, JsError> {
    Ok(npy_map(
        stitch::window_inputs(session, shifts, t0, t1).map_err(|e| JsError::new(&e))?,
    ))
}

/// The stitch (`minime-core stitch`): add every window's final units (`add_cnmf` after its
/// CNMF loop, or `add_inputs` from `ref_stitch.py`'s inputs), then `run` and `finish`.
#[wasm_bindgen]
pub struct StitchEngine {
    p: StitchParams,
    pub(crate) windows: Vec<Window>,
    mean_noise: stitch::MeanNoise,
    res: Option<stitch::StitchResult>,
}

#[wasm_bindgen]
impl StitchEngine {
    /// `files`: `stitch.json`, or `session.json` (its `stitch` object, `H` and `W` from
    /// its `pass1`).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &Map) -> Result<StitchEngine, JsError> {
        let p: StitchParams = match map_bytes(files, "stitch.json") {
            Ok(b) => serde_json::from_slice(&b).map_err(|e| JsError::new(&format!("stitch.json: {e}")))?,
            Err(_) => {
                let s: serde_json::Value =
                    serde_json::from_slice(&map_bytes(files, "session.json").map_err(|e| JsError::new(&e))?)
                        .map_err(|e| JsError::new(&format!("session.json: {e}")))?;
                let mut st = s["stitch"].clone();
                st["H"] = s["pass1"]["H"].clone();
                st["W"] = s["pass1"]["W"].clone();
                serde_json::from_value(st).map_err(|e| JsError::new(&format!("session.json: stitch: {e}")))?
            }
        };
        Ok(StitchEngine {
            p,
            windows: Vec::new(),
            mean_noise: stitch::MeanNoise::default(),
            res: None,
        })
    }

    /// Windows listed in `stitch.json`.
    pub fn n_inputs(&self) -> usize {
        self.p.windows.len()
    }

    /// Window `i` of `ref_stitch.py`'s inputs (`win_{i:02}_*`).
    pub fn add_inputs(&mut self, files: &Map, i: usize) -> Result<usize, JsError> {
        let t = *self
            .p
            .windows
            .get(i)
            .ok_or_else(|| JsError::new(&format!("stitch.json lists no window {i}")))?;
        let w = stitch::read_window(&mut |n| map_bytes(files, n), i, t, self.p.h * self.p.w)
            .map_err(|e| JsError::new(&e))?;
        let k = w.a.k;
        self.windows.push(w);
        Ok(k)
    }

    /// The final units of the window `[t0, t1)` whose CNMF loop `cnmf` ran: footprints and the
    /// peak of each unit's `YrA`. Returns the window's unit count.
    pub fn add_cnmf(&mut self, cnmf: &CnmfEngine, t0: usize, t1: usize) -> Result<usize, JsError> {
        let (a, bright) = cnmf.window()?;
        let k = a.k;
        self.windows.push(Window {
            t: [t0, t1],
            a,
            bright,
        });
        Ok(k)
    }

    /// The pass-2 factors of the window `[t0, t1)` (`pass2` finished), for the session's mean
    /// image and noise (`ymean.npy`, `sn.npy` in `finish`).
    pub fn add_pass2(&mut self, pass2: &Pass2Engine, t0: usize, t1: usize) -> Result<(), JsError> {
        if pass2.done.len() != pass2.grid.len() {
            return Err(JsError::new("finish pass 2 first"));
        }
        self.mean_noise.add(&pass2.grid, &pass2.done, self.p.w, t1 - t0);
        Ok(())
    }

    /// Stitch the windows added so far; returns the number of stitched units.
    pub fn run(&mut self) -> Result<usize, JsError> {
        let r = stitch::stitch(&self.windows, self.p.cos_thres, self.p.h * self.p.w)
            .map_err(|e| JsError::new(&e))?;
        let n = r.a.k;
        self.res = Some(r);
        Ok(n)
    }

    /// Output name -> `.npy` bytes (the names `minime-core stitch` writes; the summary is
    /// `summary_stitch.json`), plus the windows in `ref_stitch.py`'s inputs layout
    /// (`stitch.json`, `win_*`).
    pub fn finish(&self, timing_json: &str) -> Result<Map, JsError> {
        let r = self
            .res
            .as_ref()
            .ok_or_else(|| JsError::new("run the stitch first"))?;
        let mut summary: serde_json::Value = serde_json::from_str(timing_json).unwrap_or_default();
        summary["stage"] = "stitch".into();
        summary["windows"] = self.windows.len().into();
        let mut files: Vec<(String, Vec<u8>)> = stitch::outputs(r, summary)
            .into_iter()
            .map(|(n, b)| {
                (
                    if n == "summary.json" {
                        "summary_stitch.json".into()
                    } else {
                        n
                    },
                    b,
                )
            })
            .collect();
        files.extend(stitch::window_files(&self.windows, &self.p));
        if let Some((ymean, sn)) = self.mean_noise.finish() {
            let p = ymean.len();
            files.push(("ymean.npy".into(), npy::write(&[p], &ymean)));
            files.push(("sn.npy".into(), npy::write(&[p], &sn)));
        }
        Ok(npy_map(files))
    }
}
