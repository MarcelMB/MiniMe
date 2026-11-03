//! wasm-bindgen API for the browser worker: an AVI index over a synchronous JS read
//! callback (`FileReaderSync` on `File.slice`), the streaming pass and the motion stage,
//! each fed one raw frame at a time. Results come back as the same `.npy` files the
//! native harness writes.

use crate::avi::{self, AviIndex, ReadAt};
use crate::motion::{self, Motion, MotionParams, Reg, TemplateBuilder};
use crate::npy;
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

fn map_bytes(files: &Map, name: &str) -> Result<Vec<u8>, String> {
    let v = files.get(&JsValue::from_str(name));
    if v.is_undefined() {
        return Err(format!("missing input file {name}"));
    }
    Ok(Uint8Array::new(&v).to_vec())
}

/// `h * w` pixels of a raw chunk whose rows are `stride` bytes apart.
fn unpad(chunk: &[u8], h: usize, w: usize, stride: usize) -> Vec<u8> {
    if stride == w {
        return chunk[..h * w].to_vec();
    }
    (0..h)
        .flat_map(|r| chunk[r * stride..r * stride + w].iter().copied())
        .collect()
}

fn npy_map(files: Vec<(String, Vec<u8>)>) -> Map {
    let m = Map::new();
    for (name, bytes) in files {
        m.set(&JsValue::from_str(&name), &Uint8Array::from(bytes.as_slice()));
    }
    m
}

/// One streaming pass: `push_raw` every frame in order, then `finish`.
#[wasm_bindgen]
pub struct Engine {
    inp: Inputs,
    pass: StreamPass,
    stride: usize,
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

    /// Footprint update and merge; returns output name -> `.npy`/`.json` bytes.
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
    p: MotionParams,
    t0: usize,
    t1: usize,
    check: Vec<usize>,
    stride: usize,
    tmpl_n: usize,
    builder: Option<TemplateBuilder>,
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
