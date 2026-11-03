//! wasm-bindgen API for the browser worker: an AVI index over a synchronous JS read
//! callback (`FileReaderSync` on `File.slice`) and the streaming pass fed one raw frame
//! at a time. Results come back as the same `.npy` files the native harness writes.

use crate::avi::{self, AviIndex, ReadAt};
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
        let raw: Vec<u8> = if self.stride == w {
            chunk[..h * w].to_vec()
        } else {
            (0..h)
                .flat_map(|r| chunk[r * self.stride..r * self.stride + w].iter().copied())
                .collect()
        };
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
        let m = Map::new();
        for (name, bytes) in files {
            m.set(&JsValue::from_str(&name), &Uint8Array::from(bytes.as_slice()));
        }
        Ok(m)
    }
}
