use crate::cpu_chain_into;
use wasm_bindgen::prelude::*;

/// CPU chain for `n` frames, result copied out as a `Uint8Array`.
#[wasm_bindgen]
pub fn cpu_chain(frames: &[u8], shifts: &[f32], n: usize, h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * h * w];
    cpu_chain_into(frames, shifts, n, h, w, &mut out);
    out
}

#[wasm_bindgen]
pub fn cpu_median(frame: &[u8], h: usize, w: usize) -> Vec<u8> {
    crate::preprocess::median_blur(frame, h, w, 7)
}

#[wasm_bindgen]
pub fn cpu_shift(frame: &[u8], h: usize, w: usize, dy: f32, dx: f32) -> Vec<u8> {
    crate::preprocess::shift(frame, h, w, dy, dx)
}

#[wasm_bindgen]
pub fn cpu_tophat(frame: &[u8], h: usize, w: usize) -> Vec<u8> {
    crate::preprocess::tophat(frame, h, w, 15)
}

#[cfg(feature = "gpu")]
#[wasm_bindgen]
pub struct GpuPrep {
    gpu: crate::gpu::Gpu,
    out: Vec<u8>,
}

#[cfg(feature = "gpu")]
#[wasm_bindgen]
impl GpuPrep {
    /// Async: adapter, device, shader modules and pipelines.
    pub async fn create(h: usize, w: usize, max_b: usize) -> Result<GpuPrep, JsError> {
        let gpu = crate::gpu::Gpu::new(h, w, max_b)
            .await
            .map_err(|e| JsError::new(&e))?;
        Ok(GpuPrep {
            out: vec![0u8; max_b * h * w],
            gpu,
        })
    }

    pub fn info(&self) -> String {
        self.gpu.info.clone()
    }

    /// Run a batch; the result stays in wasm memory (`out_ptr`, `out_len`), as it would for the tile packing.
    pub async fn run(&mut self, frames: &[u8], shifts: &[f32], n: usize) -> Result<usize, JsError> {
        let len = n * self.gpu.h * self.gpu.w;
        self.gpu
            .run(frames, shifts, n, &mut self.out[..len])
            .await
            .map_err(|e| JsError::new(&e))?;
        Ok(len)
    }

    /// Queue a batch (upload and compute) without waiting for it.
    pub fn submit(&self, frames: &[u8], shifts: &[f32], n: usize) {
        self.gpu.submit(frames, shifts, n);
    }

    /// Wait for the queued batch and copy it into the output buffer.
    pub async fn finish(&mut self, n: usize) -> Result<usize, JsError> {
        let len = n * self.gpu.h * self.gpu.w;
        self.gpu
            .read(n, &mut self.out[..len])
            .await
            .map_err(|e| JsError::new(&e))?;
        Ok(len)
    }

    pub fn out_ptr(&self) -> usize {
        self.out.as_ptr() as usize
    }
}

// ---------------------------------------------------------------------------
// Pass 2 (tile products)
// ---------------------------------------------------------------------------

use crate::compress::{Gaussian, TileMatrix};
use crate::p2;

#[wasm_bindgen]
pub fn p2_mean(tile: &[u8], t: usize, p: usize) -> Vec<f32> {
    TileMatrix::new(tile, t, p).mean
}

#[wasm_bindgen]
pub fn p2_cpu_t_matmul(tile: &[u8], t: usize, p: usize, q: &[f32], ell: usize) -> Vec<f32> {
    TileMatrix::new(tile, t, p).t_matmul(q, ell)
}

#[wasm_bindgen]
pub fn p2_cpu_matmul(tile: &[u8], t: usize, p: usize, z: &[f32], ell: usize) -> Vec<f32> {
    TileMatrix::new(tile, t, p).matmul(z, ell)
}

/// The tile's mean/energy pass (what `TileMatrix::new` does): returns a checksum.
#[wasm_bindgen]
pub fn p2_cpu_new(tile: &[u8], t: usize, p: usize) -> f64 {
    TileMatrix::new(tile, t, p)
        .mean
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// The noise estimate of a tile (full-length rfft per pixel, `noise_frames = 0`): returns a checksum.
#[wasm_bindgen]
pub fn p2_cpu_noise(tile: &[u8], t: usize, p: usize) -> f64 {
    TileMatrix::new(tile, t, p).noise(0.06, 0).iter().sum()
}

/// `reps` times the four products of one rSVD (k = 40, one power iteration) on random input: returns a checksum.
#[wasm_bindgen]
pub fn p2_cpu_products(tile: &[u8], t: usize, p: usize, ell: usize, reps: usize) -> f64 {
    let y = TileMatrix::new(tile, t, p);
    let mut rng = Gaussian::new(1);
    let mut acc = 0.0;
    for _ in 0..reps {
        use crate::compress::Normals;
        let z = rng.draw(t * ell).unwrap();
        let a = y.matmul(&z, ell);
        let b = y.t_matmul(&a, ell);
        let c = y.matmul(&b, ell);
        let d = y.t_matmul(&c, ell);
        acc += f64::from(d[0]) + f64::from(d[d.len() - 1]);
    }
    acc
}

/// Rank, k and kept singular values of a whole CPU pass-2 solve (no noise estimate), as JSON.
#[wasm_bindgen]
pub async fn p2_cpu_compress(tile: &[u8], t: usize, p: usize) -> Result<String, JsError> {
    let y = TileMatrix::new(tile, t, p);
    let pr = p2::params(t, p);
    let k = p2::compress(&p2::Products::Cpu(&y), p, t, &pr, &mut Gaussian::new(5))
        .await
        .map_err(|e| JsError::new(&e))?;
    Ok(format!("{{\"rank\":{},\"k\":{},\"s0\":{}}}", k.rank, k.k, k.s[0]))
}

#[cfg(feature = "gpu")]
#[wasm_bindgen]
pub struct GpuPass2 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    prod: Option<crate::gpu::GpuProducts>,
}

#[cfg(feature = "gpu")]
#[wasm_bindgen]
impl GpuPass2 {
    pub async fn create() -> Result<GpuPass2, JsError> {
        let (device, queue, _) = crate::gpu::init_device().await.map_err(|e| JsError::new(&e))?;
        Ok(GpuPass2 {
            device,
            queue,
            prod: None,
        })
    }

    /// Upload a tile (and its mean) once; the products then run against it.
    pub fn set_tile(&mut self, tile: &[u8], t: usize, p: usize) {
        let mean = TileMatrix::new(tile, t, p).mean;
        self.prod = Some(crate::gpu::GpuProducts::new(
            &self.device,
            &self.queue,
            tile,
            t,
            p,
            &mean,
        ));
    }

    pub async fn t_matmul(&self, q: &[f32], ell: usize) -> Result<Vec<f32>, JsError> {
        self.prod
            .as_ref()
            .unwrap()
            .product(false, q, ell)
            .await
            .map_err(|e| JsError::new(&e))
    }

    pub async fn matmul(&self, z: &[f32], ell: usize) -> Result<Vec<f32>, JsError> {
        self.prod
            .as_ref()
            .unwrap()
            .product(true, z, ell)
            .await
            .map_err(|e| JsError::new(&e))
    }

    /// The whole rSVD + rank cut with GPU products; JSON of the result.
    pub async fn compress(&self, tile: &[u8], t: usize, p: usize) -> Result<String, JsError> {
        let g = self.prod.as_ref().unwrap();
        let pr = p2::params(t, p);
        let k = p2::compress(&p2::Products::Gpu(g), p, t, &pr, &mut Gaussian::new(5))
            .await
            .map_err(|e| JsError::new(&e))?;
        let _ = tile;
        Ok(format!("{{\"rank\":{},\"k\":{},\"s0\":{}}}", k.rank, k.k, k.s[0]))
    }

    /// CPU and GPU rSVD of the tile side by side: ranks, singular values, subspace overlaps.
    pub async fn compare(&self, tile: &[u8], t: usize, p: usize) -> Result<String, JsError> {
        p2::compare(tile, t, p, self.prod.as_ref().unwrap())
            .await
            .map_err(|e| JsError::new(&e))
    }
}
