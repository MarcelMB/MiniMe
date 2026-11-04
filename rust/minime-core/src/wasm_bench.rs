//! Bench entry for the speed work (`profile` feature, wasm only): pass 2 on one tile, the
//! current code or (with `reference`) the code from before the GEMM work, timed inside wasm
//! with `prof`'s step times. `rust/minime-core/tools/wasm_pass2/` drives it in Chrome.

use crate::compress::{self, Pass2Params, Recorded, TileFactors};
use crate::prof;
use js_sys::{Float32Array, Float64Array, Object, Reflect};
use wasm_bindgen::prelude::*;

fn set(o: &Object, k: &str, v: &JsValue) {
    Reflect::set(o, &JsValue::from_str(k), v).expect("object");
}

/// `which`: `"new"` or (with `reference`) `"ref"`. `raw` is `(T, px)` uint8, `omega` the
/// tile's recorded Gaussian draws. Returns the wall time of the call (ms), the step
/// times and the tile's factors.
#[wasm_bindgen]
pub fn pass2_bench_tile(
    which: &str,
    params_json: &str,
    raw: &[u8],
    px: usize,
    omega: &[f32],
) -> Result<JsValue, JsError> {
    let p: Pass2Params = serde_json::from_str(params_json).map_err(|e| JsError::new(&e.to_string()))?;
    let mut rng = Recorded::new(omega.to_vec());
    let _ = prof::take();
    let t0 = prof::now_ms();
    let tf: TileFactors = match which {
        #[cfg(feature = "reference")]
        "ref" => crate::compress_ref::compress_tile(raw, p.t, px, &p, &mut rng),
        "new" => compress::compress_tile(raw, p.t, px, &p, &mut rng),
        other => Err(format!("unknown implementation {other}")),
    }
    .map_err(|e| JsError::new(&e))?;
    let ms = prof::now_ms() - t0;
    let out = Object::new();
    set(&out, "ms", &ms.into());
    let steps = Object::new();
    for (name, t) in prof::take() {
        set(&steps, name, &t.into());
    }
    set(&out, "steps", &steps);
    set(&out, "rank", &(tf.rank as f64).into());
    set(&out, "k", &(tf.k as f64).into());
    set(&out, "mean", &Float32Array::from(tf.mean.as_slice()));
    set(&out, "energy", &Float64Array::from(tf.energy.as_slice()));
    set(&out, "sigma", &Float32Array::from(tf.sigma.as_slice()));
    set(&out, "s", &Float32Array::from(tf.s.as_slice()));
    set(&out, "w", &Float32Array::from(tf.w.as_slice()));
    set(&out, "v", &Float32Array::from(tf.v.as_slice()));
    Ok(out.into())
}

/// Cost of the fused multiply-adds `cnmf` uses, on data shaped like its loops: `kind` 0
/// float64 `mul_add` sums (`gram_rows`, `n` frames of a row pair), 1 float32 `mul_add` with
/// 16 sums over the frames (`y_ct_pairs`), 2 the same through `fma::fma32`, 3 float64 and
/// 4 float32 with a separate multiply and add. Returns `[ms, checksum]`.
#[wasm_bindgen]
pub fn fma_bench(kind: u32, n: usize, reps: usize) -> Vec<f64> {
    let mut s = 12345u64;
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let x64: Vec<f64> = (0..n).map(|_| rnd()).collect();
    let y64: Vec<f64> = (0..n).map(|_| rnd()).collect();
    let v32: Vec<f32> = (0..n * 16).map(|_| rnd() as f32).collect();
    let c32: Vec<f32> = (0..n).map(|_| rnd() as f32).collect();
    let t0 = prof::now_ms();
    let mut check = 0f64;
    for _ in 0..reps {
        match kind {
            0 => check += x64.iter().zip(&y64).fold(0.0, |s, (&x, &y)| x.mul_add(y, s)),
            3 => check += x64.iter().zip(&y64).fold(0.0, |s, (&x, &y)| s + x * y),
            _ => {
                let mut acc = [0f32; 16];
                for (row, &xv) in v32.chunks_exact(16).zip(&c32) {
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a = match kind {
                            1 => v.mul_add(xv, *a),
                            2 => crate::fma::fma32(v, xv, *a),
                            _ => *a + v * xv,
                        };
                    }
                }
                check += acc.iter().map(|&v| f64::from(v)).sum::<f64>();
            }
        }
    }
    vec![prof::now_ms() - t0, check]
}
