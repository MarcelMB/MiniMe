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
