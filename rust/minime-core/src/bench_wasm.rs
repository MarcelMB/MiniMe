//! Micro-benchmark entry points for the wasm build (feature `bench`, driven by
//! `tools/wasm-bench/`): every kernel in its old (scalar reference) and new form on one frame.

use crate::filters::{
    area_down2_f32, area_down2_f32_scalar, area_down2_u8, area_down2_u8_scalar, gauss_f32, gauss_f32_scalar,
    gauss_u8, gauss_u8_scalar, resize_up2_f32, resize_up2_f32_scalar,
};
use crate::motion::{registration_from_denoised, Motion, MotionParams, Reg};
use crate::preprocess::{
    median_blur, median_blur_huang, shift, shift_f32, shift_f32_scalar, shift_scalar, tophat, tophat_scalar,
};
use js_sys::{Function, Reflect};
use wasm_bindgen::prelude::*;

type Job = Box<dyn FnMut() -> Vec<u8>>;

fn params(h: usize, w: usize) -> MotionParams {
    MotionParams {
        h,
        w,
        median_ksize: 7,
        tophat_wnd: 15,
        motion_max_shift: 20,
        motion_down: 2,
        template_frames: 200,
        motion_before_tophat: true,
        motion_bandpass: [1.0, 6.0],
    }
}

/// `motion::bandpass` with the scalar filters it was first written with.
fn bandpass_scalar(den: &[u8], h: usize, w: usize) -> Vec<f32> {
    let half: Vec<f32> = area_down2_u8(den, h, w).iter().map(|&v| f32::from(v)).collect();
    let wide = resize_up2_f32_scalar(&gauss_f32_scalar(&half, h / 2, w / 2, 3.0), h / 2, w / 2);
    gauss_u8_scalar(den, h, w, 1.0)
        .iter()
        .zip(&wide)
        .map(|(&n, &b)| f32::from(n) - b)
        .collect()
}

/// One pass-1 frame as `Pass1::push` does it with a given shift: median, registration image,
/// shift, tophat, shifted registration image (its output bytes: the corrected frame and the
/// shifted registration image); `old` uses the scalar reference of every kernel.
fn pass1_frame(old: bool, raw: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (dy, dx) = (0.37f32, -1.62f32);
    let p = params(h, w);
    if old {
        let den = median_blur_huang(raw, h, w, 7);
        let reg = bandpass_scalar(&den, h, w);
        let blk = tophat_scalar(&shift_scalar(&den, h, w, dy, dx), h, w, 15);
        let mut out = blk;
        out.extend(f32_bytes(&shift_f32_scalar(&reg, h, w, dy, dx)));
        out
    } else {
        let den = median_blur(raw, h, w, 7);
        let reg = registration_from_denoised(&den, &p).shifted(h, w, dy, dx);
        let mut out = tophat(&shift(&den, h, w, dy, dx), h, w, 15);
        if let Reg::F32(r) = reg {
            out.extend(f32_bytes(&r));
        }
        out
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn job(kernel: &str, new: bool, frame: &[u8], h: usize, w: usize) -> Result<Job, JsError> {
    let f = frame.to_vec();
    let ff: Vec<f32> = frame.iter().map(|&v| f32::from(v)).collect();
    let half = area_down2_f32(&ff, h, w);
    let (dy, dx) = (0.37f32, -1.62f32);
    Ok(match (kernel, new) {
        ("median", false) => Box::new(move || median_blur_huang(&f, h, w, 7)),
        ("median", true) => Box::new(move || crate::median::median7(&f, h, w)),
        ("tophat", false) => Box::new(move || tophat_scalar(&f, h, w, 15)),
        ("tophat", true) => Box::new(move || tophat(&f, h, w, 15)),
        ("gauss_u8", false) => Box::new(move || gauss_u8_scalar(&f, h, w, 1.0)),
        ("gauss_u8", true) => Box::new(move || gauss_u8(&f, h, w, 1.0)),
        ("shift", false) => Box::new(move || shift_scalar(&f, h, w, dy, dx)),
        ("shift", true) => Box::new(move || shift(&f, h, w, dy, dx)),
        ("shift_f32", false) => Box::new(move || f32_bytes(&shift_f32_scalar(&ff, h, w, dy, dx))),
        ("shift_f32", true) => Box::new(move || f32_bytes(&shift_f32(&ff, h, w, dy, dx))),
        ("gauss_f32", false) => Box::new(move || f32_bytes(&gauss_f32_scalar(&half, h / 2, w / 2, 3.0))),
        ("gauss_f32", true) => Box::new(move || f32_bytes(&gauss_f32(&half, h / 2, w / 2, 3.0))),
        ("pass1_frame", old) => Box::new(move || pass1_frame(!old, &f, h, w)),
        ("estimate", _) => {
            let p = params(h, w);
            let reg = registration_from_denoised(&median_blur(&f, h, w, 7), &p);
            let template: Vec<f32> = half.iter().map(|&v| v - 100.0).collect();
            let m = Motion::from_template_image(&p, template).map_err(|e| JsError::new(&e))?;
            Box::new(move || m.estimate(&reg).iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        ("resize_up2", false) => Box::new(move || f32_bytes(&resize_up2_f32_scalar(&half, h / 2, w / 2))),
        ("resize_up2", true) => Box::new(move || f32_bytes(&resize_up2_f32(&half, h / 2, w / 2))),
        ("area_down2_u8", false) => Box::new(move || area_down2_u8_scalar(&f, h, w)),
        ("area_down2_u8", true) => Box::new(move || area_down2_u8(&f, h, w)),
        ("area_down2_f32", false) => Box::new(move || f32_bytes(&area_down2_f32_scalar(&ff, h, w))),
        ("area_down2_f32", true) => Box::new(move || f32_bytes(&area_down2_f32(&ff, h, w))),
        _ => return Err(JsError::new(&format!("unknown kernel {kernel}"))),
    })
}

/// The kernel's output on `frame` (float32 results as little-endian bytes).
#[wasm_bindgen]
pub fn bench_out(kernel: &str, new: bool, frame: &[u8], h: usize, w: usize) -> Result<Vec<u8>, JsError> {
    Ok(job(kernel, new, frame, h, w)?())
}

/// Milliseconds per call over `reps` calls (the input is prepared before the clock starts).
#[wasm_bindgen]
pub fn bench_time(
    kernel: &str,
    new: bool,
    frame: &[u8],
    h: usize,
    w: usize,
    reps: usize,
) -> Result<f64, JsError> {
    let mut j = job(kernel, new, frame, h, w)?;
    let perf =
        Reflect::get(&js_sys::global(), &"performance".into()).map_err(|_| JsError::new("no performance"))?;
    let now: Function = Reflect::get(&perf, &"now".into())
        .map_err(|_| JsError::new("no now"))?
        .into();
    let t = |p: &JsValue| now.call0(p).ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
    std::hint::black_box(j());
    let t0 = t(&perf);
    for _ in 0..reps {
        std::hint::black_box(j());
    }
    Ok((t(&perf) - t0) / reps as f64)
}
