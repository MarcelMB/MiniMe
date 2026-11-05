//! What the GPU path of the per-frame stages (`web/src/gpu/`) needs to give the CPU path's bits:
//! the kernels' settings (frame size, median and top-hat windows, the band-pass taps) and, for the
//! start-up self-test, the CPU's result of every kernel on a frame.
//!
//! The GPU runs median, band-pass registration image, its 2x2 area downsampling, the warp of the
//! denoised frame and the top-hat, and the warp of the registration image and its downsampling.
//! The CPU keeps what follows on the small images (the FFT estimate, the quality) and the fold, the
//! same code as `Pass1::push` (`Motion::estimate_down`, `Motion::quality_down`,
//! `Pass1::fold_corrected`).

use crate::filters::bandpass_taps;
use crate::motion::{registration_from_denoised, Correlator, MotionParams};
use crate::preprocess::{median_blur, shift, tophat};
use serde::Serialize;

/// Settings of the GPU kernels for one stage (`web/src/gpu/kernels.ts`).
#[derive(Debug, Clone, Serialize)]
pub struct Kernels {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    /// Median window (1: no median).
    pub median: usize,
    /// Top-hat half window (0: no top-hat).
    pub tophat: usize,
    /// The band-pass registration image's fixed-point taps (sigma `lo`) and float32 taps (sigma
    /// `hi / 2` on the half-resolution image); empty for a stage without it (the streaming pass).
    pub u8taps: Vec<u32>,
    pub f32taps: Vec<f32>,
}

/// The kernels of a stage, or why the GPU cannot give the CPU's bits for it. `band`: the
/// band-pass sigmas and the motion downsampling, for a stage that needs the registration image.
pub fn kernels(
    h: usize,
    w: usize,
    median: usize,
    tophat_wnd: usize,
    before_tophat: bool,
    band: Option<([f64; 2], usize)>,
) -> Result<Kernels, String> {
    if !before_tophat {
        return Err("motion_before_tophat is off (the GPU runs median, shift, then top-hat)".into());
    }
    if h < 32 || w < 32 {
        return Err(format!("{h} x {w} frames are too small for the GPU kernels"));
    }
    if median > 1 && median.is_multiple_of(2) {
        return Err(format!("median_ksize must be odd, got {median}"));
    }
    let (mut u8taps, mut f32taps) = (Vec::new(), Vec::new());
    if let Some(([lo, hi], down)) = band {
        if down != 2 {
            return Err(format!("motion_down {down} (the GPU path downsamples by 2)"));
        }
        if !h.is_multiple_of(2) || !w.is_multiple_of(2) {
            return Err(format!(
                "{h} x {w} frames: the registration image needs an even size"
            ));
        }
        (u8taps, f32taps) = bandpass_taps(lo, hi);
        // one reflection at the borders, as the kernels assume
        if u8taps.len() / 2 >= h.min(w) || f32taps.len() / 2 >= (h / 2).min(w / 2) {
            return Err(format!("band-pass {lo}, {hi} is too wide for {h} x {w} frames"));
        }
    }
    Ok(Kernels {
        h,
        w,
        median: median.max(1),
        tophat: tophat_wnd,
        u8taps,
        f32taps,
    })
}

/// The kernels of a pass-1 or template stage (its motion parameters).
pub fn motion_kernels(p: &MotionParams) -> Result<Kernels, String> {
    kernels(
        p.h,
        p.w,
        p.median_ksize,
        p.tophat_wnd,
        p.motion_before_tophat,
        Some((p.motion_bandpass, p.motion_down)),
    )
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The CPU's result of every GPU kernel on one raw frame (`h * w`, no row padding) and a shift:
/// `den` (median), `down` (the downsampled registration image, float32 bytes), `blk` (shift and
/// top-hat of `den`) and `warped_down` (the shifted registration image, downsampled).
pub fn reference(raw: &[u8], p: &MotionParams, dy: f32, dx: f32) -> Result<Vec<(String, Vec<u8>)>, String> {
    motion_kernels(p)?;
    let (h, w) = (p.h, p.w);
    if raw.len() != h * w {
        return Err(format!("a frame of {} bytes, expected {h} x {w}", raw.len()));
    }
    let corr = Correlator::new(h, w, p.motion_down)?;
    let den = median_blur(raw, h, w, p.median_ksize);
    let reg = registration_from_denoised(&den, p);
    let down = corr.downsample(&reg);
    let blk = tophat(&shift(&den, h, w, dy, dx), h, w, p.tophat_wnd);
    let warped_down = corr.downsample(&reg.shifted(h, w, dy, dx));
    let full = match &reg {
        crate::motion::Reg::F32(x) => f32_bytes(x),
        crate::motion::Reg::U8(x) => x.clone(),
    };
    Ok(vec![
        ("den".into(), den),
        ("reg".into(), full),
        ("down".into(), f32_bytes(&down)),
        ("blk".into(), blk),
        ("warped_down".into(), f32_bytes(&warped_down)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::{registration_image, Motion};
    use crate::pass1::{Pass1, Pass1Params};

    fn params(h: usize, w: usize) -> MotionParams {
        serde_json::from_value(serde_json::json!({
            "H": h, "W": w, "median_ksize": 7, "tophat_wnd": 15, "motion_max_shift": 20,
            "motion_down": 2, "template_frames": 4, "motion_before_tophat": true,
            "motion_bandpass": [1.0, 6.0]
        }))
        .unwrap()
    }

    /// Smooth blobs plus noise, moved by `t` px: frames with structure for the estimate.
    fn frame(h: usize, w: usize, t: usize) -> Vec<u8> {
        let mut s = 7 + t as u64;
        (0..h * w)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let (y, x) = ((i / w) as f64, (i % w) as f64 + 0.7 * t as f64);
                let b = 80.0 * (-((y - 30.0).powi(2) + (x - 40.0).powi(2)) / 60.0).exp()
                    + 60.0 * (-((y - 70.0).powi(2) + (x - 20.0).powi(2)) / 30.0).exp();
                (40.0 + b + (s >> 59) as f64).min(255.0) as u8
            })
            .collect()
    }

    #[test]
    fn taps_are_the_filters_own() {
        let (u8t, f32t) = bandpass_taps(1.0, 6.0);
        assert_eq!(u8t, vec![1, 14, 62, 102, 62, 14, 1]);
        assert_eq!(f32t.len(), 25);
        assert!(kernels(100, 100, 7, 15, false, None).is_err());
        assert!(kernels(100, 101, 7, 15, true, Some(([1.0, 6.0], 2))).is_err());
        assert!(kernels(100, 100, 7, 15, true, Some(([1.0, 6.0], 1))).is_err());
        assert_eq!(kernels(100, 100, 0, 0, true, None).unwrap().median, 1);
    }

    /// The split the GPU path takes (downsampled estimate and quality, `fold_corrected`) gives
    /// `Pass1::push`'s results bit for bit, estimated or given shifts.
    #[test]
    fn split_push_is_push() {
        let (h, w) = (96, 128);
        let p = params(h, w);
        let template = Correlator::new(h, w, 2)
            .unwrap()
            .prep(&registration_image(&frame(h, w, 0), &p));
        let pp: Pass1Params = serde_json::from_value(serde_json::json!({
            "H": h, "W": w, "median_ksize": 7, "tophat_wnd": 15, "motion_max_shift": 20,
            "motion_down": 2, "template_frames": 4, "motion_before_tophat": true,
            "motion_bandpass": [1.0, 6.0], "tile": 64, "seeds_wnd": 4, "seeds_stp": 2, "t0": 0, "t1": 6
        }))
        .unwrap();
        let motion = || Motion::from_template_image(&p, template.clone()).unwrap();
        for given in [false, true] {
            let mut a = Pass1::new(&pp, Some(motion()));
            let mut b = Pass1::new(&pp, Some(motion()));
            for t in 0..6 {
                let raw = frame(h, w, t);
                let g = given.then_some([0.25 * t as f32, -1.5]);
                a.push(&raw, g).unwrap();
                // the GPU path, its kernels done here by the CPU reference
                let m = b.motion().unwrap();
                let corr = m.correlator();
                let den = median_blur(&raw, h, w, 7);
                let reg = registration_from_denoised(&den, &p);
                let used = match g {
                    Some(s) => s,
                    None => {
                        let e = m.estimate_down(corr.downsample(&reg));
                        b.push_estimate(e);
                        e.map(|v| v as f32)
                    }
                };
                let m = b.motion().unwrap();
                let q = m.quality_down(m.correlator().downsample(&reg.shifted(h, w, used[0], used[1])));
                let blk = tophat(&shift(&den, h, w, used[0], used[1]), h, w, 15);
                b.fold_corrected(&blk, Some(q), used);
            }
            assert_eq!(a.shifts, b.shifts);
            assert_eq!(a.shifts64, b.shifts64);
            assert_eq!(
                a.quality.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.quality.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(a.max_proj, b.max_proj);
            assert_eq!(a.block_max, b.block_max);
            assert_eq!(a.take_tiles(), b.take_tiles());
            assert!(!given || a.shifts64.is_empty());
        }
    }

    #[test]
    fn reference_holds_every_kernel() {
        let (h, w) = (64, 80);
        let out = reference(&frame(h, w, 3), &params(h, w), 1.25, -0.5).unwrap();
        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["den", "reg", "down", "blk", "warped_down"]);
        assert_eq!(out[2].1.len(), 4 * (h / 2) * (w / 2));
        assert_eq!(out[3].1, tophat(&shift(&out[0].1, h, w, 1.25, -0.5), h, w, 15));
    }
}
