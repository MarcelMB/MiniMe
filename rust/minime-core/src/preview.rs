//! The launcher's pass-1 preview of a raw session (`minime.serve.preview`): any frame through
//! the median, the registration image and the tophat as four uint8 panels, and the motion
//! check on a short clip. The steps are the ones pass 1 runs (`preprocess`, `motion`).

use crate::motion::{registration_from_denoised, MotionParams, Reg, TemplateBuilder};
use crate::preprocess::{median_blur, tophat};
use serde::Serialize;

/// `preview.PANELS`: the order of the four panels of `frame_panels`.
pub const PANELS: [&str; 4] = ["raw", "denoised", "registration", "tophat"];

/// `preview.motion_check`'s answer.
#[derive(Debug, Serialize)]
pub struct MotionCheck {
    pub t0: usize,
    pub frames: usize,
    pub max_shift_px: f64,
    pub quality_median: f64,
    pub shifts: Vec<[f64; 2]>,
}

/// The motion parameters from a pass-1 parameter object (`session.json`'s `pass1`, or the
/// motion subset of `Params.asdict()`), for `h` x `w` frames; the object's own `H` and `W`
/// are replaced and the rest of its keys ignored.
pub fn motion_params(json: &str, h: usize, w: usize) -> Result<MotionParams, String> {
    let mut v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("params: {e}"))?;
    let o = v.as_object_mut().ok_or("params: expected an object")?;
    o.insert("H".into(), h.into());
    o.insert("W".into(), w.into());
    serde_json::from_value(v).map_err(|e| format!("params: {e}"))
}

/// What `registration_from_denoised` would index out of bounds on: no pixels, or an odd size
/// for the band-pass image (`motion::bandpass` halves the frame).
fn check_size(p: &MotionParams) -> Result<(), String> {
    if p.h == 0 || p.w == 0 {
        return Err(format!("{} x {} frames hold no pixels", p.h, p.w));
    }
    if p.motion_before_tophat && !(p.h.is_multiple_of(2) && p.w.is_multiple_of(2)) {
        return Err(format!(
            "{} x {} frames: the band-pass registration image needs an even size",
            p.h, p.w
        ));
    }
    Ok(())
}

/// `preview.frame_panels` of one raw frame (`h * w` uint8): `(4, h, w)` uint8, in `PANELS` order.
pub fn frame_panels(frame: &[u8], p: &MotionParams) -> Result<Vec<u8>, String> {
    check_size(p)?;
    let px = p.h * p.w;
    if frame.len() != px {
        return Err(format!(
            "a {} x {} frame holds {px} pixels, got {}",
            p.h,
            p.w,
            frame.len()
        ));
    }
    let den = median_blur(frame, p.h, p.w, p.median_ksize);
    let reg = registration_from_denoised(&den, p);
    let top = tophat(&den, p.h, p.w, p.tophat_wnd);
    let mut out = Vec::with_capacity(4 * px);
    out.extend_from_slice(frame);
    out.extend_from_slice(&den);
    out.extend(match &reg {
        Reg::F32(r) => scale_u8(r),
        Reg::U8(r) => scale_u8(&r.iter().map(|&v| f32::from(v)).collect::<Vec<_>>()),
    });
    out.extend_from_slice(&top);
    Ok(out)
}

/// `preview.motion_check` of `n` consecutive raw frames (`clip`, `n * h * w` uint8) that start
/// at frame `t0`: rigid shifts against a template built from the clip itself, and the median
/// frame quality. As in Python, the quality is of the registration images before the shift.
pub fn motion_check(clip: &[u8], n: usize, t0: usize, p: &MotionParams) -> Result<MotionCheck, String> {
    check_size(p)?;
    let px = p.h * p.w;
    if n == 0 {
        return Err("the clip has no frames".into());
    }
    if clip.len() != n * px {
        return Err(format!(
            "a clip of {n} frames of {} x {} holds {} pixels, got {}",
            p.h,
            p.w,
            n * px,
            clip.len()
        ));
    }
    let regs: Vec<Reg> = clip
        .chunks_exact(px)
        .map(|f| registration_from_denoised(&median_blur(f, p.h, p.w, p.median_ksize), p))
        .collect();
    // `motion.make_template`: every frame twice, in order, without copying the frames
    let mut b = TemplateBuilder::new(p, n)?;
    for _round in 0..2 {
        regs.iter().for_each(|r| b.push(r));
    }
    let tmpl = b.finish()?;
    let sh: Vec<[f64; 2]> = regs.iter().map(|r| tmpl.estimate(r)).collect();
    let quality: Vec<f32> = regs.iter().map(|r| tmpl.quality(r)).collect();
    Ok(MotionCheck {
        t0,
        frames: n,
        max_shift_px: sh.iter().flatten().fold(0.0, |m, v| f64::max(m, v.abs())),
        quality_median: f64::from(median(quality)),
        shifts: sh.iter().map(|s| s.map(round2)).collect(),
    })
}

/// `np.round(x, 2)`: the product with 100 rounded half to even.
fn round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

/// `np.median` of a float32 array: the middle value, or the float32 mean of the middle two.
fn median(mut q: Vec<f32>) -> f32 {
    q.sort_unstable_by(f32::total_cmp);
    let n = q.len();
    if n % 2 == 1 {
        q[n / 2]
    } else {
        (q[n / 2 - 1] + q[n / 2]) / 2.0
    }
}

/// `np.percentile(a, 100 q)` (linear) of an ascending float32 array, as numpy's `_lerp` computes
/// it: the float32 difference of the two neighbours scaled in float64.
fn quantile(sorted: &[f32], q: f64) -> f64 {
    let virt = (sorted.len() - 1) as f64 * q;
    let prev = virt.floor() as usize;
    let next = (prev + 1).min(sorted.len() - 1);
    let gamma = virt - prev as f64;
    let (a, b) = (sorted[prev], sorted[next]);
    let diff = f64::from(b - a);
    if gamma >= 0.5 {
        f64::from(b) - diff * (1.0 - gamma)
    } else {
        f64::from(a) + diff * gamma
    }
}

/// `preview._u8`: `a` scaled to uint8 between its 1st and 99.7th percentile.
fn scale_u8(a: &[f32]) -> Vec<u8> {
    let mut sorted = a.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let lo = quantile(&sorted, 1.0 / 100.0);
    let hi = quantile(&sorted, 99.7 / 100.0);
    let k = 255.0 / (hi - lo).max(1e-6);
    // truncated, as numpy's `astype(np.uint8)` of the clipped values
    a.iter()
        .map(|&v| ((f64::from(v) - lo) * k).clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npy;

    // `Params()` motion parameters, and the same with the tophat as registration image
    const DEFAULT: &str = r#"{"median_ksize": 7, "tophat_wnd": 15, "motion_max_shift": 20, "motion_down": 2,
        "template_frames": 200, "motion_before_tophat": true, "motion_bandpass": [1.0, 6.0]}"#;
    const TOPHAT: &str = r#"{"median_ksize": 5, "tophat_wnd": 8, "motion_max_shift": 20, "motion_down": 2,
        "template_frames": 200, "motion_before_tophat": false, "motion_bandpass": [1.0, 6.0]}"#;

    // Python's `frame_panels` of the e2e session's frame 150 (`np.save`), and `motion_check` on its
    // frames 100..108 (`json.dumps`, with `read_clip` and `clip_start` answering the saved clip);
    // 64 x 64 frames, `Params()` and the same parameters as above
    const PANELS_DEFAULT: &[u8] = include_bytes!("../tests/fixtures/preview_panels.npy");
    const PANELS_TOPHAT: &[u8] = include_bytes!("../tests/fixtures/preview_panels_tophat.npy");
    const CLIP: &[u8] = include_bytes!("../tests/fixtures/preview_clip.npy");
    const MOTION_DEFAULT: &str = include_str!("../tests/fixtures/preview_motion.json");
    const MOTION_TOPHAT: &str = include_str!("../tests/fixtures/preview_motion_tophat.json");

    /// Largest grey-level difference per panel against Python's `(4, 64, 64)`.
    fn panel_diffs(want: &[u8], params: &str) -> Vec<u8> {
        let w = npy::read_u8(want).unwrap();
        assert_eq!(w.shape, [4, 64, 64]);
        let px = 64 * 64;
        let p = motion_params(params, 64, 64).unwrap();
        // the raw panel is the frame itself
        let got = frame_panels(&w.data[..px], &p).unwrap();
        assert_eq!(got.len(), 4 * px);
        (0..4)
            .map(|k| {
                let r = k * px..(k + 1) * px;
                got[r.clone()]
                    .iter()
                    .zip(&w.data[r])
                    .map(|(&a, &b)| a.abs_diff(b))
                    .max()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn preview_panels_match_python() {
        assert_eq!(PANELS, ["raw", "denoised", "registration", "tophat"]);
        assert_eq!(panel_diffs(PANELS_DEFAULT, DEFAULT), [0, 0, 0, 0]);
    }

    #[test]
    fn preview_panels_match_python_with_the_tophat_as_registration_image() {
        assert_eq!(panel_diffs(PANELS_TOPHAT, TOPHAT), [0, 0, 0, 0]);
    }

    /// Rust's answer on the fixture clip against Python's, key by key.
    fn check_motion(params: &str, want_json: &str) {
        let clip = npy::read_u8(CLIP).unwrap();
        assert_eq!(clip.shape, [8, 64, 64]);
        let p = motion_params(params, 64, 64).unwrap();
        let got = motion_check(&clip.data, 8, 100, &p).unwrap();
        let want: serde_json::Value = serde_json::from_str(want_json).unwrap();
        assert_eq!(got.t0, want["t0"]);
        assert_eq!(got.frames, want["frames"]);
        let close = |a: f64, b: &serde_json::Value, tol: f64, what: &str| {
            let b = b.as_f64().unwrap();
            assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
        };
        close(got.max_shift_px, &want["max_shift_px"], 1e-3, "max_shift_px");
        close(
            got.quality_median,
            &want["quality_median"],
            1e-3,
            "quality_median",
        );
        let want_shifts = want["shifts"].as_array().unwrap();
        assert_eq!(got.shifts.len(), want_shifts.len());
        for (i, (g, w)) in got.shifts.iter().zip(want_shifts).enumerate() {
            for k in 0..2 {
                close(g[k], &w[k], 1e-3, &format!("shifts[{i}][{k}]"));
            }
        }
    }

    #[test]
    fn motion_check_matches_python() {
        check_motion(DEFAULT, MOTION_DEFAULT);
    }

    #[test]
    fn motion_check_matches_python_with_the_tophat_as_registration_image() {
        check_motion(TOPHAT, MOTION_TOPHAT);
    }

    #[test]
    fn scale_u8_is_numpys_percentile_scaling() {
        let a = [
            3.0, 7.0, 7.0, 1.0, 12.0, 5.0, 0.0, 9.0, 4.0, 4.0, 30.0, 2.0, -2.5, 0.25,
        ];
        let want = [41, 74, 74, 25, 114, 58, 17, 90, 50, 50, 255, 33, 0, 19];
        assert_eq!(scale_u8(&a), want);
        let b: Vec<f32> = a[..12].iter().map(|v| v.max(0.0)).collect();
        assert_eq!(scale_u8(&b), [25, 59, 59, 7, 103, 42, 0, 77, 33, 33, 255, 16]);
        // a flat image: the range is floored at 1e-6, every pixel is at the low end
        assert_eq!(scale_u8(&[9.5; 6]), [0; 6]);
    }

    #[test]
    fn motion_params_take_the_frame_size_from_the_caller() {
        let p = motion_params(&DEFAULT.replace('{', r#"{"H": 7, "W": 9, "tile": 64, "#), 64, 48).unwrap();
        assert_eq!((p.h, p.w, p.median_ksize), (64, 48, 7));
        assert!(motion_params(r#"{"median_ksize": 7}"#, 64, 64).is_err());
        assert!(motion_params("[]", 64, 64).is_err());
    }

    #[test]
    fn wrong_sized_input_is_an_error_not_a_panic() {
        let p = motion_params(DEFAULT, 64, 64).unwrap();
        let e = frame_panels(&[0; 100], &p).unwrap_err();
        assert!(e.contains("64 x 64"), "{e}");
        assert!(motion_check(&[0; 100], 1, 0, &p).is_err());
        assert!(motion_check(&[], 0, 0, &p).is_err());
        // the band-pass registration image halves the frame
        let odd = motion_params(DEFAULT, 63, 64).unwrap();
        let e = frame_panels(&vec![0; 63 * 64], &odd).unwrap_err();
        assert!(e.contains("even"), "{e}");
        let flat = motion_params(TOPHAT, 63, 64).unwrap();
        assert!(frame_panels(&vec![0; 63 * 64], &flat).is_ok());
    }

    #[test]
    fn a_tophat_window_larger_than_the_frame_does_not_panic() {
        let p = motion_params(&DEFAULT.replace("15", "100"), 64, 64).unwrap();
        assert_eq!(p.tophat_wnd, 100);
        let frame: Vec<u8> = (0..64 * 64).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(frame_panels(&frame, &p).unwrap().len(), 4 * 64 * 64);
    }
}
