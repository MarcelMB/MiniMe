//! Throwaway profile of one pass-1 frame, stage by stage, on real frames:
//!
//!     cargo run --release --example profile_frame -- RAW_DIR [TEMPLATE.npy] [N_FRAMES]
//!
//! Every stage is run on the same `N_FRAMES` frames (default 40) from the first AVI; the
//! per-frame mean of each stage is printed in ms. Not production code.

use minime_core::avi::{self, FileReader, ReadAt};
use minime_core::filters::{area_down2_f32, area_down2_u8, gauss_f32, gauss_u8, resize_up2_f32};
use minime_core::motion::{registration_from_denoised, Motion, MotionParams};
use minime_core::npy;
use minime_core::preprocess::{median_blur, shift, tophat};
use std::time::Instant;

fn frames(raw: &str, n: usize) -> Vec<Vec<u8>> {
    let mut avis: Vec<_> = std::fs::read_dir(raw)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "avi"))
        .collect();
    avis.sort();
    let mut r = FileReader::open(&avis[0]).unwrap();
    let ix = avi::index(&mut r).unwrap();
    (0..n)
        .map(|i| {
            let (off, size) = ix.frames[i * 7 % ix.frames.len()];
            ix.unpad(&r.read_at(off, size as usize).unwrap())
        })
        .collect()
}

fn time<T>(name: &str, n: usize, mut f: impl FnMut(usize) -> T) {
    let t = Instant::now();
    for i in 0..n {
        std::hint::black_box(f(i));
    }
    println!("{name:<28} {:7.3} ms", t.elapsed().as_secs_f64() * 1e3 / n as f64);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);
    let fr = frames(&args[1], n);
    let (h, w) = (600, 600);
    let p = MotionParams {
        h,
        w,
        median_ksize: 7,
        tophat_wnd: 15,
        motion_max_shift: 20,
        motion_down: 2,
        template_frames: 200,
        motion_before_tophat: true,
        motion_bandpass: [1.0, 6.0],
    };
    let den: Vec<Vec<u8>> = fr.iter().map(|f| median_blur(f, h, w, 7)).collect();
    let (dy, dx) = (0.37f32, -1.62f32);

    time("median 7", n, |i| median_blur(&fr[i], h, w, 7));
    time("bandpass (registration)", n, |i| {
        registration_from_denoised(&den[i], &p)
    });
    let half: Vec<Vec<f32>> = den
        .iter()
        .map(|d| area_down2_u8(d, h, w).iter().map(|&v| f32::from(v)).collect())
        .collect();
    time("  area_down2_u8 + cvt", n, |i| {
        area_down2_u8(&den[i], h, w)
            .iter()
            .map(|&v| f32::from(v))
            .collect::<Vec<f32>>()
    });
    time("  gauss_f32 300^2 s=3", n, |i| {
        gauss_f32(&half[i], h / 2, w / 2, 3.0)
    });
    let g: Vec<Vec<f32>> = half.iter().map(|x| gauss_f32(x, h / 2, w / 2, 3.0)).collect();
    time("  resize_up2_f32", n, |i| resize_up2_f32(&g[i], h / 2, w / 2));
    time("  gauss_u8 s=1", n, |i| gauss_u8(&den[i], h, w, 1.0));
    let reg: Vec<_> = den.iter().map(|d| registration_from_denoised(d, &p)).collect();
    time("shift u8", n, |i| shift(&den[i], h, w, dy, dx));
    time("tophat 15", n, |i| tophat(&den[i], h, w, 15));
    time("shift_f32 (reg.shifted)", n, |i| reg[i].shifted(h, w, dy, dx));
    time("area_down2_f32", n, |_| area_down2_f32(&vec![1.0; h * w], h, w));

    if let Some(t) = args.get(2).filter(|t| t.ends_with(".npy")) {
        let img = npy::read_f32(&std::fs::read(t).unwrap()).unwrap().data;
        let m = Motion::from_template_image(&p, img).unwrap();
        time("motion.estimate", n, |i| m.estimate(&reg[i]));
        let warped: Vec<_> = reg.iter().map(|r| r.shifted(h, w, dy, dx)).collect();
        time("motion.quality", n, |i| m.quality(&warped[i]));
    }
}
