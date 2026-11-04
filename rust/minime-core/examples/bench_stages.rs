//! Native single-thread cost per frame of every stage of `Pass1::push`, on real frames
//! (the `gpu_ref` dump): which share could a GPU take over?
//! cargo run --release --example bench_stages -- <dir with full_frames.u8>
use minime_core::motion::{bandpass, registration_from_denoised, Motion, MotionParams};
use minime_core::preprocess::{median_blur, shift, tophat};
use std::time::Instant;

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let all = std::fs::read(format!("{dir}/full_frames.u8")).unwrap();
    let (h, w) = (600, 600);
    let frames: Vec<&[u8]> = all.chunks_exact(h * w).take(64).collect();
    let p: MotionParams = serde_json::from_str(
        r#"{"H":600,"W":600,"median_ksize":7,"tophat_wnd":15,"motion_max_shift":20,"motion_down":2,
            "template_frames":20,"motion_before_tophat":true,"motion_bandpass":[1.0,6.0]}"#,
    )
    .unwrap();
    let dens: Vec<Vec<u8>> = frames.iter().map(|f| median_blur(f, h, w, 7)).collect();
    let motion = Motion::build(&p, 20, &mut |t| Ok(registration_from_denoised(&dens[t], &p))).unwrap();
    let n = frames.len();
    let mut time = |name: &str, f: &mut dyn FnMut(usize)| {
        for i in 0..4 {
            f(i);
        }
        let t = Instant::now();
        for i in 0..n {
            f(i);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!("{name:<28} {ms:7.3} ms/frame");
        ms
    };
    let a = time("median 7x7", &mut |i| {
        std::hint::black_box(median_blur(frames[i], h, w, 7));
    });
    let regs: Vec<_> = dens.iter().map(|d| registration_from_denoised(d, &p)).collect();
    let b = time("bandpass registration image", &mut |i| {
        std::hint::black_box(bandpass(&dens[i], h, w, 1.0, 6.0));
    });
    let mut shifts = Vec::new();
    let c = time("estimate (FFT x-corr)", &mut |i| {
        let s = motion.estimate(&regs[i]);
        if shifts.len() <= i {
            shifts.push(s);
        }
    });
    let d = time("shift u8", &mut |i| {
        let s = shifts[i % shifts.len()];
        std::hint::black_box(shift(&dens[i], h, w, s[0] as f32, s[1] as f32));
    });
    let sh: Vec<Vec<u8>> = (0..n)
        .map(|i| shift(&dens[i], h, w, shifts[i][0] as f32, shifts[i][1] as f32))
        .collect();
    let e = time("tophat 15", &mut |i| {
        std::hint::black_box(tophat(&sh[i], h, w, 15));
    });
    let g = time("reg.shifted + quality", &mut |i| {
        let s = shifts[i];
        let r = regs[i].shifted(h, w, s[0] as f32, s[1] as f32);
        std::hint::black_box(motion.quality(&r));
    });
    println!(
        "sum median+shift+tophat {:.2}; + reg/estimate/quality {:.2}; GPU-able integer share {:.0} % of {:.2}",
        a + d + e,
        b + c + g,
        100.0 * (a + d + e) / (a + b + c + d + e + g),
        a + b + c + d + e + g
    );
}
