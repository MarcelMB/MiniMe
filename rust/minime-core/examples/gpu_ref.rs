//! Reference data for the WebGPU kernel checks (`experiments/webgpu/`): real frames from the
//! 5-min synth, random shifts, and the CPU results of `preprocess.rs` (the code the wasm build
//! runs), so the GPU kernels can be compared byte for byte.
//!
//! cargo run --release --example gpu_ref -- <raw dir> <out dir>

use minime_core::avi::{index, FileReader};
use minime_core::preprocess::{median_blur, shift, tophat};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

const MEDIAN: usize = 7;
const TOPHAT: usize = 15;
/// (file, first frame, count): four stretches of consecutive frames from different files.
const PICKS: [(&str, usize, usize); 4] = [
    ("0.avi", 200, 64),
    ("2.avi", 500, 64),
    ("5.avi", 100, 64),
    ("8.avi", 700, 64),
];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// uniform in [0, 1)
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f32 {
        (lo + (hi - lo) * self.unit()) as f32
    }
}

/// Shifts covering the cases that matter: none, small sub-pixel, whole pixels, exact halves
/// (rounding ties), large (border fills with zeros), tiny negative, and way outside the frame.
fn make_shifts(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed);
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        let (dy, dx) = match i % 10 {
            0 => (0.0, 0.0),
            1 => (r.range(-1.0, 1.0), r.range(-1.0, 1.0)),
            2 | 3 | 4 => (r.range(-8.0, 8.0), r.range(-8.0, 8.0)),
            5 => ((r.range(-6.0, 6.0)).round(), (r.range(-6.0, 6.0)).round()),
            6 => (
                (r.range(-6.0, 6.0)).round() + 0.5,
                (r.range(-6.0, 6.0)).round() - 0.5,
            ),
            7 => (r.range(-40.0, 40.0), r.range(-40.0, 40.0)),
            8 => (-1.0e-7, 2.0e-7),
            _ => (r.range(-700.0, 700.0), r.range(-3.0, 3.0)),
        };
        out.push(dy);
        out.push(dx);
    }
    out
}

/// OpenCV's `getGaussianKernelBitExact` in float64 (as `filters.rs`).
fn gauss_taps(n: usize, sigma: f64) -> Vec<f64> {
    let n2 = (n - 1) / 2;
    let scale2x = -0.125 / (sigma * sigma);
    let vals: Vec<f64> = (0..n2)
        .map(|i| {
            let x = 1 - n as i64 + 2 * i as i64;
            ((x * x) as f64 * scale2x).exp()
        })
        .collect();
    let sum = vals.iter().sum::<f64>() * 2.0 + 1.0;
    let mul1 = 1.0 / sum;
    let mut k = vec![0.0; n];
    for (i, v) in vals.iter().enumerate() {
        k[i] = v * mul1;
        k[n - 1 - i] = v * mul1;
    }
    k[n2] = mul1;
    k
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    File::create(dir.join(name)).unwrap().write_all(bytes).unwrap();
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Results for `frames` (each `h * w`): median, shift of the raw and of the median frame,
/// tophat of the raw frame and of the shifted median frame (the pipeline's order), their max
/// projection and per-block maxima.
fn emit(dir: &Path, tag: &str, frames: &[Vec<u8>], h: usize, w: usize, seed: u64) {
    let n = frames.len();
    let shifts = make_shifts(n, seed);
    let mut den = Vec::new();
    let mut shraw = Vec::new();
    let mut shden = Vec::new();
    let mut top = Vec::new();
    let mut tophat_raw = Vec::new();
    let mut raw = Vec::new();
    let mut maxproj = vec![0u8; h * w];
    // blocks of about a third of the frames, uneven on purpose
    let edges = [0, n / 4 + 1, n * 5 / 8, n];
    let mut block_max = vec![0u8; (edges.len() - 1) * h * w];
    for (i, f) in frames.iter().enumerate() {
        let (dy, dx) = (shifts[2 * i], shifts[2 * i + 1]);
        let d = median_blur(f, h, w, MEDIAN);
        let sd = shift(&d, h, w, dy, dx);
        let t = tophat(&sd, h, w, TOPHAT);
        raw.extend_from_slice(f);
        shraw.extend(shift(f, h, w, dy, dx));
        tophat_raw.extend(tophat(f, h, w, TOPHAT));
        let b = (0..edges.len() - 1)
            .find(|&b| i >= edges[b] && i < edges[b + 1])
            .unwrap();
        for (px, &v) in t.iter().enumerate() {
            maxproj[px] = maxproj[px].max(v);
            let bm = &mut block_max[b * h * w + px];
            *bm = (*bm).max(v);
        }
        den.extend(d);
        shden.extend(sd);
        top.extend(t);
    }
    write(dir, &format!("{tag}_frames.u8"), &raw);
    write(dir, &format!("{tag}_shifts.f32"), &f32_bytes(&shifts));
    write(dir, &format!("{tag}_median.u8"), &den);
    write(dir, &format!("{tag}_shift_raw.u8"), &shraw);
    write(dir, &format!("{tag}_shift_median.u8"), &shden);
    write(dir, &format!("{tag}_tophat_raw.u8"), &tophat_raw);
    write(dir, &format!("{tag}_final.u8"), &top);
    write(dir, &format!("{tag}_maxproj.u8"), &maxproj);
    write(dir, &format!("{tag}_blockmax.u8"), &block_max);
    let meta = format!(
        "{{\"n\":{n},\"h\":{h},\"w\":{w},\"median\":{MEDIAN},\"tophat\":{TOPHAT},\"edges\":{:?}}}",
        edges
    );
    write(dir, &format!("{tag}_meta.json"), meta.as_bytes());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (raw_dir, out) = (Path::new(&args[1]), Path::new(&args[2]));
    fs::create_dir_all(out).unwrap();
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let (mut h, mut w) = (0, 0);
    for (file, first, count) in PICKS {
        let mut rd = FileReader::open(&raw_dir.join(file)).unwrap();
        let idx = index(&mut rd).unwrap();
        (h, w) = (idx.height, idx.width);
        for i in first..first + count {
            let (off, sz) = idx.frames[i];
            let bytes = minime_core::avi::ReadAt::read_at(&mut rd, off, sz as usize).unwrap();
            frames.push(idx.unpad(&bytes));
        }
    }
    println!("{} frames of {h}x{w}", frames.len());
    emit(out, "full", &frames, h, w, 7);
    // registration image (band-pass) of the median-filtered first 64 frames, and the float32 Gaussian taps it uses
    let mut bp = Vec::new();
    for f in frames.iter().take(64) {
        let den = median_blur(f, h, w, MEDIAN);
        bp.extend(minime_core::motion::bandpass(&den, h, w, 1.0, 6.0));
    }
    write(out, "full_bandpass.f32", &f32_bytes(&bp));
    let shifts = make_shifts(frames.len(), 7);
    let shf: Vec<f32> = (0..64)
        .flat_map(|i| {
            minime_core::preprocess::shift_f32(
                &bp[i * h * w..(i + 1) * h * w],
                h,
                w,
                shifts[2 * i],
                shifts[2 * i + 1],
            )
        })
        .collect();
    write(out, "full_shift_f32.f32", &f32_bytes(&shf));
    let taps: Vec<f32> = gauss_taps(25, 3.0).iter().map(|&v| v as f32).collect();
    write(out, "bp_taps.f32", &f32_bytes(&taps));
    // odd sizes: a 301 x 487 crop of the first 24 frames of every other stretch
    let (oh, ow) = (301, 487);
    let crops: Vec<Vec<u8>> = frames
        .iter()
        .step_by(10)
        .take(24)
        .map(|f| {
            let mut c = Vec::with_capacity(oh * ow);
            for y in 0..oh {
                c.extend_from_slice(&f[(y + 13) * w + 29..(y + 13) * w + 29 + ow]);
            }
            c
        })
        .collect();
    emit(out, "odd", &crops, oh, ow, 11);
    println!("written to {}", out.display());
}
