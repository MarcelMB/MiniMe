//! A/B micro-benchmark of the per-frame kernels, old (scalar reference) against new, run
//! alternately in one process so the machine's state cancels; prints the median of `ROUNDS`
//! rounds in ms per call.
//!
//!     cargo run --release --example bench_prep [RAW_DIR]
//!
//! With a `RAW_DIR` of AVIs the frames are real (16 frames spread over the first segment),
//! otherwise 600 x 600 noise.

use minime_core::avi::{self, FileReader, ReadAt};
use minime_core::filters::*;
use minime_core::preprocess::*;
use std::time::Instant;

const ROUNDS: usize = 7;

fn frames(raw: Option<&str>, h: usize, w: usize, n: usize) -> Vec<Vec<u8>> {
    let Some(raw) = raw else {
        let mut s = 1u64;
        return (0..n)
            .map(|_| {
                (0..h * w)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((s >> 58) as u8) + 60
                    })
                    .collect()
            })
            .collect();
    };
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
            let (off, size) = ix.frames[i * 61 % ix.frames.len()];
            ix.unpad(&r.read_at(off, size as usize).unwrap())
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Median ms per call of `old` and of `new`, alternating rounds of one pass over the frames.
fn ab<T, A, B>(name: &str, fr: &[Vec<T>], mut old: impl FnMut(&[T]) -> A, mut new: impl FnMut(&[T]) -> B) {
    let (mut to, mut tn) = (Vec::new(), Vec::new());
    for round in 0..=ROUNDS {
        let t = Instant::now();
        for f in fr {
            std::hint::black_box(old(f));
        }
        let o = t.elapsed().as_secs_f64() * 1e3 / fr.len() as f64;
        let t = Instant::now();
        for f in fr {
            std::hint::black_box(new(f));
        }
        let n = t.elapsed().as_secs_f64() * 1e3 / fr.len() as f64;
        if round > 0 {
            to.push(o);
            tn.push(n);
        }
    }
    let (o, n) = (median(to), median(tn));
    println!("{name:<24} old {o:7.3} ms   new {n:7.3} ms   {:5.2}x", o / n);
}

fn main() {
    let raw = std::env::args().nth(1);
    let (h, w) = (600, 600);
    let fr = frames(raw.as_deref(), h, w, 16);
    println!(
        "{} frames {h}x{w} ({})",
        fr.len(),
        if raw.is_some() { "real" } else { "noise" }
    );
    let (dy, dx) = (0.37f32, -1.62f32);
    let to_f32 = |f: &[u8]| -> Vec<f32> { f.iter().map(|&v| f32::from(v)).collect() };
    let den: Vec<Vec<u8>> = fr.iter().map(|f| median_blur(f, h, w, 7)).collect();
    let half: Vec<Vec<f32>> = fr.iter().map(|f| area_down2_f32(&to_f32(f), h, w)).collect();
    let g: Vec<Vec<f32>> = half.iter().map(|f| gauss_f32(f, h / 2, w / 2, 3.0)).collect();
    let reg: Vec<Vec<f32>> = den
        .iter()
        .zip(&g)
        .map(|(d, g)| {
            gauss_u8(d, h, w, 1.0)
                .iter()
                .zip(resize_up2_f32(g, h / 2, w / 2))
                .map(|(&n, b)| f32::from(n) - b)
                .collect()
        })
        .collect();
    ab(
        "median 7",
        &fr,
        |f| median_blur_huang(f, h, w, 7),
        |f| median_blur(f, h, w, 7),
    );
    ab(
        "tophat 15",
        &den,
        |f| tophat_scalar(f, h, w, 15),
        |f| tophat(f, h, w, 15),
    );
    ab(
        "gauss_u8 s=1",
        &den,
        |f| gauss_u8_scalar(f, h, w, 1.0),
        |f| gauss_u8(f, h, w, 1.0),
    );
    ab(
        "gauss_f32 300^2 s=3",
        &half,
        |f| gauss_f32_scalar(f, h / 2, w / 2, 3.0),
        |f| gauss_f32(f, h / 2, w / 2, 3.0),
    );
    ab(
        "resize_up2_f32",
        &g,
        |f| resize_up2_f32_scalar(f, h / 2, w / 2),
        |f| resize_up2_f32(f, h / 2, w / 2),
    );
    ab(
        "shift u8",
        &den,
        |f| shift_scalar(f, h, w, dy, dx),
        |f| shift(f, h, w, dy, dx),
    );
    ab(
        "shift f32",
        &reg,
        |f| shift_f32_scalar(f, h, w, dy, dx),
        |f| shift_f32(f, h, w, dy, dx),
    );
}
