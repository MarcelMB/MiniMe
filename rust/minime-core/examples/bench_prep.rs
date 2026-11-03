use minime_core::preprocess::*;
use std::time::Instant;
fn main() {
    let (h, w) = (600, 600);
    let mut s = 1u64;
    let f: Vec<u8> = (0..h * w)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 58) as u8) + 60
        })
        .collect();
    let n = 50;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(median_blur(&f, h, w, 7));
    }
    println!("median {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / n as f64);
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(shift(&f, h, w, 0.3, -1.7));
    }
    println!("shift {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / n as f64);
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(tophat(&f, h, w, 15));
    }
    println!("tophat {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / n as f64);
}
