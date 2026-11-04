//! Native wgpu (Metal) check and speed of the preprocessing chain on the reference frames.
//! cargo run --release --features gpu --example native -- <dir with full_frames.u8>
use gpu_spike::{cpu_chain_into, gpu::Gpu};
use std::time::Instant;

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let rd = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap();
    let (frames, want, shifts) = (
        rd("full_frames.u8"),
        rd("full_final.u8"),
        f32s(&rd("full_shifts.f32")),
    );
    let (h, w, n) = (600usize, 600usize, 256usize);
    let hw = h * w;
    let b = 64;
    let t0 = Instant::now();
    let gpu = pollster::block_on(Gpu::new(h, w, b)).unwrap();
    println!(
        "{} | init + pipelines {:.0} ms",
        gpu.info,
        t0.elapsed().as_secs_f64() * 1e3
    );
    let mut got = vec![0u8; n * hw];
    for f0 in (0..n).step_by(b) {
        let (fr, sh) = (&frames[f0 * hw..(f0 + b) * hw], &shifts[2 * f0..2 * (f0 + b)]);
        pollster::block_on(gpu.run(fr, sh, b, &mut got[f0 * hw..(f0 + b) * hw])).unwrap();
    }
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!(
        "native wgpu chain vs CPU reference: {} of {} bytes differ",
        bad,
        got.len()
    );

    let mut sink = vec![0u8; b * hw];
    let mut ts = vec![];
    for k in 0..8 {
        let f0 = (k % 4) * b;
        let t = Instant::now();
        pollster::block_on(gpu.run(
            &frames[f0 * hw..(f0 + b) * hw],
            &shifts[2 * f0..2 * (f0 + b)],
            b,
            &mut sink,
        ))
        .unwrap();
        if k >= 3 {
            ts.push(t.elapsed().as_secs_f64() * 1e3 / b as f64);
        }
    }
    ts.sort_by(f64::total_cmp);
    println!(
        "native wgpu sequential B={b}: {:.3} ms/frame (median of {})",
        ts[ts.len() / 2],
        ts.len()
    );

    let mut ts = vec![];
    // phases of one batch: submit (upload + encode), GPU compute, readback
    let (mut tu, mut tc, mut tr) = (vec![], vec![], vec![]);
    for k in 0..8 {
        let f0 = (k % 4) * b;
        let t = Instant::now();
        gpu.submit(&frames[f0 * hw..(f0 + b) * hw], &shifts[2 * f0..2 * (f0 + b)], b);
        let a = t.elapsed().as_secs_f64();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let c = t.elapsed().as_secs_f64();
        pollster::block_on(gpu.read(b, &mut sink)).unwrap();
        let r = t.elapsed().as_secs_f64();
        if k >= 3 {
            tu.push(a * 1e3 / b as f64);
            tc.push((c - a) * 1e3 / b as f64);
            tr.push((r - c) * 1e3 / b as f64);
        }
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "native wgpu phases per frame: submit(upload+encode) {:.3} ms, GPU wait {:.3} ms, readback {:.3} ms",
        med(tu),
        med(tc),
        med(tr)
    );
    let mut out = vec![0u8; b * hw];
    for k in 0..4 {
        let f0 = (k % 4) * b;
        let t = Instant::now();
        cpu_chain_into(
            &frames[f0 * hw..(f0 + b) * hw],
            &shifts[2 * f0..2 * (f0 + b)],
            b,
            h,
            w,
            &mut out,
        );
        ts.push(t.elapsed().as_secs_f64() * 1e3 / b as f64);
    }
    ts.sort_by(f64::total_cmp);
    println!(
        "native CPU 1 thread chain B={b}: {:.3} ms/frame (median of {})",
        ts[ts.len() / 2],
        ts.len()
    );
}
