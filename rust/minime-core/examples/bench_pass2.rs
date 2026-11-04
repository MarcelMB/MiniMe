//! A/B timing and equivalence check of pass 2 on a tile store, old code against new, one
//! tile at a time and alternating, with the per-step times of `prof`:
//!
//!     cargo run --release --features profile,reference --example bench_pass2 -- \
//!         --tiles TILE_DIR --inputs INPUTS_DIR [--n 100] [--reps 1] [--only ref|new]
//!
//! `TILE_DIR` is pass 1's tile store (`tile_NNN.partPP`), `INPUTS_DIR` holds `pass2.json` and
//! `omega_NNN.npy` (`ref_pass2.py`). The Gaussian draws are replayed, so both implementations
//! see the same test matrices.

use minime_core::compress::{self, Pass2Params, Recorded, TileFactors};
use minime_core::{compress_ref, npy, pass1, prof};
use std::path::PathBuf;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn load_tile(dir: &std::path::Path, k: usize) -> Vec<u8> {
    let mut raw = Vec::new();
    for part in 0.. {
        match std::fs::read(dir.join(format!("tile_{k:03}.part{part:02}"))) {
            Ok(b) => raw.extend_from_slice(&b),
            Err(_) if part > 0 => break,
            Err(e) => panic!("tile {k}: {e}"),
        }
    }
    raw
}

fn rel_fro(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    let den: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

/// `W V^T` of a tile, `(P, T)` row-major.
fn recon(tf: &TileFactors, p: usize, t: usize) -> Vec<f32> {
    let r = tf.rank;
    let mut out = vec![0f32; p * t];
    for px in 0..p {
        for ti in 0..t {
            let mut s = 0f64;
            for c in 0..r {
                s += f64::from(tf.w[px * r + c]) * f64::from(tf.v[ti * r + c]);
            }
            out[px * t + ti] = s as f32;
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tiles = PathBuf::from(arg(&args, "--tiles").expect("--tiles DIR"));
    let inputs = PathBuf::from(arg(&args, "--inputs").expect("--inputs DIR"));
    let reps: usize = arg(&args, "--reps").map_or(1, |v| v.parse().unwrap());
    let only = arg(&args, "--only");
    let p: Pass2Params = serde_json::from_slice(&std::fs::read(inputs.join("pass2.json")).unwrap()).unwrap();
    let grid = pass1::tile_grid(p.h, p.w, p.tile);
    let n: usize = arg(&args, "--n").map_or(grid.len(), |v| v.parse().unwrap());
    let ids: Vec<usize> = match arg(&args, "--ids") {
        Some(s) => s.split(',').map(|v| v.parse().unwrap()).collect(),
        None => (0..n.min(grid.len())).collect(),
    };
    let impls: Vec<&str> = match only.as_deref() {
        Some("ref") => vec!["ref"],
        Some("new") => vec!["new"],
        _ => vec!["ref", "new"],
    };
    let mut total = vec![0f64; 2];
    let mut steps: Vec<std::collections::BTreeMap<&'static str, f64>> = vec![Default::default(); 2];
    let mut sigma_rel = 0f64;
    let mut worst = (0f64, 0f64, 0usize, 0usize); // recon diff, w diff, rank mismatches
    for rep in 0..reps {
        for &k in &ids {
            let raw = load_tile(&tiles, k);
            let px = grid[k].pixels();
            let omega = npy::read_f32(&std::fs::read(inputs.join(format!("omega_{k:03}.npy"))).unwrap())
                .unwrap()
                .data;
            let mut out: Vec<Option<TileFactors>> = vec![None, None];
            // alternate which runs first
            let mut order: Vec<&str> = impls.clone();
            if (k + rep) % 2 == 1 {
                order.reverse();
            }
            for which in order {
                let mut rng = Recorded::new(omega.clone());
                let _ = prof::take();
                let t0 = Instant::now();
                let tf = match which {
                    "ref" => compress_ref::compress_tile(&raw, p.t, px, &p, &mut rng).unwrap(),
                    _ => compress::compress_tile(&raw, p.t, px, &p, &mut rng).unwrap(),
                };
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                let i = usize::from(which == "new");
                total[i] += dt;
                for (name, ms) in prof::take() {
                    *steps[i].entry(name).or_default() += ms;
                }
                out[i] = Some(tf);
            }
            if let (Some(a), Some(b)) = (&out[0], &out[1]) {
                if a.rank != b.rank || a.k != b.k {
                    worst.2 += 1;
                    eprintln!("tile {k}: rank {} vs {} (k {} vs {})", a.rank, b.rank, a.k, b.k);
                }
                if a.mean != b.mean || a.energy != b.energy {
                    worst.3 += 1;
                    eprintln!("tile {k}: mean/energy differ");
                }
                for (x, y) in a.sigma.iter().zip(&b.sigma) {
                    sigma_rel = sigma_rel.max(f64::from((x - y).abs() / x.abs()));
                }
                if a.rank == b.rank {
                    let ra = recon(a, px, p.t);
                    let rb = recon(b, px, p.t);
                    worst.0 = worst.0.max(rel_fro(&ra, &rb));
                    worst.1 = worst.1.max(rel_fro(&a.w, &b.w));
                }
            }
        }
    }
    let r = reps as f64;
    println!(
        "tiles {} reps {} (wall ms per pass over the tiles)",
        ids.len(),
        reps
    );
    println!("{:<14}{:>12}{:>12}{:>9}", "step", "ref", "new", "x");
    let mut names: Vec<&str> = steps[0].keys().chain(steps[1].keys()).copied().collect();
    names.sort_unstable();
    names.dedup();
    for name in names {
        let a = steps[0].get(name).copied().unwrap_or(0.0) / r;
        let b = steps[1].get(name).copied().unwrap_or(0.0) / r;
        println!(
            "{:<14}{:>12.1}{:>12.1}{:>9.2}",
            name,
            a,
            b,
            if b > 0.0 { a / b } else { f64::NAN }
        );
    }
    println!(
        "{:<14}{:>12.1}{:>12.1}{:>9.2}",
        "TOTAL",
        total[0] / r,
        total[1] / r,
        total[0] / total[1]
    );
    if impls.len() == 2 {
        println!(
            "rank/k mismatches {}, mean/energy mismatches {}, max sigma rel diff {sigma_rel:.3e}, max recon rel diff {:.3e}, max W rel diff {:.3e}",
            worst.2, worst.3, worst.0, worst.1
        );
    }
}
