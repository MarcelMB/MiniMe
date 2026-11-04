//! Pass 2 on real tiles natively: CPU (matrixmultiply) vs wgpu (Metal) products, the whole randomized SVD
//! and rank cut, and where the CPU time goes.
//! cargo run --release --features gpu --example p2_native -- <dir with tiles/tile_*.u8>
use gpu_spike::compress::{Gaussian, Normals, TileMatrix};
use gpu_spike::gpu::{init_device, GpuProducts};
use gpu_spike::p2;
use std::time::Instant;

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let (t, p) = (3000usize, 4096usize);
    let (device, queue, info) = pollster::block_on(init_device()).unwrap();
    println!("{info}");
    for id in [11, 25, 34, 44, 55, 66, 77, 88] {
        let raw = std::fs::read(format!("{dir}/tiles/tile_{id}.u8")).unwrap();
        let y = TileMatrix::new(&raw, t, p);
        let g = GpuProducts::new(&device, &queue, &raw, t, p, &y.mean);
        let r = pollster::block_on(p2::compare(&raw, t, p, &g)).unwrap();
        println!("tile {id}: {r}");
        if id == 11 {
            // where the CPU time goes (1 thread, native)
            let t0 = Instant::now();
            std::hint::black_box(y.noise(0.06, 0));
            let noise = t0.elapsed().as_secs_f64();
            let mut rng = Gaussian::new(1);
            let z = rng.draw(t * 50).unwrap();
            let t0 = Instant::now();
            let a = y.matmul(&z, 50);
            let b = y.t_matmul(&a, 50);
            let c = y.matmul(&b, 50);
            let d = y.t_matmul(&c, 50);
            std::hint::black_box(d);
            let prod = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let pr = p2::params(t, p);
            let _ = pollster::block_on(p2::compress(
                &p2::Products::Cpu(&y),
                p,
                t,
                &pr,
                &mut Gaussian::new(5),
            ))
            .unwrap();
            let whole = t0.elapsed().as_secs_f64();
            println!("  native CPU 1 thread: noise {:.0} ms, 4 products (ell=50) {:.0} ms, whole rSVD+rank cut {:.0} ms", noise * 1e3, prod * 1e3, whole * 1e3);
            let t0 = Instant::now();
            let _ = pollster::block_on(p2::compress(
                &p2::Products::Gpu(&g),
                p,
                t,
                &pr,
                &mut Gaussian::new(5),
            ))
            .unwrap();
            println!(
                "  native wgpu products: whole rSVD+rank cut {:.0} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
