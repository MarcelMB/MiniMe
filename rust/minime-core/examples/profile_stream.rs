//! Throwaway profile of the streaming pass at a given unit count, step by step, on real frames:
//!
//!     cargo run --release --example profile_stream -- INPUTS_DIR RAW_DIR [N_FRAMES] [K]
//!
//! `INPUTS_DIR` is a streaming pass's inputs (`params.json`, `A_*.npy`, `shifts.npy`, `ymean.npy`,
//! `sn.npy`); with `K` only the first K units are kept. Prints the model build (the two Jacobi
//! eigensolves for `cond`), the support build, and per frame: preprocessing, the least squares
//! and the fold of the sums, the fold split into its parts. Not production code.

use minime_core::avi::{self, FileReader, ReadAt};
use minime_core::linalg::{sym_eig, NormalSolver};
use minime_core::stream::{lane_ranges, Csr, FoldLane, StreamPass, Support, TraceModel};
use minime_core::Inputs;
use std::time::Instant;

fn frames(raw: &str, n: usize) -> Vec<Vec<u8>> {
    let mut avis: Vec<_> = std::fs::read_dir(raw)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "avi"))
        .collect();
    avis.sort_by_key(|p| {
        p.file_stem()
            .unwrap()
            .to_string_lossy()
            .parse::<u32>()
            .unwrap_or(0)
    });
    let mut r = FileReader::open(&avis[0]).unwrap();
    let ix = avi::index(&mut r).unwrap();
    (0..n.min(ix.frames.len()))
        .map(|i| {
            let (off, size) = ix.frames[i];
            ix.unpad(&r.read_at(off, size as usize).unwrap())
        })
        .collect()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, raw) = (&args[1], &args[2]);
    let n: usize = args.get(3).map_or(200, |v| v.parse().unwrap());
    let mut inp =
        Inputs::load(&mut |name| std::fs::read(format!("{dir}/{name}")).map_err(|e| e.to_string())).unwrap();
    if let Some(k) = args.get(4).map(|v| v.parse::<usize>().unwrap()) {
        let a = &inp.a;
        let end = a.indptr[k];
        inp.a = Csr {
            nrows: k,
            ncols: a.ncols,
            indptr: a.indptr[..=k].to_vec(),
            indices: a.indices[..end].to_vec(),
            data: a.data[..end].to_vec(),
        };
    }
    let (h, w) = (inp.params.h, inp.params.w);
    let k = inp.a.nrows;
    println!("K = {k}, A nnz = {}, P = {}", inp.a.data.len(), h * w);

    let t = Instant::now();
    let mut g = inp.a.gram();
    for i in 0..k {
        for j in i + 1..k {
            let m = 0.5 * (g[i * k + j] + g[j * k + i]);
            g[i * k + j] = m;
            g[j * k + i] = m;
        }
    }
    println!("gram A A^T: {:.1} ms", ms(t));
    let t = Instant::now();
    let _ = sym_eig(&g, k);
    println!("one Jacobi eigensolve (K x K): {:.0} ms", ms(t));
    let t = Instant::now();
    let full = NormalSolver::new(&g, k);
    println!(
        "NormalSolver::new (K x K): {:.0} ms, cond {:.3e}",
        ms(t),
        full.cond()
    );
    let t = Instant::now();
    let fast = NormalSolver::certified(&g, k);
    println!(
        "NormalSolver::certified (K x K): {:.0} ms, bound {:.3e}, cholesky {}",
        ms(t),
        fast.cond(),
        fast.is_cholesky()
    );
    let t = Instant::now();
    let model = TraceModel::build(&inp.a, &inp.ymean);
    println!(
        "TraceModel::build (two certified solvers): {:.0} ms, cholesky {}, [A; b] bound {:.3e}",
        ms(t),
        model.solver.is_cholesky(),
        model.solver.cond()
    );
    let mut gx = model.x.gram();
    for i in 0..=k {
        for j in i + 1..=k {
            let m = 0.5 * (gx[i * (k + 1) + j] + gx[j * (k + 1) + i]);
            gx[i * (k + 1) + j] = m;
            gx[j * (k + 1) + i] = m;
        }
    }
    let t = Instant::now();
    let jx = NormalSolver::new(&gx, k + 1);
    println!(
        "[A; b] Jacobi: {:.0} ms, cond {:.3e}, cholesky {}",
        ms(t),
        jx.cond(),
        jx.is_cholesky()
    );
    let t = Instant::now();
    let sup = Support::build(&inp.a, h, w, inp.params.dl_wnd);
    println!("Support::build: {:.0} ms, {} pairs", ms(t), sup.pix.len());
    let t = Instant::now();
    let mut pass = StreamPass::new(inp.params.clone(), &inp.a, &inp.ymean);
    println!("StreamPass::new (all of it): {:.0} ms", ms(t));

    let raw = frames(raw, n);
    let n = raw.len();
    let prep = inp.prep();
    let t = Instant::now();
    let ys: Vec<Vec<f32>> = raw
        .iter()
        .enumerate()
        .map(|(i, r)| {
            prep.frame(r, h, w, inp.shifts[2 * i], inp.shifts[2 * i + 1])
                .iter()
                .map(|&v| f32::from(v))
                .collect()
        })
        .collect();
    let t_prep = ms(t) / n as f64;
    let t = Instant::now();
    let mut sol = Vec::new();
    let mut ccs = Vec::with_capacity(n);
    let mut rhs = Vec::new();
    let mut yras: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut fs = Vec::with_capacity(n);
    for y in &ys {
        model.solve_rhs(y, &mut sol, &mut rhs);
        let cc: Vec<f64> = sol[..k].iter().map(|v| v.max(0.0)).collect();
        let mut yr = vec![0.0; k];
        model.yra(&rhs, &cc, sol[k], &mut yr);
        yras.push(yr);
        ccs.push(cc);
        fs.push(sol[k] as f32);
    }
    let t_solve = ms(t) / n as f64;
    let t = Instant::now();
    for i in 0..n {
        pass.push_solved(&ys[i], &ccs[i], &yras[i], fs[i]);
    }
    let t_fold = ms(t) / n as f64;
    let nz = ccs
        .iter()
        .map(|c| c.iter().filter(|&&v| v != 0.0).count())
        .sum::<usize>() as f64
        / n as f64;
    println!("per frame: preprocess {t_prep:.2} ms, solve {t_solve:.2} ms, fold {t_fold:.2} ms ({nz:.0} of {k} traces > 0)");

    // the fold in 8 lanes: each lane's time per frame
    let ranges = lane_ranges(k, h * w, &pass.support, 8);
    let mut worst = 0f64;
    for r in &ranges {
        let mut lane = FoldLane::new(&inp.params, &pass.support, k, *r);
        let t = Instant::now();
        for i in 0..n {
            lane.fold(&ys[i][r.pix[0]..r.pix[1]], &ccs[i]);
        }
        let per = ms(t) / n as f64;
        worst = worst.max(per);
        print!("{per:.2} ");
    }
    println!("ms per frame per lane (8 lanes; slowest {worst:.2})");

    // the fold's parts, as `StreamPass::fold` does them
    let p = h * w;
    let mut m = vec![0f64; k * k];
    let mut csum = vec![0f64; k];
    let t = Instant::now();
    for cc in &ccs {
        for i in 0..k {
            let ci = cc[i];
            if ci != 0.0 {
                for (mj, &cj) in m[i * k..(i + 1) * k].iter_mut().zip(cc) {
                    *mj += ci * cj;
                }
            }
            csum[i] += ci;
        }
    }
    let t_m = ms(t) / n as f64;
    let mut mlp = vec![0f64; k * k];
    let mut lsum = vec![0f64; k];
    let lp = ccs.clone();
    let t = Instant::now();
    for l in &lp {
        for i in 0..k {
            let li = l[i];
            for (mj, &lj) in mlp[i * k..(i + 1) * k].iter_mut().zip(l) {
                *mj += li * lj;
            }
            lsum[i] += li;
        }
    }
    let t_mlp = ms(t) / n as f64;
    let mut ysum = vec![0f64; p];
    let mut dsq = vec![0f64; p];
    let t = Instant::now();
    for (i, y) in ys.iter().enumerate() {
        for (s, &v) in ysum.iter_mut().zip(y) {
            *s += f64::from(v);
        }
        if i > 0 {
            for ((s, &v), &u) in dsq.iter_mut().zip(y).zip(&ys[i - 1]) {
                let d = f64::from(v - u);
                *s += d * d;
            }
        }
    }
    let t_y = ms(t) / n as f64;
    let mut l = vec![0f64; sup.pix.len()];
    let t = Instant::now();
    for (y, cc) in ys.iter().zip(&ccs) {
        for ((l, &p), &u) in l.iter_mut().zip(&sup.pix).zip(&sup.unit) {
            *l += f64::from(y[p]) * cc[u];
        }
    }
    let t_l = ms(t) / n as f64;
    println!(
        "fold parts per frame: c c^T {t_m:.2} ms, low-pass c c^T {t_mlp:.2} ms, y sums + dsq {t_y:.2} ms, y c over the pairs {t_l:.2} ms"
    );
    std::hint::black_box((m, mlp, ysum, dsq, l, csum, lsum));
}
