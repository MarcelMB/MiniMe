//! Pass 2 of one tile with swappable products: `compress.rs`'s randomized SVD and rank cut, with
//! `Y Z` / `Y^T Q` done on the CPU (`matrixmultiply`, the app's code) or on the GPU (`wgpu`).

use crate::compress::*;
use crate::linalg::sym_eig;

pub enum Products<'a> {
    Cpu(&'a TileMatrix<'a>),
    #[cfg(feature = "gpu")]
    Gpu(&'a crate::gpu::GpuProducts),
}

impl Products<'_> {
    /// `Y Z`: `(P, ell)` from `z` `(T, ell)`.
    pub async fn matmul(&self, z: &[f32], ell: usize) -> Result<Vec<f32>, String> {
        match self {
            Products::Cpu(y) => Ok(y.matmul(z, ell)),
            #[cfg(feature = "gpu")]
            Products::Gpu(g) => g.product(true, z, ell).await,
        }
    }

    /// `Y^T Q`: `(T, ell)` from `q` `(P, ell)`.
    pub async fn t_matmul(&self, q: &[f32], ell: usize) -> Result<Vec<f32>, String> {
        match self {
            Products::Cpu(y) => Ok(y.t_matmul(q, ell)),
            #[cfg(feature = "gpu")]
            Products::Gpu(g) => g.product(false, q, ell).await,
        }
    }
}

/// `compress::rsvd_streamed` with the products behind `Products`.
pub async fn rsvd(
    prod: &Products<'_>,
    p: usize,
    t: usize,
    k: usize,
    n_iter: usize,
    rng: &mut dyn Normals,
) -> Result<Svd, String> {
    let ell = k + N_OVERSAMPLES;
    let omega = rng.draw(t * ell)?;
    let mut q = qr_q(&prod.matmul(&omega, ell).await?, p, ell);
    for _ in 0..n_iter {
        let z = qr_q(&prod.t_matmul(&q, ell).await?, t, ell);
        q = qr_q(&prod.matmul(&z, ell).await?, p, ell);
    }
    let bt = prod.t_matmul(&q, ell).await?;
    let mut g = vec![0f64; ell * ell];
    for row in bt.chunks_exact(ell) {
        for a in 0..ell {
            let ra = f64::from(row[a]);
            for b in a..ell {
                g[a * ell + b] += ra * f64::from(row[b]);
            }
        }
    }
    for a in 0..ell {
        for b in 0..a {
            g[a * ell + b] = g[b * ell + a];
        }
    }
    let (evals, evecs) = sym_eig(&g, ell);
    let order: Vec<usize> = (0..ell).rev().take(k).collect();
    let s: Vec<f64> = order.iter().map(|&j| evals[j].max(0.0).sqrt()).collect();
    let mut vt = vec![0f32; k * t];
    for (i, &j) in order.iter().enumerate() {
        if s[i] == 0.0 {
            continue;
        }
        for (ti, row) in bt.chunks_exact(ell).enumerate() {
            let d: f64 = (0..ell).map(|a| f64::from(row[a]) * evecs[a * ell + j]).sum();
            vt[i * t + ti] = (d / s[i]) as f32;
        }
    }
    let mut u = vec![0f32; p * k];
    for px in 0..p {
        let qr = &q[px * ell..(px + 1) * ell];
        for (i, &j) in order.iter().enumerate() {
            let d: f64 = (0..ell).map(|a| f64::from(qr[a]) * evecs[a * ell + j]).sum();
            u[px * k + i] = d as f32;
        }
    }
    Ok(Svd { k, u, s, vt })
}

/// Kept factors of `compress::compress_tile` without the noise estimate: `(W (P, r), V (T, r), s, k)`.
pub struct Kept {
    pub w: Vec<f32>,
    pub v: Vec<f32>,
    pub s: Vec<f64>,
    pub k: usize,
    pub rank: usize,
}

pub async fn compress(
    prod: &Products<'_>,
    p: usize,
    t: usize,
    pr: &Pass2Params,
    rng: &mut dyn Normals,
) -> Result<Kept, String> {
    let k_cap = MAX_RANK.min(p.saturating_sub(1)).min(t.saturating_sub(1)).max(1);
    let mut k = pr.svd_k0.min(k_cap);
    let (svd, mut keep) = loop {
        let svd = rsvd(prod, p, t, k, pr.svd_power_iter, rng).await?;
        let keep = autocorr_keep(&svd.vt, t, pr.rank_z, pr.rank_rho_min);
        if k >= k_cap || !keep[k.saturating_sub(ADAPT_MARGIN)..].iter().any(|&b| b) {
            break (svd, keep);
        }
        k = (2 * k).min(k_cap);
    };
    if !keep.iter().any(|&b| b) {
        keep[0] = true;
    }
    let idx: Vec<usize> = (0..k).filter(|&i| keep[i]).collect();
    let r = idx.len();
    let mut w = vec![0f32; p * r];
    for px in 0..p {
        for (c, &i) in idx.iter().enumerate() {
            w[px * r + c] = (f64::from(svd.u[px * k + i]) * svd.s[i]) as f32;
        }
    }
    let mut v = vec![0f32; t * r];
    for ti in 0..t {
        for (c, &i) in idx.iter().enumerate() {
            v[ti * r + c] = svd.vt[i * t + ti];
        }
    }
    Ok(Kept {
        w,
        v,
        s: idx.iter().map(|&i| svd.s[i]).collect(),
        k,
        rank: r,
    })
}

pub fn params(t: usize, p: usize) -> Pass2Params {
    Pass2Params {
        h: 64,
        w: p / 64,
        t,
        tile: 64,
        svd_k0: 40,
        svd_power_iter: 1,
        rank_rho_min: 0.2,
        rank_z: 4.0,
        noise_freq: 0.06,
        noise_frames: 0,
    }
}

/// Overlap of the column spaces of two `(n, r)` and `(n, r2)` matrices with orthonormal-ish columns:
/// `||A^T B||_F^2 / min(r, r2)` after normalising the columns (1 = same subspace).
pub fn overlap(a: &[f32], ra: usize, b: &[f32], rb: usize, n: usize) -> f64 {
    let norm = |m: &[f32], r: usize| -> Vec<f64> {
        let mut out: Vec<f64> = m.iter().map(|&v| f64::from(v)).collect();
        for c in 0..r {
            let nn: f64 = (0..n).map(|i| out[i * r + c].powi(2)).sum::<f64>().sqrt();
            for i in 0..n {
                out[i * r + c] /= nn.max(1e-30);
            }
        }
        out
    };
    let (a, b) = (norm(a, ra), norm(b, rb));
    let mut f = 0.0;
    for i in 0..ra {
        for j in 0..rb {
            let d: f64 = (0..n).map(|x| a[x * ra + i] * b[x * rb + j]).sum();
            f += d * d;
        }
    }
    f / ra.min(rb) as f64
}

/// Compare CPU and GPU products on one tile: ranks, singular values, subspaces.
#[cfg(feature = "gpu")]
pub async fn compare(
    raw: &[u8],
    t: usize,
    p: usize,
    gpu: &crate::gpu::GpuProducts,
) -> Result<String, String> {
    let y = TileMatrix::new(raw, t, p);
    let pr = params(t, p);
    let cpu = compress(&Products::Cpu(&y), p, t, &pr, &mut Gaussian::new(5)).await?;
    let gp = compress(&Products::Gpu(gpu), p, t, &pr, &mut Gaussian::new(5)).await?;
    let m = cpu.rank.min(gp.rank);
    let s_rel = (0..m)
        .map(|i| ((cpu.s[i] - gp.s[i]) / cpu.s[i]).abs())
        .fold(0.0, f64::max);
    let ov_u = overlap(&cpu.w, cpu.rank, &gp.w, gp.rank, p);
    let ov_v = overlap(&cpu.v, cpu.rank, &gp.v, gp.rank, t);
    Ok(format!(
        "{{\"rank_cpu\":{},\"rank_gpu\":{},\"k_cpu\":{},\"k_gpu\":{},\"s_max_rel_diff\":{:.3e},\"overlap_U\":{:.9},\"overlap_V\":{:.9}}}",
        cpu.rank, gp.rank, cpu.k, gp.k, s_rel, ov_u, ov_v
    ))
}
