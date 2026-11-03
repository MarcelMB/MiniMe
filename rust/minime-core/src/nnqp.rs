//! Gram-shared non-negative coordinate descent for the CNMF spatial update, a line-by-line
//! port of `minime.spatial_kernel` (per pixel: `min_{w>=0} 1/2 w'G_s w - b_s'w +
//! lam sum(w)`, stopped on a certified relative duality gap).

use crate::linalg::sym_eig;

pub const REL_GAP_TOL: f64 = 1e-10;
pub const MAX_SWEEPS: usize = 100_000;
pub const EIG_RTOL: f64 = 1e-12;

/// `bs' Gs^+ bs`, tiny eigenvalues treated as zero.
fn projected_energy(gs: &[f64], bs: &[f64]) -> f64 {
    let n = bs.len();
    let (evals, evecs) = sym_eig(gs, n);
    let cut = EIG_RTOL * evals[n - 1];
    let mut q = 0.0;
    for (j, &lj) in evals.iter().enumerate() {
        if lj > cut && lj > 0.0 {
            let mut u = 0.0;
            for a in 0..n {
                u += evecs[a * n + j] * bs[a];
            }
            q += u * u / lj;
        }
    }
    q
}

fn cd_pixel(gs: &[f64], bs: &[f64], yn2: f64, lam: f64, w: &mut [f64], gw: &mut [f64]) {
    let n = bs.len();
    w.fill(0.0);
    gw.fill(0.0);
    for _ in 0..MAX_SWEEPS {
        let mut max_delta = 0.0f64;
        let mut max_w = 0.0f64;
        for k in 0..n {
            let gkk = gs[k * n + k];
            let old = w[k];
            let new = if gkk <= 0.0 {
                0.0
            } else {
                let v = (bs[k] - (gw[k] - gkk * old) - lam) / gkk;
                if v < 0.0 {
                    0.0
                } else {
                    v
                }
            };
            let d = new - old;
            if d != 0.0 {
                w[k] = new;
                for j in 0..n {
                    gw[j] += d * gs[j * n + k];
                }
                let ad = d.abs();
                if ad > max_delta {
                    max_delta = ad;
                }
            }
            if new > max_w {
                max_w = new;
            }
        }
        let (mut bw, mut wgw, mut sw, mut cmax) = (0.0, 0.0, 0.0, -1e300f64);
        for j in 0..n {
            bw += bs[j] * w[j];
            wgw += w[j] * gw[j];
            sw += w[j];
            let c = bs[j] - gw[j];
            if c > cmax {
                cmax = c;
            }
        }
        let mut rr = yn2 - 2.0 * bw + wgw;
        if rr < 0.0 {
            rr = 0.0;
        }
        let primal = 0.5 * rr + lam * sw;
        let s = if cmax > lam { lam / cmax } else { 1.0 };
        let dual = s * (yn2 - bw) - 0.5 * s * s * rr;
        let gap = primal - dual;
        let scale = if primal > 1e-300 { primal } else { 1e-300 };
        if gap <= REL_GAP_TOL * scale {
            break;
        }
        if max_delta <= 1e-15 * max_w {
            break;
        }
    }
}

/// Every active pixel's QP. `g` is `(k, k)`; `bvals[j]` is `B[pixel, unit]` for pair `j`
/// (pairs grouped by pixel through `indptr`, units in `units`); returns the solution per
/// pair.
pub fn solve_nnqp(
    g: &[f64],
    k: usize,
    bvals: &[f64],
    lam: &[f64],
    indptr: &[usize],
    units: &[usize],
) -> Vec<f64> {
    let mut values = vec![0.0; units.len()];
    for p in 0..indptr.len().saturating_sub(1) {
        let (s, e) = (indptr[p], indptr[p + 1]);
        let n = e - s;
        if n == 0 {
            continue;
        }
        let mut gs = vec![0.0; n * n];
        for a in 0..n {
            for c in 0..n {
                gs[a * n + c] = g[units[s + a] * k + units[s + c]];
            }
        }
        let bs = &bvals[s..e];
        let yn2 = projected_energy(&gs, bs);
        let mut w = vec![0.0; n];
        let mut gw = vec![0.0; n];
        cd_pixel(&gs, bs, yn2, lam[p], &mut w, &mut gw);
        values[s..e].copy_from_slice(&w);
    }
    values
}
