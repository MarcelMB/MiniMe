//! `minime.cnmf.run_cnmf` on one window's factors: the initial background, then `n_iter` x
//! (spatial update on the dilated supports with the NNQP kernel, size filter, rows scaled
//! to maximum 1; background; residual traces `YrA`; loop merge), as `pipeline.run_window2`
//! runs it after init.
//!
//! The one float32 product that decides the supports, `YCt` (`B` of the NNQP), follows
//! Accelerate's sgemm order: `V_b^T Ct` as sequential float32 FMAs over the frames, then
//! `W_b` times that as sequential FMAs over the components (bit-exact against numpy on
//! Apple silicon). Everything else is float64 as in Python; the sparse products follow
//! scipy's orders (`csr_matvec`, `csc_matvecs`, `csr_matmat`, which drops zero sums).

use crate::corr::TracesCorr;
use crate::factors::Factors;
use crate::gmm::pairwise_sum;
use crate::init::{centroids, gram, group_mean_a, group_mean_c, merge_labels, Csr};
use crate::nnqp::solve_nnqp;
use crate::npy;
use crate::stream::{self, Support};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct CnmfParams {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub tile: usize,
    pub noise_freq: f64,
    pub dl_wnd: usize,
    pub sparse_snr: f64,
    pub size_thres: [Option<f64>; 2],
    pub n_iter: usize,
    pub background_clip: bool,
    pub loop_merge_corr: f64,
    pub merge_min_dist: f64,
}

/// Frames per chunk of the clipped background (`update_background`'s `chunk`).
pub const BG_CHUNK: usize = 1000;
/// `cnmf.F64_CHECK_RTOL`.
pub const F64_CHECK_RTOL: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Spatial update
// ---------------------------------------------------------------------------

/// `cnmf.dilate_footprints` by pixel: active pixels and their allowed units, ascending.
pub fn dilate(a: &Csr, h: usize, w: usize, dl_wnd: usize) -> Support {
    let a64 = stream::Csr {
        nrows: a.k,
        ncols: a.p,
        indptr: a.indptr.clone(),
        indices: a.indices.clone(),
        data: a.data.iter().map(|&v| f64::from(v)).collect(),
    };
    Support::build(&a64, h, w, dl_wnd)
}

/// The rows of `c` (`(k, t)` float32) centred and scaled to unit norm (`column_norms`),
/// float64, and the norms.
fn unit_traces(c: &[f32], k: usize, t: usize) -> (Vec<f64>, Vec<f64>) {
    let mut cc: Vec<f64> = c.iter().map(|&v| f64::from(v)).collect();
    let mut scale = vec![1f64; k];
    for (row, s) in cc.chunks_exact_mut(t.max(1)).zip(&mut scale) {
        let mean = pairwise_sum(row) / t as f64;
        row.iter_mut().for_each(|x| *x -= mean);
        let sq: Vec<f64> = row.iter().map(|x| x * x).collect();
        let nrm = pairwise_sum(&sq).sqrt();
        if nrm != 0.0 {
            *s = nrm;
        }
        row.iter_mut().for_each(|x| *x /= *s);
    }
    (cc, scale)
}

/// `Cc Cc^T` (`(k, k)`, symmetric as numpy's syrk).
fn gram_rows(cc: &[f64], k: usize, t: usize) -> Vec<f64> {
    let mut g = vec![0f64; k * k];
    for i in 0..k {
        let a = &cc[i * t..(i + 1) * t];
        for j in i..k {
            let b = &cc[j * t..(j + 1) * t];
            let s = a.iter().zip(b).fold(0.0, |s, (&x, &y)| x.mul_add(y, s));
            g[i * k + j] = s;
            g[j * k + i] = s;
        }
    }
    g
}

/// `Factors.YCt(Cc)` at the support pairs (float32, as float64): per tile, `V_b^T Ct` for
/// the units the tile's pairs need (sequential float32 FMAs over the frames), then each
/// pair's `W_b[loc] . that` (sequential FMAs over the components).
fn y_ct_pairs(f: &Factors, cc: &[f64], k: usize, sup: &Support) -> Vec<f64> {
    let t = f.t;
    let ct: Vec<f32> = cc.iter().map(|&v| v as f32).collect(); // (k, t)
                                                               // pairs grouped by tile
    let mut by_tile: Vec<Vec<usize>> = vec![Vec::new(); f.tiles.len()];
    for (j, &px) in sup.pix.iter().enumerate() {
        by_tile[f.locate(px).0].push(j);
    }
    let mut out = vec![0f64; sup.pix.len()];
    let mut col = vec![usize::MAX; k];
    for (b, pairs) in by_tile.iter().enumerate() {
        if pairs.is_empty() {
            continue;
        }
        let tf = &f.tiles[b];
        let r = tf.rank;
        let mut units: Vec<usize> = pairs.iter().map(|&j| sup.unit[j]).collect();
        units.sort_unstable();
        units.dedup();
        // tmp[c * r + q] = sum_t V[t, q] Ct[t, units[c]]
        let mut tmp = vec![0f32; units.len() * r];
        for (c, &u) in units.iter().enumerate() {
            col[u] = c;
            let x = &ct[u * t..(u + 1) * t];
            let acc = &mut tmp[c * r..(c + 1) * r];
            for (vrow, &xv) in tf.v.chunks_exact(r.max(1)).zip(x) {
                for (a, &v) in acc.iter_mut().zip(vrow) {
                    *a = v.mul_add(xv, *a);
                }
            }
        }
        for &j in pairs {
            let l = f.locate(sup.pix[j]).1;
            let wrow = &tf.w[l * r..(l + 1) * r];
            let tc = &tmp[col[sup.unit[j]] * r..(col[sup.unit[j]] + 1) * r];
            let mut acc = 0f32;
            for (&wv, &tv) in wrow.iter().zip(tc) {
                acc = wv.mul_add(tv, acc);
            }
            out[j] = f64::from(acc);
        }
        for &u in &units {
            col[u] = usize::MAX;
        }
    }
    out
}

/// `cnmf._size_mask`.
fn size_ok(nnz: usize, thres: [Option<f64>; 2]) -> bool {
    let n = nnz as f64;
    nnz > 0 && thres[0].is_none_or(|lo| lo == 0.0 || n > lo) && thres[1].is_none_or(|hi| hi == 0.0 || n < hi)
}

pub struct Spatial {
    pub sup: Support,
    pub b: Vec<f64>,   // YCt per pair
    pub lam: Vec<f64>, // per active pixel
    pub w: Vec<f64>,   // NNQP solution per pair (unit-norm traces)
    pub nnz: Vec<usize>,
    pub mask: Vec<bool>,
    pub norm_fac: Vec<f32>,
    pub a: Csr,
}

/// `cnmf.update_spatial`.
pub fn update_spatial(
    f: &Factors,
    a: &Csr,
    c: &[f32],
    sn: &[f32],
    p: &CnmfParams,
) -> Result<Spatial, String> {
    let (k, t) = (a.k, f.t);
    let sup = dilate(a, f.h, f.w, p.dl_wnd);
    let (cc, scale) = unit_traces(c, k, t);
    let g = gram_rows(&cc, k, t);
    let b = y_ct_pairs(f, &cc, k, &sup);
    let pen = (t as f64).sqrt() * p.sparse_snr;
    let lam: Vec<f64> = sup.active.iter().map(|&px| pen * f64::from(sn[px])).collect();
    let w = solve_nnqp(&g, k, &b, &lam, &sup.indptr, &sup.unit);
    // rows per unit, pixels ascending (pairs are sorted by pixel); zeros eliminated
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k];
    for (j, &x) in w.iter().enumerate() {
        let u = sup.unit[j];
        let v = x / scale[u];
        if v != 0.0 {
            rows[u].push((sup.pix[j], v));
        }
    }
    let nnz: Vec<usize> = rows.iter().map(Vec::len).collect();
    let mask: Vec<bool> = nnz.iter().map(|&n| size_ok(n, p.size_thres)).collect();
    if !mask.iter().any(|&m| m) {
        return Err("all units dropped by size_thres".into());
    }
    let mut out = Csr {
        k: 0,
        p: a.p,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    let mut norm_fac = Vec::new();
    for (r, _) in rows.iter().zip(&mask).filter(|(_, &m)| m) {
        let mx = r.iter().fold(0f64, |m, &(_, v)| m.max(v));
        let inv = 1.0 / mx;
        for &(px, v) in r {
            let x = inv * v;
            if x != 0.0 {
                out.indices.push(px);
                out.data.push(x as f32);
            }
        }
        out.indptr.push(out.indices.len());
        out.k += 1;
        norm_fac.push(mx as f32);
    }
    Ok(Spatial {
        sup,
        b,
        lam,
        w,
        nnz,
        mask,
        norm_fac,
        a: out,
    })
}

// ---------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------

/// Tile-local pixel rows of tile `b` as flat pixels (`Factors.tile_rows`).
fn tile_rows(f: &Factors, b: usize) -> Vec<usize> {
    let g = &f.grid[b];
    (g.y0..g.y1)
        .flat_map(|y| (g.x0..g.x1).map(move |x| y * f.w + x))
        .collect()
}

/// `A x` (`csr_matvec`, float64).
fn a_dot(a: &Csr, x: &[f64]) -> Vec<f64> {
    (0..a.k)
        .map(|i| {
            let (idx, val) = a.row(i);
            idx.iter()
                .zip(val)
                .fold(0.0, |s, (&j, &v)| f64::from(v).mul_add(x[j], s))
        })
        .collect()
}

/// `cnmf._background_unclipped`: `b = clip(mean_t Y - A^T mean_t C, 0)` (unless given) and
/// `f_t = (Y_t - A^T C_t) . b / (b . b)` on the factors.
fn background_unclipped(f: &Factors, a: &Csr, c: &[f32], given: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
    let (k, t) = (a.k, f.t);
    let b64: Vec<f64> = match given {
        Some(b) => b.iter().map(|&v| f64::from(v)).collect(),
        None => {
            let mut ymean = vec![0f64; f.p()];
            for (bi, tf) in f.tiles.iter().enumerate() {
                let r = tf.rank;
                let mut vm = vec![0f64; r];
                for vrow in tf.v.chunks_exact(r.max(1)) {
                    for (s, &v) in vm.iter_mut().zip(vrow) {
                        *s += f64::from(v);
                    }
                }
                vm.iter_mut().for_each(|s| *s /= t as f64);
                for (l, px) in tile_rows(f, bi).into_iter().enumerate() {
                    let wrow = &tf.w[l * r..(l + 1) * r];
                    let s = wrow
                        .iter()
                        .zip(&vm)
                        .fold(0.0, |s, (&w, &m)| f64::from(w).mul_add(m, s));
                    ymean[px] = s + f64::from(tf.mean[l]);
                }
            }
            let cmean: Vec<f64> = c
                .chunks_exact(t.max(1))
                .map(|row| {
                    let r: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
                    pairwise_sum(&r) / t as f64
                })
                .collect();
            let mut fit = vec![0f64; f.p()];
            for (i, &cm) in cmean.iter().enumerate() {
                let (idx, val) = a.row(i);
                for (&j, &v) in idx.iter().zip(val) {
                    fit[j] = f64::from(v).mul_add(cm, fit[j]);
                }
            }
            ymean.iter().zip(&fit).map(|(&y, &x)| (y - x).max(0.0)).collect()
        }
    };
    let ab = a_dot(a, &b64);
    let mut num = vec![0f64; t];
    for (ti, n) in num.iter_mut().enumerate() {
        let s = (0..k).fold(0.0, |s, i| f64::from(c[i * t + ti]).mul_add(ab[i], s));
        *n = -s;
    }
    for (bi, tf) in f.tiles.iter().enumerate() {
        let r = tf.rank;
        let rows = tile_rows(f, bi);
        let mut wb = vec![0f64; r];
        let mut mb = 0f64;
        for (l, &px) in rows.iter().enumerate() {
            let bv = b64[px];
            for (s, &w) in wb.iter_mut().zip(&tf.w[l * r..(l + 1) * r]) {
                *s = f64::from(w).mul_add(bv, *s);
            }
            mb = f64::from(tf.mean[l]).mul_add(bv, mb);
        }
        for (n, vrow) in num.iter_mut().zip(tf.v.chunks_exact(r.max(1))) {
            *n += vrow
                .iter()
                .zip(&wb)
                .fold(0.0, |s, (&v, &w)| f64::from(v).mul_add(w, s));
        }
        num.iter_mut().for_each(|n| *n += mb);
    }
    let bb = b64.iter().fold(0.0, |s, &v| v.mul_add(v, s));
    let fb = num
        .iter()
        .map(|&n| if bb > 0.0 { (n / bb) as f32 } else { 0.0 })
        .collect();
    (b64.iter().map(|&v| v as f32).collect(), fb)
}

/// One tile's dense `A_b` over the units that touch it: `(units, (k_b, p_b) float32)`.
fn tile_footprints(a: &Csr, rows: &[usize], loc: &[usize]) -> (Vec<usize>, Vec<f32>) {
    let pb = rows.len();
    let (lo, hi) = (rows[0], rows[pb - 1]);
    let mut units = Vec::new();
    let mut dense = Vec::new();
    for i in 0..a.k {
        let (idx, val) = a.row(i);
        let s = idx.partition_point(|&j| j < lo);
        let e = idx.partition_point(|&j| j <= hi);
        let mut row = vec![0f32; pb];
        let mut any = false;
        for (&j, &v) in idx[s..e].iter().zip(&val[s..e]) {
            if loc[j] != usize::MAX {
                row[loc[j]] = v;
                any = true;
            }
        }
        if any {
            units.push(i);
            dense.extend(row);
        }
    }
    (units, dense)
}

/// `clip(Y_b[:, t0:t1] - A_b^T C_b[:, t0:t1], 0)`, `(p_b, t1 - t0)` float32: the
/// reconstruction and `A_b^T C_b` as sequential float32 FMAs (over the components, then
/// over the units; a zero footprint value adds exactly nothing, so it is skipped), each
/// frame in its own lane.
fn clipped_residual(f: &Factors, b: usize, ab: &[f32], cb: &[&[f32]], t0: usize, t1: usize) -> Vec<f32> {
    let tf = &f.tiles[b];
    let (r, pb, tc) = (tf.rank, tf.mean.len(), t1 - t0);
    let mut vt = vec![0f32; r * tc]; // V[t0:t1]^T
    for (ti, vrow) in tf.v[t0 * r..t1 * r].chunks_exact(r.max(1)).enumerate() {
        for (q, &v) in vrow.iter().enumerate() {
            vt[q * tc + ti] = v;
        }
    }
    let mut out = vec![0f32; pb * tc];
    let mut fit = vec![0f32; tc];
    for (l, o) in out.chunks_exact_mut(tc.max(1)).enumerate() {
        for (&w, vq) in tf.w[l * r..(l + 1) * r].iter().zip(vt.chunks_exact(tc.max(1))) {
            for (x, &v) in o.iter_mut().zip(vq) {
                *x = w.mul_add(v, *x);
            }
        }
        fit.fill(0.0);
        for (kb, crow) in cb.iter().enumerate() {
            let a = ab[kb * pb + l];
            if a != 0.0 {
                for (x, &c) in fit.iter_mut().zip(&crow[t0..t1]) {
                    *x = a.mul_add(c, *x);
                }
            }
        }
        let m = tf.mean[l];
        for (x, &y) in o.iter_mut().zip(&fit) {
            *x = ((*x + m) - y).max(0.0);
        }
    }
    out
}

/// `cnmf.f64_spot_check`: relative error of a float32 `R^T b` against float64.
#[allow(clippy::too_many_arguments)]
fn f64_spot_check(
    f: &Factors,
    b: usize,
    ab: &[f32],
    cb: &[&[f32]],
    t0: usize,
    t1: usize,
    bt: &[f32],
    part: &[f32],
) -> f64 {
    let tf = &f.tiles[b];
    let (r, pb) = (tf.rank, tf.mean.len());
    let (mut den, mut diff) = (0f64, 0f64);
    for (ti, &pv) in (t0..t1).zip(part) {
        let mut s = 0f64;
        for l in 0..pb {
            let y: f64 = (0..r)
                .map(|q| f64::from(tf.w[l * r + q]) * f64::from(tf.v[ti * r + q]))
                .sum::<f64>()
                + f64::from(tf.mean[l]);
            let fit: f64 = cb
                .iter()
                .enumerate()
                .map(|(kb, crow)| f64::from(ab[kb * pb + l]) * f64::from(crow[ti]))
                .sum();
            s += (y - fit).max(0.0) * f64::from(bt[l]);
        }
        den += s * s;
        diff += (f64::from(pv) - s).powi(2);
    }
    let (den, diff) = (den.sqrt(), diff.sqrt());
    if den > 0.0 {
        diff / den
    } else {
        diff
    }
}

/// `cnmf._background_clipped` (Minian's): `b = mean_t clip(Y - A^T C, 0)` (unless given),
/// `f_t = clip(Y_t - A^T C_t, 0) . b / (b . b)`, `BG_CHUNK` frames of one tile at a time.
/// Also returns the spot check's relative error.
fn background_clipped(f: &Factors, a: &Csr, c: &[f32], given: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, f64) {
    let t = f.t;
    let mut b_out: Vec<f32> = given.map_or_else(|| vec![0f32; f.p()], <[f32]>::to_vec);
    let mut num = vec![0f64; t];
    let check_tile = t % f.tiles.len();
    let mut err = 0.0;
    let mut loc = vec![usize::MAX; f.p()];
    let ranges: Vec<(usize, usize)> = (0..t)
        .step_by(BG_CHUNK)
        .map(|t0| (t0, t.min(t0 + BG_CHUNK)))
        .collect();
    for b in 0..f.tiles.len() {
        let rows = tile_rows(f, b);
        for (l, &px) in rows.iter().enumerate() {
            loc[px] = l;
        }
        let (units, ab) = tile_footprints(a, &rows, &loc);
        let cb: Vec<&[f32]> = units.iter().map(|&i| &c[i * t..(i + 1) * t]).collect();
        if given.is_none() {
            let mut acc = vec![0f64; rows.len()];
            for &(t0, t1) in &ranges {
                let res = clipped_residual(f, b, &ab, &cb, t0, t1);
                for (s, row) in acc.iter_mut().zip(res.chunks_exact(t1 - t0)) {
                    let r: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
                    *s += pairwise_sum(&r);
                }
            }
            for (&px, &s) in rows.iter().zip(&acc) {
                b_out[px] = (s / t as f64) as f32;
            }
        }
        let bt: Vec<f32> = rows.iter().map(|&px| b_out[px]).collect();
        for &(t0, t1) in &ranges {
            let res = clipped_residual(f, b, &ab, &cb, t0, t1);
            let tc = t1 - t0;
            let mut part = vec![0f32; tc];
            for (row, &bv) in res.chunks_exact(tc).zip(&bt) {
                for (pv, &x) in part.iter_mut().zip(row) {
                    *pv = x.mul_add(bv, *pv);
                }
            }
            for (n, &pv) in num[t0..t1].iter_mut().zip(&part) {
                *n += f64::from(pv);
            }
            if b == check_tile && t1 == t {
                err = f64_spot_check(f, b, &ab, &cb, t0, t1, &bt, &part);
            }
        }
        for &px in &rows {
            loc[px] = usize::MAX;
        }
    }
    let bb = b_out
        .iter()
        .fold(0f64, |s, &v| f64::from(v).mul_add(f64::from(v), s));
    let fb = num
        .iter()
        .map(|&n| if bb > 0.0 { (n / bb) as f32 } else { 0.0 })
        .collect();
    (b_out, fb, err)
}

/// `cnmf.update_background`: `(b, f, spot-check error)` (the error is 0 unclipped).
pub fn update_background(
    f: &Factors,
    a: &Csr,
    c: &[f32],
    given: Option<&[f32]>,
    clip: bool,
) -> (Vec<f32>, Vec<f32>, f64) {
    if clip {
        background_clipped(f, a, c, given)
    } else {
        let (b, fb) = background_unclipped(f, a, c, given);
        (b, fb, 0.0)
    }
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

/// `Factors.AtY` on a float64 `A` (as `compute_trace` calls it): per tile, `A_b W_b` in
/// float64 over the tile's pixels in order, times `V_b^T`, plus `A_b mean_b`.
fn a_ty64(f: &Factors, a: &Csr) -> Vec<f64> {
    let t = f.t;
    let mut out = vec![0f64; a.k * t];
    let mut parts: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); f.tiles.len()];
    for i in 0..a.k {
        let (idx, val) = a.row(i);
        for (&px, &v) in idx.iter().zip(val) {
            let (b, l) = f.locate(px);
            parts[b].push((i, l, v));
        }
    }
    for (b, part) in parts.iter_mut().enumerate() {
        part.sort_unstable_by_key(|&(i, l, _)| (i, l));
        let tf = &f.tiles[b];
        let r = tf.rank;
        let mut s = 0;
        while s < part.len() {
            let i = part[s].0;
            let e = s + part[s..].iter().take_while(|p| p.0 == i).count();
            let mut u = vec![0f64; r];
            let mut m = 0f64;
            for &(_, l, v) in &part[s..e] {
                let v = f64::from(v);
                for (uk, &wk) in u.iter_mut().zip(&tf.w[l * r..(l + 1) * r]) {
                    *uk = v.mul_add(f64::from(wk), *uk);
                }
                m = v.mul_add(f64::from(tf.mean[l]), m);
            }
            for (o, vrow) in out[i * t..(i + 1) * t]
                .iter_mut()
                .zip(tf.v.chunks_exact(r.max(1)))
            {
                let low = u
                    .iter()
                    .zip(vrow)
                    .fold(0.0, |acc, (&x, &y)| x.mul_add(f64::from(y), acc));
                *o += low;
                *o += m;
            }
            s = e;
        }
    }
    out
}

/// `cnmf.compute_trace`: `clip(C + A_norm ((Y - b f^T)^T A - C^T A A^T)^T, 0)` float32.
pub fn compute_trace(f: &Factors, a: &Csr, b: &[f32], c: &[f32], fb: &[f32]) -> Vec<f32> {
    let (k, t) = (a.k, f.t);
    let b64: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
    let ab = a_dot(a, &b64);
    let mut yta = a_ty64(f, a);
    for (i, row) in yta.chunks_exact_mut(t.max(1)).enumerate() {
        for (y, &fv) in row.iter_mut().zip(fb) {
            *y -= ab[i] * f64::from(fv);
        }
    }
    let aa = gram(a);
    let norm: Vec<f64> = (0..k)
        .map(|i| {
            let s = a.row(i).1.iter().fold(0f64, |s, &v| {
                let v = f64::from(v);
                (v * v).mul_add(1.0, s)
            });
            1.0 / s
        })
        .collect();
    // per unit, the units whose footprints overlap it (nonzero A A^T), ascending
    let nbr: Vec<Vec<(usize, f64)>> = (0..k)
        .map(|j| {
            (0..k)
                .filter(|&i| aa[i * k + j] != 0.0)
                .map(|i| (i, aa[i * k + j]))
                .collect()
        })
        .collect();
    let mut out = vec![0f32; k * t];
    for j in 0..k {
        for ti in 0..t {
            let cta = nbr[j]
                .iter()
                .fold(0.0, |s, &(i, g)| f64::from(c[i * t + ti]).mul_add(g, s));
            let v = f64::from(c[j * t + ti]) + (yta[j * t + ti] - cta) * norm[j];
            out[j * t + ti] = v.max(0.0) as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Loop merge
// ---------------------------------------------------------------------------

pub struct Merge {
    pub pairs: Vec<[usize; 2]>, // units whose footprints overlap (A A^T > 0), i < j
    pub pair_corr: Vec<f64>,
    pub labels: Vec<usize>,
    pub a: Csr,
    pub c: Vec<f32>,
    pub extra: Vec<f32>,
}

/// Unit pairs `(i, j)`, `i < j`, with a positive float32 entry in `A A^T` (a shared pixel
/// whose product does not vanish), lexicographic.
fn product_pairs(a: &Csr) -> Vec<[usize; 2]> {
    let mut by_px: Vec<Vec<(usize, f32)>> = vec![Vec::new(); a.p];
    for i in 0..a.k {
        let (idx, val) = a.row(i);
        for (&j, &v) in idx.iter().zip(val) {
            by_px[j].push((i, v));
        }
    }
    let mut hit = vec![false; a.k * a.k];
    for col in &by_px {
        for (x, &(i, vi)) in col.iter().enumerate() {
            for &(j, vj) in &col[x + 1..] {
                if vi * vj > 0.0 {
                    hit[i * a.k + j] = true;
                }
            }
        }
    }
    (0..a.k * a.k)
        .filter(|&ij| hit[ij])
        .map(|ij| [ij / a.k, ij % a.k])
        .collect()
}

/// `init.unit_merge` as the loop calls it: groups of overlapping units whose `traces_corr`
/// exceeds `thres` (or closer than `min_dist`), each replaced by its mean (`a`, `c` and
/// `extra`).
#[allow(clippy::too_many_arguments)]
pub fn unit_merge(
    a: &Csr,
    c: &[f32],
    extra: &[f32],
    t: usize,
    w: usize,
    noise_freq: f64,
    thres: f64,
    min_dist: f64,
) -> Merge {
    let pairs = product_pairs(a);
    let tc = TracesCorr::new(c, a.k, t, noise_freq);
    let pair_corr: Vec<f64> = pairs.iter().map(|&[i, j]| tc.pair(i, j)).collect();
    let mut edges: Vec<[usize; 2]> = pairs
        .iter()
        .zip(&pair_corr)
        .filter(|(_, &r)| r > thres)
        .map(|(&e, _)| e)
        .collect();
    if min_dist > 0.0 {
        let cent = centroids(a, w);
        for i in 0..a.k {
            for j in i + 1..a.k {
                let (dy, dx) = (cent[i][0] - cent[j][0], cent[i][1] - cent[j][1]);
                if (dy * dy + dx * dx).sqrt() < min_dist {
                    edges.push([i, j]);
                }
            }
        }
    }
    let labels = merge_labels(a.k, &edges);
    Merge {
        a: group_mean_a(a, &labels),
        c: group_mean_c(c, t, &labels),
        extra: group_mean_c(extra, t, &labels),
        pairs,
        pair_corr,
        labels,
    }
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

pub struct Iteration {
    pub spatial: Spatial,
    pub c_sp: Vec<f32>,
    pub b: Vec<f32>,
    pub f: Vec<f32>,
    pub yra: Vec<f32>,
    pub merge: Merge,
}

pub struct CnmfResult {
    pub b_init: Vec<f32>,
    pub f_init: Vec<f32>,
    pub iters: Vec<Iteration>,
    pub t: usize,
    pub units_in: usize,
    pub spot_checks: Vec<f64>,
    pub seconds: Vec<(String, f64)>,
}

impl CnmfResult {
    /// `CnmfResult.A`, `.C`, `.YrA`, `.b`, `.f` (the last iteration's).
    pub fn last(&self) -> &Iteration {
        self.iters.last().expect("n_iter >= 1")
    }
}

fn check_finite(stage: &str, arrays: &[(&str, &[f32])]) -> Result<(), String> {
    let bad: Vec<&str> = arrays
        .iter()
        .filter(|(_, v)| v.iter().any(|x| !x.is_finite()))
        .map(|&(n, _)| n)
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("CNMF {stage}: non-finite values in {}", bad.join(", ")))
    }
}

/// `cnmf.run_cnmf` on init's units `a` and traces `c` (`(k, t)` float32).
pub fn run(
    f: &Factors,
    a: &Csr,
    c: &[f32],
    p: &CnmfParams,
    now: &dyn Fn() -> f64,
) -> Result<CnmfResult, String> {
    if p.h != f.h || p.w != f.w || p.t != f.t {
        return Err(format!(
            "cnmf.json says {}x{}, {} frames; the factors are {}x{}, {} frames",
            p.h, p.w, p.t, f.h, f.w, f.t
        ));
    }
    if p.n_iter == 0 {
        return Err("n_iter must be at least 1".into());
    }
    let (t, units_in) = (f.t, a.k);
    let sn: Vec<f32> = {
        let mut s = vec![0f32; f.p()];
        for (px, v) in s.iter_mut().enumerate() {
            *v = f.sigma(px);
        }
        s
    };
    let mut seconds = Vec::new();
    let mut t0 = now();
    let mut lap = |name: String, seconds: &mut Vec<(String, f64)>| {
        let n = now();
        seconds.push((name, n - t0));
        t0 = n;
    };
    let mut spot_checks = Vec::new();
    let (b_init, f_init, err) = update_background(f, a, c, None, p.background_clip);
    spot_checks.push(err);
    check_finite("initial background", &[("b", &b_init), ("f", &f_init)])?;
    lap("background_init".into(), &mut seconds);
    let (mut a, mut c) = (a.clone(), c.to_vec());
    let mut b = b_init.clone();
    let mut iters = Vec::new();
    for it in 0..p.n_iter {
        let sp = update_spatial(f, &a, &c, &sn, p)?;
        let kept: Vec<usize> = (0..a.k).filter(|&i| sp.mask[i]).collect();
        let mut c_sp = Vec::with_capacity(kept.len() * t);
        for (&i, &nf) in kept.iter().zip(&sp.norm_fac) {
            c_sp.extend(c[i * t..(i + 1) * t].iter().map(|&x| x * nf));
        }
        check_finite(
            &format!("spatial update (iteration {it})"),
            &[("A_data", &sp.a.data), ("C", &c_sp)],
        )?;
        lap(format!("spatial_{it}"), &mut seconds);
        let last = it == p.n_iter - 1;
        let (bn, fb, err) = update_background(
            f,
            &sp.a,
            &c_sp,
            if last { None } else { Some(&b) },
            p.background_clip,
        );
        spot_checks.push(err);
        check_finite(&format!("background (iteration {it})"), &[("b", &bn), ("f", &fb)])?;
        b = bn;
        lap(format!("background_{it}"), &mut seconds);
        let yra = compute_trace(f, &sp.a, &b, &c_sp, &fb);
        check_finite(&format!("traces (iteration {it})"), &[("YrA", &yra)])?;
        lap(format!("trace_{it}"), &mut seconds);
        let m = unit_merge(
            &sp.a,
            &yra,
            &c_sp,
            t,
            f.w,
            p.noise_freq,
            p.loop_merge_corr,
            p.merge_min_dist,
        );
        lap(format!("merge_{it}"), &mut seconds);
        a = m.a.clone();
        c = m.c.clone();
        iters.push(Iteration {
            spatial: sp,
            c_sp,
            b: b.clone(),
            f: fb,
            yra,
            merge: m,
        });
    }
    Ok(CnmfResult {
        b_init,
        f_init,
        iters,
        t,
        units_in,
        spot_checks,
        seconds,
    })
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

fn csr_files(name: &str, a: &Csr, out: &mut Vec<(String, Vec<u8>)>) {
    let ptr: Vec<i64> = a.indptr.iter().map(|&v| v as i64).collect();
    let idx: Vec<i64> = a.indices.iter().map(|&v| v as i64).collect();
    out.push((format!("{name}_indptr.npy"), npy::write(&[ptr.len()], &ptr)));
    out.push((format!("{name}_indices.npy"), npy::write(&[idx.len()], &idx)));
    out.push((format!("{name}_data.npy"), npy::write(&[a.data.len()], &a.data)));
}

fn ints(v: &[usize]) -> Vec<i64> {
    v.iter().map(|&x| x as i64).collect()
}

/// The files `ref_cnmf.py` writes to `ref/` (all but `G_{i}` and `kkt_{i}`), same names.
pub fn outputs(r: &CnmfResult, mut summary: serde_json::Value) -> Vec<(String, Vec<u8>)> {
    let t = r.t;
    let mut out = Vec::new();
    let put = |out: &mut Vec<(String, Vec<u8>)>, name: String, bytes: Vec<u8>| {
        out.push((format!("{name}.npy"), bytes))
    };
    put(
        &mut out,
        "b_init".into(),
        npy::write(&[r.b_init.len()], &r.b_init),
    );
    put(&mut out, "f_init".into(), npy::write(&[t], &r.f_init));
    for (i, it) in r.iters.iter().enumerate() {
        let sp = &it.spatial;
        put(
            &mut out,
            format!("sub_{i}_active"),
            npy::write(&[sp.sup.active.len()], &ints(&sp.sup.active)),
        );
        put(
            &mut out,
            format!("sub_{i}_indptr"),
            npy::write(&[sp.sup.indptr.len()], &ints(&sp.sup.indptr)),
        );
        put(
            &mut out,
            format!("sub_{i}_units"),
            npy::write(&[sp.sup.unit.len()], &ints(&sp.sup.unit)),
        );
        put(&mut out, format!("B_{i}"), npy::write(&[sp.b.len()], &sp.b));
        put(&mut out, format!("lam_{i}"), npy::write(&[sp.lam.len()], &sp.lam));
        put(&mut out, format!("w_{i}"), npy::write(&[sp.w.len()], &sp.w));
        put(
            &mut out,
            format!("nnz_{i}"),
            npy::write(&[sp.nnz.len()], &ints(&sp.nnz)),
        );
        put(
            &mut out,
            format!("mask_{i}"),
            npy::write(&[sp.mask.len()], &sp.mask),
        );
        put(
            &mut out,
            format!("norm_fac_{i}"),
            npy::write(&[sp.norm_fac.len()], &sp.norm_fac),
        );
        csr_files(&format!("spatial_{i}"), &sp.a, &mut out);
        put(&mut out, format!("C_sp_{i}"), npy::write(&[sp.a.k, t], &it.c_sp));
        put(&mut out, format!("b_{i}"), npy::write(&[it.b.len()], &it.b));
        put(&mut out, format!("f_{i}"), npy::write(&[t], &it.f));
        put(&mut out, format!("YrA_{i}"), npy::write(&[sp.a.k, t], &it.yra));
        let m = &it.merge;
        let pairs: Vec<i64> = m.pairs.iter().flatten().map(|&v| v as i64).collect();
        put(
            &mut out,
            format!("unit_pairs_{i}"),
            npy::write(&[m.pairs.len(), 2], &pairs),
        );
        put(
            &mut out,
            format!("unit_corr_{i}"),
            npy::write(&[m.pair_corr.len()], &m.pair_corr),
        );
        put(
            &mut out,
            format!("merge_labels_{i}"),
            npy::write(&[m.labels.len()], &ints(&m.labels)),
        );
        let mut counts = vec![0usize; m.a.k];
        m.labels.iter().for_each(|&l| counts[l] += 1);
        put(
            &mut out,
            format!("merge_counts_{i}"),
            npy::write(&[counts.len()], &ints(&counts)),
        );
        csr_files(&format!("merge_{i}"), &m.a, &mut out);
    }
    let last = r.last();
    csr_files("A", &last.merge.a, &mut out);
    put(
        &mut out,
        "C".into(),
        npy::write(&[last.merge.a.k, t], &last.merge.extra),
    );
    put(
        &mut out,
        "YrA".into(),
        npy::write(&[last.merge.a.k, t], &last.merge.c),
    );
    put(&mut out, "b".into(), npy::write(&[last.b.len()], &last.b));
    put(&mut out, "f".into(), npy::write(&[t], &last.f));
    let mut counts = serde_json::Map::new();
    for (i, it) in r.iters.iter().enumerate() {
        counts.insert(format!("units_after_spatial_{i}"), it.spatial.a.k.into());
        counts.insert(format!("units_after_merge_{i}"), it.merge.a.k.into());
    }
    counts.insert("iter".into(), r.iters.len().into());
    summary["frames"] = t.into();
    summary["iterations"] = r.iters.len().into();
    summary["units_in"] = r.units_in.into();
    summary["units_final"] = last.merge.a.k.into();
    summary["counts"] = counts.into();
    summary["support_pairs"] = r
        .iters
        .iter()
        .enumerate()
        .map(|(i, it)| (i.to_string(), serde_json::Value::from(it.spatial.w.len())))
        .collect::<serde_json::Map<_, _>>()
        .into();
    summary["overlap_pairs"] = r
        .iters
        .iter()
        .enumerate()
        .map(|(i, it)| (i.to_string(), serde_json::Value::from(it.merge.pairs.len())))
        .collect::<serde_json::Map<_, _>>()
        .into();
    summary["spot_checks"] = r.spot_checks.clone().into();
    summary["stages"] = r
        .seconds
        .iter()
        .map(|(n, s)| (n.clone(), serde_json::Value::from(*s)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    out.push((
        "summary.json".into(),
        serde_json::to_vec_pretty(&summary).unwrap(),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(rows: &[&[(usize, f32)]], p: usize) -> Csr {
        let mut indptr = vec![0];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for r in rows {
            for &(j, v) in *r {
                indices.push(j);
                data.push(v);
            }
            indptr.push(indices.len());
        }
        Csr {
            k: rows.len(),
            p,
            indptr,
            indices,
            data,
        }
    }

    #[test]
    fn dilation_skips_zeros_and_clips_at_the_border() {
        // 3x3 frame, unit 0 at the centre, unit 1 a stored zero only
        let a = csr(&[&[(4, 1.0)], &[(0, 0.0)]], 9);
        let s = dilate(&a, 3, 3, 1);
        assert_eq!(s.active, vec![1, 3, 4, 5, 7]);
        assert!(s.unit.iter().all(|&u| u == 0));
    }

    #[test]
    fn size_mask_is_strict() {
        assert!(!size_ok(0, [None, None]));
        assert!(!size_ok(25, [Some(25.0), None]));
        assert!(size_ok(26, [Some(25.0), None]));
        assert!(!size_ok(30, [Some(25.0), Some(30.0)]));
    }

    #[test]
    fn product_pairs_ignore_vanishing_products() {
        let tiny = f32::from_bits(1); // smallest subnormal: its square underflows
        let a = csr(&[&[(0, 1.0), (1, tiny)], &[(1, tiny)], &[(0, 0.5)]], 2);
        assert_eq!(product_pairs(&a), vec![[0, 2]]);
    }

    #[test]
    fn unit_traces_are_centred_unit_rows() {
        let (cc, scale) = unit_traces(&[1.0, 2.0, 3.0, 5.0, 5.0, 5.0], 2, 3);
        assert!((scale[0] - 2f64.sqrt()).abs() < 1e-15);
        assert_eq!(scale[1], 1.0);
        assert_eq!(&cc[3..], &[0.0, 0.0, 0.0]);
        let g = gram_rows(&cc, 2, 3);
        assert!((g[0] - 1.0).abs() < 1e-15);
    }
}
