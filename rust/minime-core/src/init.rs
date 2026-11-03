//! `minime.init` on one window's factors: `initA` (each merged seed's correlation window
//! thresholded), `initC` (least-squares traces through the normal equations, `A Y` from
//! the factors) and the initial `unit_merge` (mean of each group of overlapping units whose
//! low-passed traces correlate), as `pipeline.init_units` runs them.
//!
//! Sums follow scipy's and numpy's orders where they decide float32 values: `A Y` is, per
//! tile, the sparse product `A_b W_b` accumulated pixel by pixel in the tile's own order
//! (scipy's `csc_matvecs`, float32), then that times `V_b^T` (float32, sequential FMAs as
//! in `Factors::trace`), added to a float64 sum, plus `A_b mean_b` in float64.

use crate::corr::{FactorCorr, TracesCorr};
use crate::factors::Factors;
use crate::linalg::NormalSolver;
use crate::npy;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct InitParams {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub tile: usize,
    pub noise_freq: f64,
    pub inita_corr: f64,
    pub inita_wnd: usize,
    pub init_merge_corr: f64,
    pub merge_min_dist: f64,
}

/// `(k, p)` float32 CSR footprints, column indices ascending in each row.
#[derive(Clone)]
pub struct Csr {
    pub k: usize,
    pub p: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f32>,
}

impl Csr {
    pub fn row(&self, i: usize) -> (&[usize], &[f32]) {
        let (a, b) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[a..b], &self.data[a..b])
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Units per pixel (the transpose's rows): `(indptr over pixels, units)`, units ascending.
    fn by_pixel(&self) -> (Vec<usize>, Vec<usize>) {
        let mut ptr = vec![0usize; self.p + 1];
        for &j in &self.indices {
            ptr[j + 1] += 1;
        }
        for j in 0..self.p {
            ptr[j + 1] += ptr[j];
        }
        let mut fill = ptr.clone();
        let mut units = vec![0usize; self.nnz()];
        for i in 0..self.k {
            for &j in self.row(i).0 {
                units[fill[j]] = i;
                fill[j] += 1;
            }
        }
        (ptr, units)
    }
}

// ---------------------------------------------------------------------------
// Footprints
// ---------------------------------------------------------------------------

/// `FactorCorr._disk`: offsets `(dy, dx)` with `dy^2 + dx^2 <= wnd^2`, row-major.
pub fn disk(wnd: usize) -> Vec<(isize, isize)> {
    let r = wnd as isize;
    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dy * dy + dx * dx <= r * r {
                out.push((dy, dx));
            }
        }
    }
    out
}

/// `init.initA`: each seed's window pixels whose correlation to it is `>= thres`; a seed
/// with none yields no row. Also returns every window correlation (seeds in order).
pub fn init_a(
    fc: &mut FactorCorr,
    f: &Factors,
    seeds: &[[i64; 2]],
    thres: f64,
    wnd: usize,
) -> (Csr, Vec<f64>) {
    let off = disk(wnd);
    let mut indptr = vec![0usize];
    let (mut indices, mut data, mut window) = (Vec::new(), Vec::new(), Vec::new());
    for &[h0, w0] in seeds {
        let seed = h0 as usize * f.w + w0 as usize;
        let before = indices.len();
        for &(dy, dx) in &off {
            let (h, w) = (h0 as isize + dy, w0 as isize + dx);
            if h < 0 || h >= f.h as isize || w < 0 || w >= f.w as isize {
                continue;
            }
            let px = h as usize * f.w + w as usize;
            let c = fc.pair(f, seed, px);
            window.push(c);
            if c >= thres {
                indices.push(px);
                data.push(c as f32);
            }
        }
        if indices.len() > before {
            indptr.push(indices.len());
        }
    }
    let a = Csr {
        k: indptr.len() - 1,
        p: f.p(),
        indptr,
        indices,
        data,
    };
    (a, window)
}

/// `init.centroids`: intensity-weighted `(row, col)` of each footprint, NaN if empty.
pub fn centroids(a: &Csr, w: usize) -> Vec<[f64; 2]> {
    (0..a.k)
        .map(|i| {
            let (idx, val) = a.row(i);
            let (mut m, mut y, mut x) = (0.0, 0.0, 0.0);
            for (&j, &v) in idx.iter().zip(val) {
                let v = f64::from(v);
                m += v;
                y += v * (j / w) as f64;
                x += v * (j % w) as f64;
            }
            if m > 0.0 {
                [y / m, x / m]
            } else {
                [f64::NAN; 2]
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

/// `Factors.AtY`: `A Y` as `(k, t)` float64, tile by tile.
pub fn a_ty(f: &Factors, a: &Csr) -> Vec<f64> {
    let t = f.t;
    let mut out = vec![0f64; a.k * t];
    // per unit, its pixels grouped by tile in the tile's own pixel order
    let mut parts: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); f.tiles.len()];
    for i in 0..a.k {
        let (idx, val) = a.row(i);
        for (&px, &v) in idx.iter().zip(val) {
            let (b, l) = f.locate(px);
            parts[b].push((i, l, v));
        }
    }
    let mut low = vec![0f32; t];
    for (b, part) in parts.iter_mut().enumerate() {
        part.sort_unstable_by_key(|&(i, l, _)| (i, l));
        let tf = &f.tiles[b];
        let r = tf.rank;
        let mut s = 0;
        while s < part.len() {
            let i = part[s].0;
            let e = s + part[s..].iter().take_while(|p| p.0 == i).count();
            let mut u = vec![0f32; r];
            let mut m = 0f64;
            for &(_, l, v) in &part[s..e] {
                for (uk, &wk) in u.iter_mut().zip(&tf.w[l * r..(l + 1) * r]) {
                    *uk = v.mul_add(wk, *uk);
                }
                m = f64::from(v).mul_add(f64::from(tf.mean[l]), m);
            }
            for (o, vrow) in low.iter_mut().zip(tf.v.chunks_exact(r.max(1))) {
                let mut acc = 0f32;
                for (&x, &y) in u.iter().zip(vrow) {
                    acc = x.mul_add(y, acc);
                }
                *o = acc;
            }
            for (o, &x) in out[i * t..(i + 1) * t].iter_mut().zip(&low) {
                *o += f64::from(x);
                *o += m;
            }
            s = e;
        }
    }
    out
}

/// `A A^T` (float64, `(k, k)`): each entry summed over the shared pixels in ascending order.
pub fn gram(a: &Csr) -> Vec<f64> {
    let (ptr, units) = a.by_pixel();
    let mut g = vec![0f64; a.k * a.k];
    for i in 0..a.k {
        let (idx, val) = a.row(i);
        for (&j, &v) in idx.iter().zip(val) {
            let v = f64::from(v);
            for &k in &units[ptr[j]..ptr[j + 1]] {
                let (kidx, kval) = a.row(k);
                let pos = kidx.binary_search(&j).expect("pixel in row");
                g[i * a.k + k] += v * f64::from(kval[pos]);
            }
        }
    }
    g
}

/// `init.initC`: `(k, t)` float32 traces from `(A A^T) C = A Y`; also `cond(G)` and
/// whether the solve used Cholesky.
pub fn init_c(f: &Factors, a: &Csr) -> (Vec<f32>, f64, bool) {
    let (k, t) = (a.k, f.t);
    let g = gram(a);
    let solver = NormalSolver::new(&g, k);
    let b = a_ty(f, a);
    let mut c = vec![0f32; k * t];
    let mut col = vec![0f64; k];
    for ti in 0..t {
        for (i, x) in col.iter_mut().enumerate() {
            *x = b[i * t + ti];
        }
        solver.solve(&mut col);
        for (i, &x) in col.iter().enumerate() {
            c[i * t + ti] = x as f32;
        }
    }
    (c, solver.cond(), solver.is_cholesky())
}

// ---------------------------------------------------------------------------
// Unit merge
// ---------------------------------------------------------------------------

/// Unit pairs `(i, j)`, `i < j`, whose footprints share a pixel, in lexicographic order.
pub fn overlap_pairs(a: &Csr) -> Vec<[usize; 2]> {
    let (ptr, units) = a.by_pixel();
    let mut seen = vec![usize::MAX; a.k];
    let mut out = Vec::new();
    for i in 0..a.k {
        let mut row = Vec::new();
        for &j in a.row(i).0 {
            for &k in &units[ptr[j]..ptr[j + 1]] {
                if k > i && seen[k] != i {
                    seen[k] = i;
                    row.push(k);
                }
            }
        }
        row.sort_unstable();
        out.extend(row.into_iter().map(|k| [i, k]));
    }
    out
}

/// Connected components of `edges` on `n` nodes, numbered in order of first appearance.
pub fn merge_labels(n: usize, edges: &[[usize; 2]]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &[i, j] in edges {
        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut label = vec![usize::MAX; n];
    let mut out = vec![0; n];
    let mut next = 0;
    for (i, o) in out.iter_mut().enumerate() {
        let r = root(&mut parent, i);
        if label[r] == usize::MAX {
            label[r] = next;
            next += 1;
        }
        *o = label[r];
    }
    out
}

/// Members of each group, ascending (`_group_mean_matrix`'s rows).
fn groups(labels: &[usize]) -> Vec<Vec<usize>> {
    let n = labels.iter().max().map_or(0, |&m| m + 1);
    let mut g = vec![Vec::new(); n];
    for (i, &l) in labels.iter().enumerate() {
        g[l].push(i);
    }
    g
}

/// `M @ A` with `M` the group means: float64 sums over the members in order (zero sums
/// dropped), then float32.
pub fn group_mean_a(a: &Csr, labels: &[usize]) -> Csr {
    let mut indptr = vec![0usize];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    let mut acc = vec![0f64; a.p];
    let mut used = vec![false; a.p];
    for members in groups(labels) {
        let v = 1.0 / members.len() as f64;
        let mut cols = Vec::new();
        for &k in &members {
            let (idx, val) = a.row(k);
            for (&j, &x) in idx.iter().zip(val) {
                if !used[j] {
                    used[j] = true;
                    acc[j] = 0.0;
                    cols.push(j);
                }
                acc[j] = v.mul_add(f64::from(x), acc[j]);
            }
        }
        cols.sort_unstable();
        for j in cols {
            used[j] = false;
            // scipy's csr_matmat drops entries whose sum is zero
            if acc[j] != 0.0 {
                indices.push(j);
                data.push(acc[j] as f32);
            }
        }
        indptr.push(indices.len());
    }
    Csr {
        k: indptr.len() - 1,
        p: a.p,
        indptr,
        indices,
        data,
    }
}

/// `M @ C`: float64 sums over the members in order, then float32.
pub fn group_mean_c(c: &[f32], t: usize, labels: &[usize]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut acc = vec![0f64; t];
    for members in groups(labels) {
        let v = 1.0 / members.len() as f64;
        acc.fill(0.0);
        for &k in &members {
            for (s, &x) in acc.iter_mut().zip(&c[k * t..(k + 1) * t]) {
                *s = v.mul_add(f64::from(x), *s);
            }
        }
        out.extend(acc.iter().map(|&x| x as f32));
    }
    out
}

// ---------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------

pub struct InitResult {
    pub seeds: usize,
    pub a0: Csr, // initA
    pub window_corr: Vec<f64>,
    pub c0: Vec<f32>, // initC
    pub cond: f64,
    pub cholesky: bool,
    pub pairs: Vec<[usize; 2]>, // overlapping units
    pub pair_corr: Vec<f64>,    // their traces_corr
    pub labels: Vec<usize>,
    pub a: Csr, // after the merge
    pub c: Vec<f32>,
    pub t: usize,
    pub w: usize,
    pub seconds: Vec<(&'static str, f64)>,
}

/// `pipeline.init_units` on the merged seeds `seeds` ((n, 2) row, col).
pub fn run(
    fc: &mut FactorCorr,
    f: &Factors,
    seeds: &[[i64; 2]],
    p: &InitParams,
    now: &dyn Fn() -> f64,
) -> Result<InitResult, String> {
    if p.h != f.h || p.w != f.w || p.t != f.t {
        return Err(format!(
            "init.json says {}x{}, {} frames; the factors are {}x{}, {} frames",
            p.h, p.w, p.t, f.h, f.w, f.t
        ));
    }
    let mut seconds = Vec::new();
    let mut t0 = now();
    let mut lap = |name: &'static str, seconds: &mut Vec<(&'static str, f64)>| {
        let n = now();
        seconds.push((name, n - t0));
        t0 = n;
    };
    let (a0, window_corr) = init_a(fc, f, seeds, p.inita_corr, p.inita_wnd);
    if a0.k == 0 {
        return Err("initA produced no footprints".into());
    }
    lap("initA", &mut seconds);
    let (c0, cond, cholesky) = init_c(f, &a0);
    lap("initC", &mut seconds);
    let pairs = overlap_pairs(&a0);
    let tc = TracesCorr::new(&c0, a0.k, f.t, fc.noise_freq);
    let pair_corr: Vec<f64> = pairs.iter().map(|&[i, j]| tc.pair(i, j)).collect();
    let mut edges: Vec<[usize; 2]> = pairs
        .iter()
        .zip(&pair_corr)
        .filter(|(_, &c)| c > p.init_merge_corr)
        .map(|(&e, _)| e)
        .collect();
    if p.merge_min_dist > 0.0 {
        let cent = centroids(&a0, f.w);
        for i in 0..a0.k {
            for j in i + 1..a0.k {
                let (dy, dx) = (cent[i][0] - cent[j][0], cent[i][1] - cent[j][1]);
                // NaN centroids never compare close
                if (dy * dy + dx * dx).sqrt() < p.merge_min_dist {
                    edges.push([i, j]);
                }
            }
        }
    }
    let labels = merge_labels(a0.k, &edges);
    let a = group_mean_a(&a0, &labels);
    let c = group_mean_c(&c0, f.t, &labels);
    lap("unit_merge", &mut seconds);
    Ok(InitResult {
        seeds: seeds.len(),
        a0,
        window_corr,
        c0,
        cond,
        cholesky,
        pairs,
        pair_corr,
        labels,
        a,
        c,
        t: f.t,
        w: f.w,
        seconds,
    })
}

fn csr_files(name: &str, a: &Csr, out: &mut Vec<(String, Vec<u8>)>) {
    let ptr: Vec<i64> = a.indptr.iter().map(|&v| v as i64).collect();
    let idx: Vec<i64> = a.indices.iter().map(|&v| v as i64).collect();
    out.push((format!("{name}_indptr.npy"), npy::write(&[ptr.len()], &ptr)));
    out.push((format!("{name}_indices.npy"), npy::write(&[idx.len()], &idx)));
    out.push((format!("{name}_data.npy"), npy::write(&[a.data.len()], &a.data)));
}

/// The files `ref_init.py` writes to `ref/`, under the same names.
pub fn outputs(r: &InitResult, p: &InitParams, mut summary: serde_json::Value) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    csr_files("initA", &r.a0, &mut out);
    out.push((
        "initA_window_corr.npy".into(),
        npy::write(&[r.window_corr.len()], &r.window_corr),
    ));
    let cent = |a: &Csr| -> Vec<f64> { centroids(a, r.w).into_iter().flatten().collect() };
    out.push((
        "initA_centroids.npy".into(),
        npy::write(&[r.a0.k, 2], &cent(&r.a0)),
    ));
    out.push(("C_initC.npy".into(), npy::write(&[r.a0.k, r.t], &r.c0)));
    let pairs: Vec<i64> = r.pairs.iter().flatten().map(|&v| v as i64).collect();
    out.push(("unit_pairs.npy".into(), npy::write(&[r.pairs.len(), 2], &pairs)));
    out.push((
        "unit_corr.npy".into(),
        npy::write(&[r.pair_corr.len()], &r.pair_corr),
    ));
    let labels: Vec<i64> = r.labels.iter().map(|&v| v as i64).collect();
    out.push(("merge_labels.npy".into(), npy::write(&[labels.len()], &labels)));
    let counts: Vec<i64> = groups(&r.labels).iter().map(|g| g.len() as i64).collect();
    out.push(("merge_counts.npy".into(), npy::write(&[counts.len()], &counts)));
    csr_files("A_merge", &r.a, &mut out);
    out.push(("C_merge.npy".into(), npy::write(&[r.a.k, r.t], &r.c)));
    out.push(("merge_centroids.npy".into(), npy::write(&[r.a.k, 2], &cent(&r.a))));
    summary["seeds"] = r.seeds.into();
    summary["units_initA"] = r.a0.k.into();
    summary["initA_nnz"] = r.a0.nnz().into();
    summary["units_init_merge"] = r.a.k.into();
    summary["overlap_pairs"] = r.pairs.len().into();
    summary["merged_pairs"] = r
        .pair_corr
        .iter()
        .filter(|&&c| c > p.init_merge_corr)
        .count()
        .into();
    summary["cond_G"] = r.cond.into();
    summary["solver"] = if r.cholesky { "cholesky" } else { "pinv" }.into();
    summary["stages"] = r
        .seconds
        .iter()
        .map(|&(n, s)| (n.to_string(), serde_json::Value::from(s)))
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
    fn disk_matches_mgrid_order() {
        let d = disk(1);
        assert_eq!(d, vec![(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]);
        assert_eq!(disk(10).len(), 317);
    }

    #[test]
    fn overlap_gram_and_merge() {
        let a = csr(
            &[
                &[(0, 1.0), (1, 2.0)],
                &[(5, 1.0)],
                &[(1, 3.0), (2, 1.0)],
                &[(2, 1.0)],
            ],
            6,
        );
        assert_eq!(overlap_pairs(&a), vec![[0, 2], [2, 3]]);
        let g = gram(&a);
        assert_eq!(g[2], 6.0);
        assert_eq!(g[2 * 4 + 2], 10.0);
        let labels = merge_labels(4, &[[2, 3], [0, 2]]);
        assert_eq!(labels, vec![0, 1, 0, 0]);
        let m = group_mean_a(&a, &labels);
        assert_eq!(m.k, 2);
        assert_eq!(m.indices, vec![0, 1, 2, 5]);
        assert_eq!(m.data, vec![1.0 / 3.0, 5.0 / 3.0, 2.0 / 3.0, 1.0]);
        let c = group_mean_c(&[3.0, 6.0, 9.0, 0.0], 1, &labels);
        assert_eq!(c, vec![4.0, 6.0]);
        assert_eq!(centroids(&a, 3)[0], [0.0, 2.0 / 3.0]);
    }
}
