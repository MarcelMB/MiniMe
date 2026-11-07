//! The `fresh` window run's last step, `stream.stitch`: one unit list from the final units of
//! every window, and the plumbing of the window loop (`stream.time_windows`, one window's
//! stage parameters and shifts from the session's).
//!
//! Units of different windows whose footprints (rows scaled to maximum 1) have a cosine
//! above `cos_thres` are linked; each unit joins its brightest brighter linked neighbour
//! (`corr::brightest_parent`) and every tree is averaged into one footprint. The sums follow
//! scipy's (1.18): the stack's rows are sorted (`astype` canonicalises it), `diags(d) @ A`
//! and `M @ A` are `csr_matmat` (fused multiply-adds into a zeroed accumulator, each output
//! row in reverse order of first appearance, zero sums dropped), `A.multiply(A)` keeps the
//! row order when every row is strictly increasing and reverses it otherwise, and a row sum
//! is `np.add.reduceat`: the first value plus the pairwise sum of the rest. So the norms and
//! cosines, and with them the edges, come out bit for bit.

use crate::compress::TileFactors;
use crate::corr::{brightest_parent, link_brighter};
use crate::gmm::pairwise_sum;
use crate::init::Csr;
use crate::npy;
use crate::pass1::Tile;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

/// `stream.time_windows`: consecutive `[t0, t1)` windows of about `length` frames covering
/// `t` (a last piece shorter than half a window joins the previous one), every bound snapped
/// to the nearest of `edges` (the first on a tie).
pub fn time_windows(t: usize, length: usize, edges: &[usize]) -> Vec<[usize; 2]> {
    let mut bounds: Vec<usize> = (0..t).step_by(length.max(1)).chain([t]).collect();
    if bounds.len() > 2 && bounds[bounds.len() - 1] - bounds[bounds.len() - 2] < length / 2 {
        bounds.remove(bounds.len() - 2);
    }
    if !edges.is_empty() {
        bounds = bounds
            .iter()
            .map(|&b| {
                let d = |e: usize| e.abs_diff(b);
                edges
                    .iter()
                    .copied()
                    .fold(edges[0], |best, e| if d(e) < d(best) { e } else { best })
            })
            .collect();
        bounds.sort_unstable();
        bounds.dedup();
    }
    bounds.windows(2).map(|w| [w[0], w[1]]).collect()
}

/// One window's stage files from the session's `session.json` (`ref_stitch.py
/// --session-inputs`): `params.json` (pass 1, with `t0`, `t1`), `pass2.json`, `seeds.json`,
/// `init.json` and `cnmf.json` (with `T`), and `shifts.npy` (frames `t0..t1` of the
/// session's `(T, 2)` float32 shifts).
pub fn window_inputs(
    session: &[u8],
    shifts: &[u8],
    t0: usize,
    t1: usize,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let s: serde_json::Value = serde_json::from_slice(session).map_err(|e| format!("session.json: {e}"))?;
    let stage = |name: &str| -> Result<serde_json::Map<String, serde_json::Value>, String> {
        s[name]
            .as_object()
            .cloned()
            .ok_or_else(|| format!("session.json: no \"{name}\" object"))
    };
    let mut out = Vec::new();
    let mut p1 = stage("pass1")?;
    p1.insert("t0".into(), t0.into());
    p1.insert("t1".into(), t1.into());
    out.push(("params.json".into(), serde_json::to_vec(&p1).unwrap()));
    for name in ["pass2", "seeds", "init", "cnmf"] {
        let mut p = stage(name)?;
        p.insert("T".into(), (t1 - t0).into());
        out.push((format!("{name}.json"), serde_json::to_vec(&p).unwrap()));
    }
    // no shifts: the window's pass 1 estimates them against the template
    if shifts.is_empty() {
        return Ok(out);
    }
    let sh = npy::read_f32(shifts)?;
    if sh.shape.len() != 2 || sh.shape[1] != 2 || sh.shape[0] < t1 || t0 >= t1 {
        return Err(format!("shifts.npy is {:?}; window {t0}-{t1}", sh.shape));
    }
    out.push((
        "shifts.npy".into(),
        npy::write(&[t1 - t0, 2], &sh.data[2 * t0..2 * t1]),
    ));
    Ok(out)
}

/// `stream.window_mean_noise`, window by window: the session's mean image (frame-weighted
/// mean of the windows' pass-2 means) and noise (quadratic mean of the windows' pass-2 noise)
/// for the streaming pass after the stitch, in float64 as the prototype keeps them.
#[derive(Default)]
pub struct MeanNoise {
    mean: Vec<f64>,
    sq: Vec<f64>,
    t: usize,
    n: usize,
}

impl MeanNoise {
    /// Add one window of `t` frames: its pass-2 tiles on the `grid` of a `w`-wide frame.
    pub fn add(&mut self, grid: &[Tile], tiles: &[TileFactors], w: usize, t: usize) {
        if self.n == 0 {
            let p: usize = grid.iter().map(Tile::pixels).sum();
            self.mean = vec![0.0; p];
            self.sq = vec![0.0; p];
        }
        let tf = t as f64;
        for (g, f) in grid.iter().zip(tiles) {
            let mut l = 0;
            for y in g.y0..g.y1 {
                for x in g.x0..g.x1 {
                    let (m, sg) = (f64::from(f.mean[l]), f64::from(f.sigma[l]));
                    self.mean[y * w + x] += tf * m;
                    self.sq[y * w + x] += sg * sg;
                    l += 1;
                }
            }
        }
        self.t += t;
        self.n += 1;
    }

    /// `(ymean, sn)`, or `None` before the first window.
    pub fn finish(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.n == 0 {
            return None;
        }
        let (t, n) = (self.t as f64, self.n as f64);
        let mean = self.mean.iter().map(|&v| v / t).collect();
        let sn = self.sq.iter().map(|&v| (v / n).sqrt()).collect();
        Some((mean, sn))
    }
}

/// The streaming pass's `params.json` for `t` frames from `session.json`'s `stream` object
/// (`ref_dump.py`'s parameters without `T`, `K`).
pub fn stream_params(session: &[u8], t: usize) -> Result<Vec<u8>, String> {
    let s: serde_json::Value = serde_json::from_slice(session).map_err(|e| format!("session.json: {e}"))?;
    let mut p = s["stream"]
        .as_object()
        .cloned()
        .ok_or("session.json: no \"stream\" object")?;
    p.insert("T".into(), t.into());
    Ok(serde_json::to_vec(&p).unwrap())
}

// ---------------------------------------------------------------------------
// Stitch
// ---------------------------------------------------------------------------

/// `ref_stitch.py`'s `stitch.json` (the windows are informational).
#[derive(Debug, Clone, Deserialize)]
pub struct StitchParams {
    pub cos_thres: f64,
    /// Groups whose mean footprints' centroids are closer than this (px) are joined
    /// (`init.unit_merge`'s `min_dist`, the session's `merge_min_dist`); 0, the default, off.
    #[serde(default)]
    pub min_dist: f64,
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(default)]
    pub windows: Vec<[usize; 2]>,
    /// Keep every linked pair in the outputs (`stitch_with`'s `keep_pairs`); unset: kept for
    /// `ref_stitch.py`'s inputs (`stitch.json`), not for a session's window loop.
    #[serde(default)]
    pub keep_pairs: Option<bool>,
}

/// The final units of one window: footprints (any row order) and the peak of each unit's
/// `YrA`.
pub struct Window {
    pub t: [usize; 2],
    pub a: Csr,
    pub bright: Vec<f64>,
}

/// A float64 sparse row in scipy's stored order.
type Row = Vec<(usize, f64)>;

/// `diags(d) @ rows` (`csr_matmat`): each row scaled, reversed, zeros dropped.
fn scale_rows(rows: &[Row], d: &[f64]) -> Vec<Row> {
    rows.iter()
        .zip(d)
        .map(|(r, &s)| {
            r.iter()
                .rev()
                .map(|&(j, x)| (j, s.mul_add(x, 0.0)))
                .filter(|&(_, v)| v != 0.0)
                .collect()
        })
        .collect()
}

/// Float64 sparse rows in one flat, exactly sized store (pixel indices as u32): the stacked
/// window footprints, 12 bytes per entry instead of a `Row` per unit.
struct FlatRows {
    ptr: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<f64>,
}

impl FlatRows {
    fn len(&self) -> usize {
        self.ptr.len() - 1
    }

    /// Row `i` in stored order.
    fn row(&self, i: usize) -> impl DoubleEndedIterator<Item = (usize, f64)> + '_ {
        let r = self.ptr[i]..self.ptr[i + 1];
        self.idx[r.clone()]
            .iter()
            .zip(&self.val[r])
            .map(|(&j, &v)| (j as usize, v))
    }
}

/// `_row_max_one`: `1 / max(row, 0)` per row (1 for an all-zero row).
fn inv_row_max(rows: &[Row]) -> Vec<f64> {
    rows.iter()
        .map(|r| {
            let m = r.iter().fold(0f64, |m, &(_, v)| m.max(v));
            1.0 / if m == 0.0 { 1.0 } else { m }
        })
        .collect()
}

/// `stream._roots`: the root of every node of a `brightest_parent` forest.
pub fn roots(parent: &[usize]) -> Vec<usize> {
    let mut root = parent.to_vec();
    loop {
        let next: Vec<usize> = root.iter().map(|&r| root[r]).collect();
        if next == root {
            return root;
        }
        root = next;
    }
}

pub struct StitchResult {
    pub norms: Vec<f64>,
    pub pairs: Vec<[usize; 2]>, // units of different windows with a non-zero cosine, i < j
    pub pair_cos: Vec<f64>,
    pub edges: Vec<[usize; 2]>, // the pairs above cos_thres
    pub parent: Vec<usize>,
    pub root: Vec<usize>,
    pub group: Vec<usize>,
    pub a: Csr, // (groups, P) float32, rows with maximum 1, indices ascending
    pub bright: Vec<f64>,
    pub n_windows: Vec<usize>,
    pub joined: usize,    // groups the `min_dist` rule joined into another group
    pub kept_pairs: bool, // `pairs`, `pair_cos` and `edges` hold every pair (else they are empty)
    pub n_pairs: usize,
    pub n_edges: usize,
}

/// `M @ A`: each group's mean, members in order (scipy's `csr_matmat` order: one value per
/// pixel per member, each row in reverse order of first appearance, zero sums dropped).
fn group_means(a: &FlatRows, members: &[Vec<usize>], p: usize) -> Vec<Row> {
    let mut mean: Vec<Row> = Vec::with_capacity(members.len());
    let mut sums = vec![0f64; p];
    let mut used = vec![false; p];
    for m in members {
        let v = 1.0 / m.len() as f64;
        let mut order = Vec::new();
        for &i in m {
            for (px, x) in a.row(i) {
                if !used[px] {
                    used[px] = true;
                    sums[px] = 0.0;
                    order.push(px);
                }
                sums[px] = v.mul_add(x, sums[px]);
            }
        }
        let row: Row = order
            .iter()
            .rev()
            .map(|&px| {
                used[px] = false;
                (px, sums[px])
            })
            .filter(|&(_, s)| s != 0.0)
            .collect();
        mean.push(row);
    }
    mean
}

fn members_of(group: &[usize], n_g: usize) -> Vec<Vec<usize>> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_g];
    for (i, &g) in group.iter().enumerate() {
        members[g].push(i);
    }
    members
}

/// `stream._close_groups`: the new label of every group. Groups whose mean footprints (rows
/// scaled to maximum 1) have centroids closer than `min_dist` px are adjacent, and each joins
/// its brightest adjacent group brighter than itself (`brightest_parent` on `bright`, each
/// group's brightest unit), so chains of close groups cannot percolate into neighbouring cells.
fn close_groups(means: &[Row], bright: &[f64], w: usize, min_dist: f64) -> Vec<usize> {
    let cent: Vec<Option<(f64, f64)>> = means
        .iter()
        .map(|r| {
            let (mut m, mut sy, mut sx) = (0f64, 0f64, 0f64);
            for &(px, v) in r {
                m += v;
                sy += v * (px / w) as f64;
                sx += v * (px % w) as f64;
            }
            (m > 0.0).then(|| (sy / m, sx / m))
        })
        .collect();
    // pairs closer than min_dist: grid cells of side min_dist, neighbours in the 3 x 3 block
    let cell = |v: f64| (v / min_dist).floor() as i64;
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
    for (g, c) in cent.iter().enumerate() {
        if let Some((y, x)) = c {
            grid.entry((cell(*y), cell(*x))).or_default().push(g);
        }
    }
    let mut edges = Vec::new();
    for (g, c) in cent.iter().enumerate() {
        let Some((y, x)) = *c else { continue };
        let (cy, cx) = (cell(y), cell(x));
        for dy in -1..=1 {
            for dx in -1..=1 {
                for &h in grid.get(&(cy + dy, cx + dx)).map_or(&[][..], |v| &v[..]) {
                    let (hy, hx) = cent[h].expect("a centroid");
                    if h > g && (y - hy).hypot(x - hx) < min_dist {
                        edges.push([g, h]);
                    }
                }
            }
        }
    }
    let root = roots(&brightest_parent(means.len(), &edges, bright));
    let mut uniq = root.clone();
    uniq.sort_unstable();
    uniq.dedup();
    root.iter()
        .map(|r| uniq.binary_search(r).expect("a root"))
        .collect()
}

/// `stream.stitch` of `windows` (in time order).
pub fn stitch(windows: &[Window], cos_thres: f64, p: usize) -> Result<StitchResult, String> {
    stitch_with(windows, cos_thres, 0.0, 0, p, true)
}

/// `stream.stitch` of `windows` (in time order) with `min_dist` (px; 0 = off) on frames `w`
/// pixels wide. `keep_pairs` keeps every linked pair, its cosine and the edges (for checks
/// against `ref_stitch.py`); they grow with the square of the number of windows (a cell is
/// paired with itself in every other window: 27 M pairs at 8 h), so a session run leaves them
/// out and keeps only their counts.
pub fn stitch_with(
    windows: &[Window],
    cos_thres: f64,
    min_dist: f64,
    w: usize,
    p: usize,
    keep_pairs: bool,
) -> Result<StitchResult, String> {
    // A = _row_max_one(vstack(...).astype(float64)), built row by row into one flat store
    let nnz: usize = windows.iter().map(|u| u.a.indices.len()).sum();
    let mut a = FlatRows {
        ptr: vec![0],
        idx: Vec::with_capacity(nnz),
        val: Vec::with_capacity(nnz),
    };
    if p > u32::MAX as usize {
        return Err(format!("{p} pixels: more than u32 indices hold"));
    }
    let (mut win, mut bright) = (Vec::new(), Vec::new());
    let mut r: Row = Vec::new();
    for (w, u) in windows.iter().enumerate() {
        if u.a.p != p || u.bright.len() != u.a.k {
            return Err(format!(
                "window {w}: {} units over {} pixels with {} brightness values; expected {p} pixels",
                u.a.k,
                u.a.p,
                u.bright.len()
            ));
        }
        for i in 0..u.a.k {
            let (idx, val) = u.a.row(i);
            // `vstack(...).astype(float64)` sorts the rows (scipy canonicalises a stack
            // whose rows are not all sorted, and the windows' rows come in any order)
            r.clear();
            r.extend(idx.iter().zip(val).map(|(&j, &x)| (j, f64::from(x))));
            r.sort_unstable_by_key(|&(j, _)| j);
            if r.windows(2).any(|q| q[0].0 == q[1].0) {
                return Err(format!("window {w}, unit {i}: a pixel appears twice"));
            }
            // `diags(1 / max) @ rows` as `scale_rows` does it: reversed, zeros dropped
            let s = inv_row_max(std::slice::from_ref(&r))[0];
            for &(j, x) in r.iter().rev() {
                let v = s.mul_add(x, 0.0);
                if v != 0.0 {
                    a.idx.push(j as u32);
                    a.val.push(v);
                }
            }
            a.ptr.push(a.idx.len());
            win.push(w);
        }
        bright.extend_from_slice(&u.bright);
    }
    drop(r);
    if let Some(i) = bright.iter().position(|b| !b.is_finite()) {
        return Err(format!("unit {i}: brightness {}", bright[i]));
    }
    let n = a.len();
    // norms = sqrt((A * A).sum(axis=1))
    let increasing = (0..n).all(|i| a.idx[a.ptr[i]..a.ptr[i + 1]].windows(2).all(|w| w[0] < w[1]));
    let norms: Vec<f64> = (0..n)
        .map(|i| {
            let mut sq: Vec<f64> = a.row(i).map(|(_, v)| v * v).filter(|&v| v != 0.0).collect();
            if !increasing {
                sq.reverse();
            }
            let s = match sq.split_first() {
                Some((&first, rest)) => first + pairwise_sum(rest),
                None => 0.0,
            };
            let v = s.sqrt();
            if v == 0.0 {
                1.0
            } else {
                v
            }
        })
        .collect();
    let inv_norm: Vec<f64> = norms.iter().map(|&v| 1.0 / v).collect();
    // An = diags(1 / norms) @ A, row i in An's order (`scale_rows`), made on the fly from A
    let an_row = |i: usize| {
        let s = inv_norm[i];
        a.row(i)
            .rev()
            .map(move |(j, x)| (j, s.mul_add(x, 0.0)))
            .filter(|&(_, v)| v != 0.0)
    };
    // S = triu(An @ An^T, 1): per row i, sums over its pixels in An's row order. An by pixel
    // (its transpose), units ascending per pixel, as flat arrays
    let mut start = vec![0usize; p + 1];
    for i in 0..n {
        for (j, _) in an_row(i) {
            start[j + 1] += 1;
        }
    }
    for j in 0..p {
        start[j + 1] += start[j];
    }
    let mut fill = start.clone();
    let mut px_unit = vec![0u32; start[p]];
    let mut px_val = vec![0f64; start[p]];
    for i in 0..n {
        for (j, v) in an_row(i) {
            px_unit[fill[j]] = i as u32;
            px_val[fill[j]] = v;
            fill[j] += 1;
        }
    }
    drop(fill);
    // each unit joins its brightest brighter linked unit as the links are found; the pairs and
    // edges themselves (quadratic in the number of windows) are kept only when asked for
    let mut parent: Vec<usize> = (0..n).collect();
    let (mut pairs, mut pair_cos, mut edges) = (Vec::new(), Vec::new(), Vec::new());
    let (mut n_pairs, mut n_edges) = (0usize, 0usize);
    let mut acc = vec![0f64; n];
    let mut seen = vec![false; n];
    let mut touched = Vec::new();
    for i in 0..n {
        for (px, v) in an_row(i) {
            for e in start[px]..start[px + 1] {
                let j = px_unit[e] as usize;
                if j > i && win[j] != win[i] {
                    if !seen[j] {
                        seen[j] = true;
                        acc[j] = 0.0;
                        touched.push(j);
                    }
                    acc[j] = v.mul_add(px_val[e], acc[j]);
                }
            }
        }
        touched.sort_unstable();
        for &j in &touched {
            seen[j] = false;
            if acc[j] != 0.0 {
                n_pairs += 1;
                if keep_pairs {
                    pairs.push([i, j]);
                    pair_cos.push(acc[j]);
                }
                if acc[j] > cos_thres {
                    n_edges += 1;
                    if keep_pairs {
                        edges.push([i, j]);
                    }
                    link_brighter(&mut parent, i, j, &bright);
                }
            }
        }
        touched.clear();
    }
    drop((start, px_unit, px_val, acc, seen));
    let root = roots(&parent);
    let compact = |labels: &[usize]| -> Vec<usize> {
        let mut uniq = labels.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        labels
            .iter()
            .map(|r| uniq.binary_search(r).expect("a label"))
            .collect()
    };
    let mut group = compact(&root);
    let mut n_g = group.iter().max().map_or(0, |&g| g + 1);
    let mut joined = 0;
    if min_dist > 0.0 && n_g > 1 {
        if w == 0 || !p.is_multiple_of(w) {
            return Err(format!(
                "min_dist needs the frame width ({w} does not divide {p} pixels)"
            ));
        }
        let members = members_of(&group, n_g);
        let means = group_means(&a, &members, p);
        let bg: Vec<f64> = members
            .iter()
            .map(|m| m.iter().fold(f64::NEG_INFINITY, |b, &i| b.max(bright[i])))
            .collect();
        let comp = close_groups(&scale_rows(&means, &inv_row_max(&means)), &bg, w, min_dist);
        let n_c = comp.iter().max().map_or(0, |&c| c + 1);
        joined = n_g - n_c;
        group = group.iter().map(|&g| comp[g]).collect();
        n_g = n_c;
    }
    let members = members_of(&group, n_g);
    let mean = group_means(&a, &members, p);
    let scaled = scale_rows(&mean, &inv_row_max(&mean));
    let mut out = Csr {
        k: n_g,
        p,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for mut r in scaled {
        r.sort_unstable_by_key(|&(j, _)| j);
        for (j, v) in r {
            out.indices.push(j);
            out.data.push(v as f32);
        }
        out.indptr.push(out.indices.len());
    }
    let mut bright_g = vec![0f64; n_g];
    let mut n_windows = vec![0usize; n_g];
    for (g, m) in members.iter().enumerate() {
        for &i in m {
            bright_g[g] = bright_g[g].max(bright[i]);
        }
        let mut ws: Vec<usize> = m.iter().map(|&i| win[i]).collect();
        ws.dedup(); // members ascending, so their windows are too
        n_windows[g] = ws.len();
    }
    Ok(StitchResult {
        norms,
        pairs,
        pair_cos,
        edges,
        parent,
        root,
        group,
        a: out,
        bright: bright_g,
        n_windows,
        joined,
        kept_pairs: keep_pairs,
        n_pairs,
        n_edges,
    })
}

// ---------------------------------------------------------------------------
// Inputs and outputs
// ---------------------------------------------------------------------------

/// Window `i` of `ref_stitch.py`'s `inputs/` (`win_{i:02}_A_*.npy`, `win_{i:02}_bright.npy`).
pub fn read_window(
    get: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
    i: usize,
    t: [usize; 2],
    p: usize,
) -> Result<Window, String> {
    let mut usizes = |n: &str| -> Result<Vec<usize>, String> {
        Ok(npy::read_i64(&get(&format!("win_{i:02}_A_{n}.npy"))?)?
            .data
            .iter()
            .map(|&v| v as usize)
            .collect())
    };
    let indptr = usizes("indptr")?;
    let indices = usizes("indices")?;
    let a = Csr {
        k: indptr.len().saturating_sub(1),
        p,
        indptr,
        indices,
        data: npy::read_f32(&get(&format!("win_{i:02}_A_data.npy"))?)?.data,
    };
    let bright = npy::read_f64(&get(&format!("win_{i:02}_bright.npy"))?)?.data;
    Ok(Window { t, a, bright })
}

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

fn pair_array(name: &str, v: &[[usize; 2]]) -> (String, Vec<u8>) {
    let flat: Vec<i64> = v.iter().flatten().map(|&x| x as i64).collect();
    (format!("{name}.npy"), npy::write(&[v.len(), 2], &flat))
}

/// The windows in `ref_stitch.py`'s `inputs/` layout (`stitch.json` and `win_{i:02}_*`), so a
/// browser run's own window units can be stitched by Python too.
pub fn window_files(windows: &[Window], p: &StitchParams) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let meta = serde_json::json!({
        "cos_thres": p.cos_thres,
        "H": p.h,
        "W": p.w,
        "windows": windows.iter().map(|u| u.t).collect::<Vec<_>>(),
    });
    out.push(("stitch.json".into(), serde_json::to_vec_pretty(&meta).unwrap()));
    for (i, u) in windows.iter().enumerate() {
        csr_files(&format!("win_{i:02}_A"), &u.a, &mut out);
        out.push((
            format!("win_{i:02}_bright.npy"),
            npy::write(&[u.bright.len()], &u.bright),
        ));
    }
    out
}

/// The files `ref_stitch.py` writes to `ref/`, under the same names.
pub fn outputs(r: &StitchResult, mut summary: serde_json::Value) -> Vec<(String, Vec<u8>)> {
    let mut out = vec![("norms.npy".to_string(), npy::write(&[r.norms.len()], &r.norms))];
    if r.kept_pairs {
        out.push(pair_array("pairs", &r.pairs));
        out.push((
            "pair_cos.npy".into(),
            npy::write(&[r.pair_cos.len()], &r.pair_cos),
        ));
        out.push(pair_array("edges", &r.edges));
    }
    out.extend([
        (
            "parent.npy".into(),
            npy::write(&[r.parent.len()], &ints(&r.parent)),
        ),
        ("root.npy".into(), npy::write(&[r.root.len()], &ints(&r.root))),
        ("group.npy".into(), npy::write(&[r.group.len()], &ints(&r.group))),
        ("bright.npy".into(), npy::write(&[r.bright.len()], &r.bright)),
        (
            "n_windows.npy".into(),
            npy::write(&[r.n_windows.len()], &ints(&r.n_windows)),
        ),
    ]);
    csr_files("A", &r.a, &mut out);
    let mut found = vec![0usize; r.n_windows.iter().max().map_or(1, |&m| m + 1)];
    for &w in &r.n_windows {
        found[w] += 1;
    }
    summary["units_in"] = r.group.len().into();
    summary["pairs"] = r.n_pairs.into();
    summary["edges"] = r.n_edges.into();
    summary["units_out"] = r.a.k.into();
    summary["joined_min_dist"] = r.joined.into();
    summary["found_in_n_windows"] = found.into();
    out.push((
        "summary.json".into(),
        serde_json::to_vec_pretty(&summary).unwrap(),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_noise_weights_frames_and_squares() {
        let grid = vec![Tile {
            y0: 0,
            y1: 1,
            x0: 0,
            x1: 2,
        }];
        let tile = |m: [f32; 2], s: [f32; 2]| TileFactors {
            w: vec![],
            v: vec![],
            s: vec![],
            mean: m.to_vec(),
            sigma: s.to_vec(),
            energy: vec![0.0; 2],
            rank: 0,
            k: 0,
        };
        let mut mn = MeanNoise::default();
        assert!(mn.finish().is_none());
        mn.add(&grid, &[tile([1.0, 2.0], [3.0, 0.0])], 2, 1000);
        mn.add(&grid, &[tile([4.0, 2.0], [4.0, 2.0])], 2, 3000);
        let (m, s) = mn.finish().unwrap();
        assert_eq!(m, vec![(1000.0 + 12000.0) / 4000.0, 2.0]);
        assert_eq!(s, vec![12.5f64.sqrt(), 2.0f64.sqrt()]);
    }

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
    fn time_windows_snap_and_join_the_short_tail() {
        let starts: Vec<usize> = (0..=36).map(|i| i * 1000).collect();
        let w = time_windows(36000, 3000, &starts);
        assert_eq!(w.len(), 12);
        assert_eq!(w[0], [0, 3000]);
        assert_eq!(w[11], [33000, 36000]);
        // 7000 frames in 3000-frame windows: the 1000-frame tail joins the last window
        assert_eq!(time_windows(7000, 3000, &[]), vec![[0, 3000], [3000, 7000]]);
        // bounds snap to the nearest segment start, the first on a tie
        assert_eq!(
            time_windows(6000, 3000, &[0, 2500, 3500, 6000]),
            vec![[0, 2500], [2500, 6000]]
        );
    }

    #[test]
    fn brightest_parent_breaks_ties_by_index() {
        let p = brightest_parent(4, &[[0, 1], [1, 2], [2, 3], [0, 2]], &[1.0, 3.0, 3.0, 0.5]);
        // 1 and 2 tie: 1 counts as brighter; 0 joins 1 (its brightest), 3 joins 2
        assert_eq!(p, vec![1, 1, 1, 2]);
        assert_eq!(roots(&p), vec![1, 1, 1, 1]);
    }

    #[test]
    fn stitch_links_only_across_windows() {
        // window 0: units 0 and 1 overlap (same window: never linked); window 1: unit 2 is a
        // copy of unit 0, brighter
        let w0 = Window {
            t: [0, 10],
            a: csr(&[&[(0, 1.0), (1, 0.5)], &[(1, 0.5), (2, 1.0)]], 4),
            bright: vec![2.0, 1.0],
        };
        let w1 = Window {
            t: [10, 20],
            a: csr(&[&[(0, 2.0), (1, 1.0)]], 4),
            bright: vec![5.0],
        };
        let r = stitch(&[w0, w1], 0.7, 4).unwrap();
        assert_eq!(r.edges, vec![[0, 2]]);
        assert_eq!(r.parent, vec![2, 1, 2]);
        assert_eq!(r.group, vec![1, 0, 1]);
        assert_eq!(r.n_windows, vec![1, 2]);
        assert_eq!(r.bright, vec![1.0, 5.0]);
        assert_eq!(r.a.row(1), (&[0usize, 1][..], &[1.0f32, 0.5][..]));
    }

    #[test]
    fn min_dist_joins_groups_whose_centres_are_too_close_for_two_cells() {
        // 4 x 4 frame: a cell at (0, 0)-(0, 1) in window 0 and a shifted, worse copy at
        // (0, 1)-(1, 1) in window 1 (cosine 0.5: no link); a far unit at (3, 3) in window 1
        let px = |y: usize, x: usize| y * 4 + x;
        let w0 = Window {
            t: [0, 10],
            a: csr(&[&[(px(0, 0), 1.0), (px(0, 1), 1.0)]], 16),
            bright: vec![4.0],
        };
        let w1 = Window {
            t: [10, 20],
            a: csr(&[&[(px(0, 1), 1.0), (px(1, 1), 1.0)], &[(px(3, 3), 1.0)]], 16),
            bright: vec![2.0, 1.0],
        };
        let ws = [w0, w1];
        let off = stitch(&ws, 0.7, 16).unwrap();
        assert_eq!((off.a.k, off.joined), (3, 0));
        // centres (0, 0.5) and (0.5, 1): 0.71 px apart; the far unit is 3.5 px from both
        let on = stitch_with(&ws, 0.7, 1.0, 4, 16, true).unwrap();
        assert_eq!((on.a.k, on.joined), (2, 1));
        assert_eq!(on.group, vec![0, 0, 1]);
        assert_eq!(on.n_windows, vec![2, 1]);
        assert_eq!(on.bright, vec![4.0, 1.0]);
        // the mean of the two footprints, rows scaled to maximum 1
        assert_eq!(
            on.a.row(0),
            (&[px(0, 0), px(0, 1), px(1, 1)][..], &[0.5f32, 1.0, 0.5][..])
        );
        // a threshold below the distance changes nothing
        assert_eq!(stitch_with(&ws, 0.7, 0.7, 4, 16, true).unwrap().group, off.group);
        assert!(stitch_with(&ws, 0.7, 1.0, 0, 16, true).is_err());
    }

    #[test]
    fn stitch_without_pairs_gives_the_same_units() {
        // three windows of the same two cells (one dimmer copy per window) plus a stray unit
        let win = |t: usize, b: [f64; 3]| Window {
            t: [t, t + 10],
            a: csr(
                &[
                    &[(1, 0.5), (0, 1.0)], // rows in any order
                    &[(5, 1.0), (6, 0.8)],
                    &[(10, 1.0), (11, 0.2)],
                ],
                16,
            ),
            bright: b.to_vec(),
        };
        let ws = [
            win(0, [2.0, 1.0, 0.5]),
            win(10, [3.0, 0.5, 0.4]),
            win(20, [1.0, 4.0, 0.3]),
        ];
        let full = stitch_with(&ws, 0.7, 1.0, 4, 16, true).unwrap();
        let lean = stitch_with(&ws, 0.7, 1.0, 4, 16, false).unwrap();
        assert!(lean.pairs.is_empty() && lean.pair_cos.is_empty() && lean.edges.is_empty());
        assert_eq!((lean.n_pairs, lean.n_edges), (full.pairs.len(), full.edges.len()));
        assert_eq!(lean.parent, full.parent);
        assert_eq!(
            lean.parent,
            brightest_parent(9, &full.edges, &[2.0, 1.0, 0.5, 3.0, 0.5, 0.4, 1.0, 4.0, 0.3])
        );
        assert_eq!(lean.group, full.group);
        assert_eq!(
            (&lean.a.indptr, &lean.a.indices, &lean.a.data),
            (&full.a.indptr, &full.a.indices, &full.a.data)
        );
        assert_eq!(lean.bright, full.bright);
        let names = |r: &StitchResult| -> Vec<String> {
            outputs(r, serde_json::json!({}))
                .into_iter()
                .map(|(n, _)| n)
                .collect()
        };
        assert!(names(&full).contains(&"pairs.npy".to_string()));
        assert!(!names(&lean).contains(&"pairs.npy".to_string()));
    }
}
