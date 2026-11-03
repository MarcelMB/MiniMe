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

use crate::corr::brightest_parent;
use crate::gmm::pairwise_sum;
use crate::init::Csr;
use crate::npy;
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

// ---------------------------------------------------------------------------
// Stitch
// ---------------------------------------------------------------------------

/// `ref_stitch.py`'s `stitch.json` (the windows are informational).
#[derive(Debug, Clone, Deserialize)]
pub struct StitchParams {
    pub cos_thres: f64,
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(default)]
    pub windows: Vec<[usize; 2]>,
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
}

/// `stream.stitch` of `windows` (in time order).
pub fn stitch(windows: &[Window], cos_thres: f64, p: usize) -> Result<StitchResult, String> {
    let mut rows: Vec<Row> = Vec::new();
    let (mut win, mut bright) = (Vec::new(), Vec::new());
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
            let mut r: Row = idx.iter().zip(val).map(|(&j, &x)| (j, f64::from(x))).collect();
            r.sort_unstable_by_key(|&(j, _)| j);
            if r.windows(2).any(|q| q[0].0 == q[1].0) {
                return Err(format!("window {w}, unit {i}: a pixel appears twice"));
            }
            rows.push(r);
            win.push(w);
        }
        bright.extend_from_slice(&u.bright);
    }
    if let Some(i) = bright.iter().position(|b| !b.is_finite()) {
        return Err(format!("unit {i}: brightness {}", bright[i]));
    }
    let n = rows.len();
    // A = _row_max_one(vstack)
    let a = scale_rows(&rows, &inv_row_max(&rows));
    // norms = sqrt((A * A).sum(axis=1))
    let increasing = a.iter().all(|r| r.windows(2).all(|w| w[0].0 < w[1].0));
    let norms: Vec<f64> = a
        .iter()
        .map(|r| {
            let mut sq: Vec<f64> = r.iter().map(|&(_, v)| v * v).filter(|&v| v != 0.0).collect();
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
    let an = scale_rows(&a, &inv_norm);
    // S = triu(An @ An^T, 1): per row i, sums over its pixels in An's row order
    let mut by_px: Vec<Vec<(usize, f64)>> = vec![Vec::new(); p];
    for (i, r) in an.iter().enumerate() {
        for &(j, v) in r {
            by_px[j].push((i, v));
        }
    }
    let (mut pairs, mut pair_cos, mut edges) = (Vec::new(), Vec::new(), Vec::new());
    let mut acc = vec![0f64; n];
    let mut seen = vec![false; n];
    for (i, r) in an.iter().enumerate() {
        let mut touched = Vec::new();
        for &(px, v) in r {
            for &(j, x) in &by_px[px] {
                if j > i && win[j] != win[i] {
                    if !seen[j] {
                        seen[j] = true;
                        acc[j] = 0.0;
                        touched.push(j);
                    }
                    acc[j] = v.mul_add(x, acc[j]);
                }
            }
        }
        touched.sort_unstable();
        for j in touched {
            seen[j] = false;
            if acc[j] != 0.0 {
                pairs.push([i, j]);
                pair_cos.push(acc[j]);
                if acc[j] > cos_thres {
                    edges.push([i, j]);
                }
            }
        }
    }
    let parent = brightest_parent(n, &edges, &bright);
    let root = roots(&parent);
    let mut uniq = root.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let group: Vec<usize> = root
        .iter()
        .map(|r| uniq.binary_search(r).expect("a root"))
        .collect();
    let n_g = uniq.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_g];
    for (i, &g) in group.iter().enumerate() {
        members[g].push(i);
    }
    // M @ A: each group's mean, members in order (one value per pixel per member)
    let mut mean: Vec<Row> = Vec::with_capacity(n_g);
    let mut sums = vec![0f64; p];
    let mut used = vec![false; p];
    for m in &members {
        let v = 1.0 / m.len() as f64;
        let mut order = Vec::new();
        for &i in m {
            for &(px, x) in &a[i] {
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
    let mut out = vec![
        ("norms.npy".to_string(), npy::write(&[r.norms.len()], &r.norms)),
        pair_array("pairs", &r.pairs),
        (
            "pair_cos.npy".into(),
            npy::write(&[r.pair_cos.len()], &r.pair_cos),
        ),
        pair_array("edges", &r.edges),
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
    ];
    csr_files("A", &r.a, &mut out);
    let mut found = vec![0usize; r.n_windows.iter().max().map_or(1, |&m| m + 1)];
    for &w in &r.n_windows {
        found[w] += 1;
    }
    summary["units_in"] = r.group.len().into();
    summary["pairs"] = r.pairs.len().into();
    summary["edges"] = r.edges.len().into();
    summary["units_out"] = r.a.k.into();
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
}
