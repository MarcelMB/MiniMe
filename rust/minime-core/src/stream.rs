//! One streaming pass with fixed footprints (`experiments/streaming/stream.py`):
//! per-frame least-squares traces on `[A; b]` plus per-segment running sums, then the
//! footprint update from those sums (`spatial_from_stats`) and the unit merge
//! (`merge_from_stats`). State is O(nnz(support) + K^2 + P) per segment, whatever `T`.

use crate::butter::{butter_lowpass_sos, SosFilter};
use crate::linalg::NormalSolver;
use crate::nnqp::solve_nnqp;
use serde::Deserialize;

/// The settings `ref_dump.py` writes to `inputs/params.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub segment_frames: usize,
    pub median_ksize: usize,
    pub tophat_wnd: usize,
    pub motion_before_tophat: bool,
    /// "template" (the inputs' `fill.npy` where a frame has no data) or "zero" (missing in
    /// inputs written before Nov 2026)
    #[serde(default = "zero_fill")]
    pub motion_fill: String,
    pub dl_wnd: usize,
    pub sparse_snr: f64,
    pub size_thres: [Option<f64>; 2],
    pub noise_freq: f64,
    pub loop_merge_corr: f64,
}

fn zero_fill() -> String {
    "zero".into()
}

/// CSR matrix, `f64` values.
#[derive(Debug, Clone, Default)]
pub struct Csr {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl Csr {
    pub fn row(&self, r: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.indptr[r], self.indptr[r + 1]);
        (&self.indices[a..b], &self.data[a..b])
    }

    /// `(K, K)` dense `self @ self.T`.
    pub fn gram(&self) -> Vec<f64> {
        let k = self.nrows;
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.ncols];
        for r in 0..k {
            let (ix, v) = self.row(r);
            for (&c, &x) in ix.iter().zip(v) {
                by_col[c].push((r, x));
            }
        }
        let mut g = vec![0.0; k * k];
        for col in &by_col {
            for &(i, a) in col {
                for &(j, b) in col {
                    g[i * k + j] += a * b;
                }
            }
        }
        g
    }
}

/// Frames' least-squares `(c_t, f_t)` on `X = [A; b]` with one factorisation of `X X^T`.
pub struct TraceModel {
    pub k: usize,
    pub b: Vec<f64>,
    pub x: Csr,
    pub solver: NormalSolver,
}

fn symmetrize(g: &mut [f64], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            let m = 0.5 * (g[i * n + j] + g[j * n + i]);
            g[i * n + j] = m;
            g[j * n + i] = m;
        }
    }
}

impl TraceModel {
    /// `b = clip(ymean - A^T cbar, 0)` with `cbar` the least-squares fit of the mean image.
    pub fn build(a: &Csr, ymean: &[f64]) -> Self {
        let k = a.nrows;
        let mut g = a.gram();
        symmetrize(&mut g, k);
        let mut cbar: Vec<f64> = (0..k)
            .map(|r| {
                let (ix, v) = a.row(r);
                ix.iter().zip(v).map(|(&c, &x)| x * ymean[c]).sum()
            })
            .collect();
        // `certified`: the same solver as `NormalSolver::new` without its O(K^3) eigensolve when
        // a bound shows Cholesky (16 s at K = 827)
        NormalSolver::certified(&g, k).solve(&mut cbar);
        let mut fit = vec![0.0; a.ncols];
        for (r, &cb) in cbar.iter().enumerate() {
            let (ix, v) = a.row(r);
            for (&c, &x) in ix.iter().zip(v) {
                fit[c] += x * cb;
            }
        }
        let b: Vec<f64> = ymean.iter().zip(&fit).map(|(&y, &f)| (y - f).max(0.0)).collect();
        let mut x = a.clone();
        x.nrows = k + 1;
        for (p, &v) in b.iter().enumerate() {
            if v != 0.0 {
                x.indices.push(p);
                x.data.push(v);
            }
        }
        x.indptr.push(x.indices.len());
        let mut gx = x.gram();
        symmetrize(&mut gx, k + 1);
        let solver = NormalSolver::certified(&gx, k + 1);
        Self { k, b, x, solver }
    }

    /// Unclipped `(c (K,), f)` of one frame.
    pub fn solve(&self, y: &[f32], sol: &mut Vec<f64>) {
        sol.clear();
        for r in 0..self.x.nrows {
            let (ix, v) = self.x.row(r);
            let mut s = 0.0;
            for (&c, &a) in ix.iter().zip(v) {
                s += a * f64::from(y[c]);
            }
            sol.push(s);
        }
        self.solver.solve(sol);
    }
}

/// The (pixel, unit) pairs `cnmf.update_spatial` solves for: dilated supports, sorted by
/// pixel then unit.
pub struct Support {
    pub active: Vec<usize>,
    pub indptr: Vec<usize>,
    pub unit: Vec<usize>,
    pub pix: Vec<usize>,
}

impl Support {
    pub fn build(a: &Csr, h: usize, w: usize, dl_wnd: usize) -> Self {
        let r = dl_wnd as isize;
        let disk: Vec<(isize, isize)> = (-r..=r)
            .flat_map(|dy| (-r..=r).map(move |dx| (dy, dx)))
            .filter(|&(dy, dx)| dx * dx + dy * dy <= r * r)
            .collect();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut mark = vec![u32::MAX; h * w];
        for u in 0..a.nrows {
            let (ix, v) = a.row(u);
            let mut touched = Vec::new();
            for (&p, &x) in ix.iter().zip(v) {
                if x <= 0.0 {
                    continue;
                }
                let (py, px) = ((p / w) as isize, (p % w) as isize);
                for &(dy, dx) in &disk {
                    let (yy, xx) = (py + dy, px + dx);
                    if yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize {
                        let q = yy as usize * w + xx as usize;
                        if mark[q] != u as u32 {
                            mark[q] = u as u32;
                            touched.push(q);
                        }
                    }
                }
            }
            pairs.extend(touched.into_iter().map(|q| (q, u)));
        }
        pairs.sort_unstable();
        let mut active = Vec::new();
        let mut indptr = vec![0];
        for (i, &(p, _)) in pairs.iter().enumerate() {
            if i == 0 || pairs[i - 1].0 != p {
                if i > 0 {
                    indptr.push(i);
                }
                active.push(p);
            }
        }
        indptr.push(pairs.len());
        if pairs.is_empty() {
            indptr = vec![0];
        }
        Self {
            active,
            indptr,
            unit: pairs.iter().map(|&(_, u)| u).collect(),
            pix: pairs.iter().map(|&(p, _)| p).collect(),
        }
    }
}

/// Running sums over the frames of one segment (`stream.SegmentStats`).
#[derive(Clone)]
pub struct Stats {
    pub n: usize,
    pub l: Vec<f64>,
    pub m: Vec<f64>,
    pub csum: Vec<f64>,
    pub ysum: Vec<f64>,
    pub dsq: Vec<f64>,
    pub mlp: Vec<f64>,
    pub lsum: Vec<f64>,
}

impl Stats {
    pub fn zeros(nnz: usize, k: usize, p: usize) -> Self {
        Self {
            n: 0,
            l: vec![0.0; nnz],
            m: vec![0.0; k * k],
            csum: vec![0.0; k],
            ysum: vec![0.0; p],
            dsq: vec![0.0; p],
            mlp: vec![0.0; k * k],
            lsum: vec![0.0; k],
        }
    }

    pub fn add(&mut self, o: &Stats) {
        self.n += o.n;
        for (a, b) in [
            (&mut self.l, &o.l),
            (&mut self.m, &o.m),
            (&mut self.csum, &o.csum),
            (&mut self.ysum, &o.ysum),
            (&mut self.dsq, &o.dsq),
            (&mut self.mlp, &o.mlp),
            (&mut self.lsum, &o.lsum),
        ] {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
    }

    pub fn nbytes(&self) -> usize {
        8 * (self.l.len() + 2 * self.m.len() + self.csum.len() * 2 + self.ysum.len() * 2)
    }
}

/// The pass's state: model, support, the sums and the traces not yet taken.
///
/// Only the running segment's sums and the total of the finished segments are kept, so the
/// state does not grow with T; the segments are added in order, as `stream.total` does.
pub struct StreamPass {
    pub params: Params,
    pub model: TraceModel,
    pub support: Support,
    /// sums of the finished segments
    pub done: Stats,
    /// sums of the running segment
    pub seg: Stats,
    pub lowpass: SosFilter,
    /// (n, K) frame-major clipped traces (the YrA analogue) since the last `take_c`
    pub c: Vec<f32>,
    pub f: Vec<f32>,
    pub t: usize,
    prev: Option<Vec<f32>>,
    sol: Vec<f64>,
    cc: Vec<f64>,
    lp: Vec<f64>,
}

impl StreamPass {
    pub fn new(params: Params, a: &Csr, ymean: &[f64]) -> Self {
        let k = a.nrows;
        let p = params.h * params.w;
        let model = TraceModel::build(a, ymean);
        let support = Support::build(a, params.h, params.w, params.dl_wnd);
        let done = Stats::zeros(support.pix.len(), k, p);
        let lowpass = SosFilter::new(butter_lowpass_sos(4, 2.0 * params.noise_freq), k);
        Self {
            c: Vec::new(),
            f: Vec::with_capacity(params.t),
            params,
            model,
            support,
            seg: done.clone(),
            done,
            lowpass,
            t: 0,
            prev: None,
            sol: Vec::with_capacity(k + 1),
            cc: vec![0.0; k],
            lp: vec![0.0; k],
        }
    }

    /// Fold in one preprocessed frame (`P` values).
    pub fn push(&mut self, y: &[f32]) {
        let k = self.model.k;
        self.model.solve(y, &mut self.sol);
        for i in 0..k {
            self.cc[i] = self.sol[i].max(0.0);
        }
        self.fold(y, self.sol[k] as f32);
    }

    /// Fold in one preprocessed frame whose least squares was solved elsewhere (a pool worker
    /// running `TraceModel::solve` on the same model): `cc` the clipped traces `max(c, 0)` in
    /// float64, `f` the background trace as float32. The same arithmetic as `push`, which
    /// solves and then calls `fold`.
    pub fn push_solved(&mut self, y: &[f32], cc: &[f64], f: f32) {
        self.cc.copy_from_slice(cc);
        self.fold(y, f);
    }

    /// The frame's traces are in `self.cc`: store them and add the frame to the running sums.
    fn fold(&mut self, y: &[f32], f: f32) {
        let k = self.model.k;
        self.c.extend(self.cc.iter().map(|&v| v as f32));
        self.f.push(f);
        if self.t > 0 && self.t.is_multiple_of(self.params.segment_frames) {
            self.done.add(&self.seg);
            self.seg = Stats::zeros(self.seg.l.len(), k, self.seg.ysum.len());
        }
        let st = &mut self.seg;
        st.n += 1;
        for i in 0..k {
            let ci = self.cc[i];
            if ci != 0.0 {
                let row = &mut st.m[i * k..(i + 1) * k];
                for (mj, &cj) in row.iter_mut().zip(&self.cc) {
                    *mj += ci * cj;
                }
            }
            st.csum[i] += ci;
        }
        self.lp.copy_from_slice(&self.cc);
        self.lowpass.step(&mut self.lp);
        for i in 0..k {
            let li = self.lp[i];
            let row = &mut st.mlp[i * k..(i + 1) * k];
            for (mj, &lj) in row.iter_mut().zip(&self.lp) {
                *mj += li * lj;
            }
            st.lsum[i] += li;
        }
        for (s, &v) in st.ysum.iter_mut().zip(y) {
            *s += f64::from(v);
        }
        if let Some(prev) = &self.prev {
            for ((s, &v), &u) in st.dsq.iter_mut().zip(y).zip(prev) {
                let d = f64::from(v - u);
                *s += d * d;
            }
        }
        let sup = &self.support;
        for ((l, &p), &u) in st.l.iter_mut().zip(&sup.pix).zip(&sup.unit) {
            *l += f64::from(y[p]) * self.cc[u];
        }
        match &mut self.prev {
            Some(prev) => prev.copy_from_slice(y),
            None => self.prev = Some(y.to_vec()),
        }
        self.t += 1;
    }

    pub fn total(&self) -> Stats {
        let mut tot = self.done.clone();
        tot.add(&self.seg);
        tot
    }

    /// The traces since the last call, (n, K) frame-major; the caller stores them.
    pub fn take_c(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.c)
    }

    /// Bytes a browser keeps between frames (sums, support, model, filter state).
    pub fn state_bytes(&self) -> usize {
        let sup = 8 * (self.support.active.len() + self.support.indptr.len() + 2 * self.support.pix.len());
        let model = 8 * self.model.b.len() + 16 * self.model.x.data.len() + 8 * (self.model.k + 1).pow(2);
        self.done.nbytes() + self.seg.nbytes() + sup + model + 32 * self.model.k
    }
}

/// The share of `StreamPass::fold`'s sums one fold lane keeps: rows `rows` of the K x K sums
/// (and of `csum`, `lsum`), pixels `pix` of `ysum` and `dsq`, and the support pairs `pairs` (the
/// pairs on those pixels: the support is sorted by pixel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneRange {
    pub rows: [usize; 2],
    pub pix: [usize; 2],
    pub pairs: [usize; 2],
}

/// The fold split into `lanes` shares of about equal work: rows evenly (each row costs `2 K`), pixels
/// by their cost (`ysum`, `dsq` and one multiply-add per support pair on the pixel).
pub fn lane_ranges(k: usize, p: usize, sup: &Support, lanes: usize) -> Vec<LaneRange> {
    let lanes = lanes.max(1);
    // pairs per pixel, then cumulative cost
    let mut per = vec![0usize; p];
    for &q in &sup.pix {
        per[q] += 1;
    }
    let total: usize = per.iter().map(|&n| 2 + n).sum();
    let mut pix_bounds = vec![0usize];
    let mut acc = 0usize;
    for (q, &n) in per.iter().enumerate() {
        acc += 2 + n;
        while pix_bounds.len() < lanes && acc * lanes >= total * pix_bounds.len() {
            pix_bounds.push(q + 1);
        }
    }
    while pix_bounds.len() <= lanes {
        pix_bounds.push(p);
    }
    *pix_bounds.last_mut().unwrap() = p;
    let first_pair = |q: usize| sup.pix.partition_point(|&x| x < q);
    (0..lanes)
        .map(|j| LaneRange {
            rows: [k * j / lanes, k * (j + 1) / lanes],
            pix: [pix_bounds[j], pix_bounds[j + 1]],
            pairs: [first_pair(pix_bounds[j]), first_pair(pix_bounds[j + 1])],
        })
        .collect()
}

/// One lane of the fold (`lane_ranges`): the same per-element sums as `StreamPass::fold`, in the
/// same frame order, on its rows, pixels and pairs, with its own copy of the low-pass filter (run on
/// all K traces, as `fold` runs it) and of the previous frame on its pixels. Lanes put together
/// (`StreamPass::put_lane`) are the one-thread sums bit for bit, however many there are.
pub struct FoldLane {
    pub range: LaneRange,
    k: usize,
    segment_frames: usize,
    pair_pix: Vec<usize>,
    pair_unit: Vec<usize>,
    lowpass: SosFilter,
    /// sums of the finished segments and of the running one, on this lane's share
    pub done: Stats,
    pub seg: Stats,
    prev: Option<Vec<f32>>,
    lp: Vec<f64>,
    pub t: usize,
}

impl FoldLane {
    pub fn new(params: &Params, sup: &Support, k: usize, range: LaneRange) -> Self {
        let rows = range.rows[1] - range.rows[0];
        let zeros = Stats {
            n: 0,
            l: vec![0.0; range.pairs[1] - range.pairs[0]],
            m: vec![0.0; rows * k],
            csum: vec![0.0; rows],
            ysum: vec![0.0; range.pix[1] - range.pix[0]],
            dsq: vec![0.0; range.pix[1] - range.pix[0]],
            mlp: vec![0.0; rows * k],
            lsum: vec![0.0; rows],
        };
        Self {
            range,
            k,
            segment_frames: params.segment_frames,
            pair_pix: sup.pix[range.pairs[0]..range.pairs[1]].to_vec(),
            pair_unit: sup.unit[range.pairs[0]..range.pairs[1]].to_vec(),
            lowpass: SosFilter::new(butter_lowpass_sos(4, 2.0 * params.noise_freq), k),
            seg: zeros.clone(),
            done: zeros,
            prev: None,
            lp: vec![0.0; k],
            t: 0,
        }
    }

    /// Fold in the next frame: `y` its pixels `pix[0]..pix[1]`, `cc` all K clipped traces.
    pub fn fold(&mut self, y: &[f32], cc: &[f64]) {
        let k = self.k;
        let [r0, r1] = self.range.rows;
        let p0 = self.range.pix[0];
        if self.t > 0 && self.t.is_multiple_of(self.segment_frames) {
            self.done.add(&self.seg);
            let z = |v: &Vec<f64>| vec![0.0; v.len()];
            let s = &self.seg;
            self.seg = Stats {
                n: 0,
                l: z(&s.l),
                m: z(&s.m),
                csum: z(&s.csum),
                ysum: z(&s.ysum),
                dsq: z(&s.dsq),
                mlp: z(&s.mlp),
                lsum: z(&s.lsum),
            };
        }
        let st = &mut self.seg;
        st.n += 1;
        for i in r0..r1 {
            let ci = cc[i];
            if ci != 0.0 {
                let row = &mut st.m[(i - r0) * k..(i - r0 + 1) * k];
                for (mj, &cj) in row.iter_mut().zip(cc) {
                    *mj += ci * cj;
                }
            }
            st.csum[i - r0] += ci;
        }
        self.lp.copy_from_slice(cc);
        self.lowpass.step(&mut self.lp);
        for i in r0..r1 {
            let li = self.lp[i];
            let row = &mut st.mlp[(i - r0) * k..(i - r0 + 1) * k];
            for (mj, &lj) in row.iter_mut().zip(&self.lp) {
                *mj += li * lj;
            }
            st.lsum[i - r0] += li;
        }
        for (s, &v) in st.ysum.iter_mut().zip(y) {
            *s += f64::from(v);
        }
        if let Some(prev) = &self.prev {
            for ((s, &v), &u) in st.dsq.iter_mut().zip(y).zip(prev) {
                let d = f64::from(v - u);
                *s += d * d;
            }
        }
        for ((l, &q), &u) in st.l.iter_mut().zip(&self.pair_pix).zip(&self.pair_unit) {
            *l += f64::from(y[q - p0]) * cc[u];
        }
        match &mut self.prev {
            Some(prev) => prev.copy_from_slice(y),
            None => self.prev = Some(y.to_vec()),
        }
        self.t += 1;
    }
}

impl Stats {
    /// One flat float64 array: `n`, then `l`, `m`, `csum`, `ysum`, `dsq`, `mlp`, `lsum`.
    pub fn to_flat(&self) -> Vec<f64> {
        let mut out = vec![self.n as f64];
        for a in [
            &self.l, &self.m, &self.csum, &self.ysum, &self.dsq, &self.mlp, &self.lsum,
        ] {
            out.extend_from_slice(a);
        }
        out
    }
}

impl StreamPass {
    /// Store one frame's traces without folding it into the sums (a pool's lanes fold it):
    /// `cc` the clipped traces, `f` the background trace.
    pub fn push_traces(&mut self, cc: &[f64], f: f32) {
        self.c.extend(cc.iter().map(|&v| v as f32));
        self.f.push(f);
        self.t += 1;
    }

    /// Put a lane's sums (`FoldLane::done`, `seg` as `Stats::to_flat`) in their place in this pass's.
    pub fn put_lane(&mut self, r: LaneRange, done: &[f64], seg: &[f64]) -> Result<(), String> {
        let k = self.model.k;
        let rows = r.rows[1] - r.rows[0];
        let (np, nl) = (r.pix[1] - r.pix[0], r.pairs[1] - r.pairs[0]);
        let want = 1 + nl + 2 * rows * k + 2 * rows + 2 * np;
        if done.len() != want
            || seg.len() != want
            || r.rows[1] > k
            || r.pix[1] > self.done.ysum.len()
            || r.pairs[1] > self.done.l.len()
        {
            return Err(format!(
                "a fold lane's sums hold {} values, expected {want}",
                done.len()
            ));
        }
        for (dst, src) in [(&mut self.done, done), (&mut self.seg, seg)] {
            dst.n = src[0] as usize;
            let mut at = 1;
            let mut take = |n: usize| {
                let s = &src[at..at + n];
                at += n;
                s
            };
            dst.l[r.pairs[0]..r.pairs[1]].copy_from_slice(take(nl));
            dst.m[r.rows[0] * k..r.rows[1] * k].copy_from_slice(take(rows * k));
            dst.csum[r.rows[0]..r.rows[1]].copy_from_slice(take(rows));
            dst.ysum[r.pix[0]..r.pix[1]].copy_from_slice(take(np));
            dst.dsq[r.pix[0]..r.pix[1]].copy_from_slice(take(np));
            dst.mlp[r.rows[0] * k..r.rows[1] * k].copy_from_slice(take(rows * k));
            dst.lsum[r.rows[0]..r.rows[1]].copy_from_slice(take(rows));
        }
        Ok(())
    }
}

/// `cnmf._size_mask`: strict `low < nnz < high`, empty units dropped.
fn size_ok(nnz: usize, thres: [Option<f64>; 2]) -> bool {
    let n = nnz as f64;
    nnz > 0 && thres[0].is_none_or(|lo| lo == 0.0 || n > lo) && thres[1].is_none_or(|hi| hi == 0.0 || n < hi)
}

/// Result of `spatial_from_stats`: kept units' footprints (rows scaled to maximum 1,
/// float32 values), the kept-unit mask over the input units and the row maxima.
pub struct SpatialUpdate {
    pub a: Csr,
    pub mask: Vec<bool>,
    pub norm_fac: Vec<f32>,
}

/// `cnmf.update_spatial` from the sums: same centring, unit-norm scaling, penalty
/// `sqrt(n) * sparse_snr * sn` and post-processing, no frame read.
pub fn spatial_from_stats(
    k: usize,
    p_total: usize,
    sup: &Support,
    st: &Stats,
    sn: &[f64],
    sparse_snr: f64,
    size_thres: [Option<f64>; 2],
) -> Result<SpatialUpdate, String> {
    let n = st.n as f64;
    let cbar: Vec<f64> = st.csum.iter().map(|&c| c / n).collect();
    let ybar: Vec<f64> = st.ysum.iter().map(|&y| y / n).collect();
    let mut cov = st.m.clone();
    for i in 0..k {
        for j in 0..k {
            cov[i * k + j] -= n * (cbar[i] * cbar[j]);
        }
    }
    let scale: Vec<f64> = (0..k)
        .map(|i| {
            let s = cov[i * k + i].max(0.0).sqrt();
            if s == 0.0 {
                1.0
            } else {
                s
            }
        })
        .collect();
    let g: Vec<f64> = (0..k * k)
        .map(|ij| cov[ij] / (scale[ij / k] * scale[ij % k]))
        .collect();
    let bvals: Vec<f64> = (0..sup.pix.len())
        .map(|j| {
            let (p, u) = (sup.pix[j], sup.unit[j]);
            (st.l[j] - n * ybar[p] * cbar[u]) / scale[u]
        })
        .collect();
    let lam: Vec<f64> = sup
        .active
        .iter()
        .map(|&p| n.sqrt() * sparse_snr * sn[p])
        .collect();
    let values = solve_nnqp(&g, k, &bvals, &lam, &sup.indptr, &sup.unit);
    // rows per unit, pixels ascending (pairs are sorted by pixel)
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k];
    for (j, &v) in values.iter().enumerate() {
        let v = v / scale[sup.unit[j]];
        if v != 0.0 {
            rows[sup.unit[j]].push((sup.pix[j], v));
        }
    }
    let mask: Vec<bool> = rows.iter().map(|r| size_ok(r.len(), size_thres)).collect();
    if !mask.iter().any(|&m| m) {
        return Err("all units dropped by size_thres".into());
    }
    let mut a = Csr {
        nrows: 0,
        ncols: p_total,
        indptr: vec![0],
        ..Default::default()
    };
    let mut norm_fac = Vec::new();
    for (r, keep) in rows.iter().zip(&mask) {
        if !keep {
            continue;
        }
        let mx = r.iter().fold(f64::NEG_INFINITY, |m, &(_, v)| m.max(v));
        let inv = 1.0 / mx;
        for &(p, v) in r {
            a.indices.push(p);
            a.data.push(f64::from((inv * v) as f32));
        }
        a.indptr.push(a.indices.len());
        a.nrows += 1;
        norm_fac.push(mx as f32);
    }
    Ok(SpatialUpdate { a, mask, norm_fac })
}

/// Connected components of an undirected graph (adjacency lists), labelled by smallest
/// member; returns groups in order of their smallest member.
fn components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut seen = vec![false; n];
    let mut groups = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let mut g = vec![s];
        seen[s] = true;
        let mut i = 0;
        while i < g.len() {
            for &v in &adj[g[i]] {
                if !seen[v] {
                    seen[v] = true;
                    g.push(v);
                }
            }
            i += 1;
        }
        g.sort_unstable();
        groups.push(g);
    }
    groups
}

/// The loop's unit merge on the sums: overlapping kept units whose low-passed traces
/// correlate above `thres_corr` are averaged (`merge_from_stats(lowpassed=True)`).
pub fn merge_from_stats(a: &Csr, st: &Stats, k: usize, mask: &[bool], thres_corr: f64) -> Csr {
    let n = st.n as f64;
    let kept: Vec<usize> = (0..k).filter(|&i| mask[i]).collect();
    let lbar: Vec<f64> = st.lsum.iter().map(|&l| l / n).collect();
    let kk = kept.len();
    let cov = |i: usize, j: usize| st.mlp[kept[i] * k + kept[j]] - n * (lbar[kept[i]] * lbar[kept[j]]);
    let sd: Vec<f64> = (0..kk).map(|i| cov(i, i).max(1e-12).sqrt()).collect();
    // overlap: some shared pixel with a non-zero float32 product
    let mut by_pix: Vec<Vec<(usize, f32)>> = vec![Vec::new(); a.ncols];
    for r in 0..a.nrows {
        let (ix, v) = a.row(r);
        for (&p, &x) in ix.iter().zip(v) {
            by_pix[p].push((r, x as f32));
        }
    }
    let mut overlap = vec![false; kk * kk];
    for col in &by_pix {
        for &(i, x) in col {
            for &(j, y) in col {
                if i != j && x * y > 0.0 {
                    overlap[i * kk + j] = true;
                }
            }
        }
    }
    let mut adj = vec![Vec::new(); kk];
    for i in 0..kk {
        for j in 0..kk {
            // scipy's connected_components(directed=False): an edge either way links
            if overlap[i * kk + j] && cov(i, j) / (sd[i] * sd[j]) > thres_corr {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut out = Csr {
        nrows: 0,
        ncols: a.ncols,
        indptr: vec![0],
        ..Default::default()
    };
    let mut acc = vec![0.0f64; a.ncols];
    let mut hit = vec![false; a.ncols];
    for g in components(&adj) {
        let wgt = 1.0 / g.len() as f64;
        let mut cols = Vec::new();
        for &r in &g {
            let (ix, v) = a.row(r);
            for (&p, &x) in ix.iter().zip(v) {
                if !hit[p] {
                    hit[p] = true;
                    cols.push(p);
                }
                acc[p] += wgt * x;
            }
        }
        cols.sort_unstable();
        for p in cols {
            out.indices.push(p);
            out.data.push(f64::from(acc[p] as f32));
            acc[p] = 0.0;
            hit[p] = false;
        }
        out.indptr.push(out.indices.len());
        out.nrows += 1;
    }
    out
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn lcg(seed: &mut u32) -> f64 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(*seed >> 8) / f64::from(1u32 << 24)
    }

    /// Frames solved on one `TraceModel` and folded in by `push_solved` (what a pool worker
    /// and the accumulating engine do) give the sums, traces and low-pass state of `push`.
    #[test]
    fn solved_frames_fold_in_exactly_as_push_does() {
        let (h, w, t, k) = (10usize, 12usize, 47usize, 4usize);
        let p = h * w;
        let params = Params {
            h,
            w,
            t,
            segment_frames: 20, // two segment boundaries inside the pass
            median_ksize: 3,
            tophat_wnd: 3,
            motion_before_tophat: true,
            motion_fill: "zero".into(),
            dl_wnd: 2,
            sparse_snr: 1.0,
            size_thres: [None, None],
            noise_freq: 0.06,
            loop_merge_corr: 0.9,
        };
        let mut seed = 7u32;
        // k blobs of a few pixels each
        let (mut indptr, mut indices, mut data) = (vec![0usize], Vec::new(), Vec::new());
        for u in 0..k {
            let (cy, cx) = (2 + 2 * u, 2 + 3 * (u % 3));
            for dy in 0..3 {
                for dx in 0..3 {
                    indices.push((cy + dy).min(h - 1) * w + (cx + dx).min(w - 1));
                    data.push(0.2 + lcg(&mut seed));
                }
            }
            indptr.push(indices.len());
        }
        // duplicate pixels (clamped at the border) are harmless for this test
        let a = Csr {
            nrows: k,
            ncols: p,
            indptr,
            indices,
            data,
        };
        let ymean: Vec<f64> = (0..p).map(|_| 20.0 + 10.0 * lcg(&mut seed)).collect();
        let frames: Vec<Vec<f32>> = (0..t)
            .map(|_| {
                (0..p)
                    .map(|_| (255.0 * lcg(&mut seed)) as f32 * (lcg(&mut seed) as f32))
                    .collect()
            })
            .collect();

        let mut one = StreamPass::new(params.clone(), &a, &ymean);
        for y in &frames {
            one.push(y);
        }

        let model = TraceModel::build(&a, &ymean);
        let mut two = StreamPass::new(params.clone(), &a, &ymean);
        let mut sol = Vec::new();
        let mut solved = Vec::new();
        for y in &frames {
            model.solve(y, &mut sol);
            let cc: Vec<f64> = sol[..k].iter().map(|v| v.max(0.0)).collect();
            two.push_solved(y, &cc, sol[k] as f32);
            solved.push((cc, sol[k] as f32));
        }

        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        // the fold in lanes (rows, pixels and pairs shared out), put together: the same sums bit for bit
        for lanes in [1, 2, 3, 5, 9] {
            let mut three = StreamPass::new(params.clone(), &a, &ymean);
            let ranges = lane_ranges(k, p, &three.support, lanes);
            assert_eq!(ranges.len(), lanes);
            assert_eq!(ranges[0].pix[0], 0);
            assert_eq!(ranges[lanes - 1].pix[1], p);
            assert_eq!(ranges[lanes - 1].pairs[1], three.support.pix.len());
            let mut ls: Vec<FoldLane> = ranges
                .iter()
                .map(|&r| FoldLane::new(&params, &three.support, k, r))
                .collect();
            for (y, (cc, f)) in frames.iter().zip(&solved) {
                three.push_traces(cc, *f);
                for l in &mut ls {
                    l.fold(&y[l.range.pix[0]..l.range.pix[1]], cc);
                }
            }
            for l in &ls {
                three
                    .put_lane(l.range, &l.done.to_flat(), &l.seg.to_flat())
                    .unwrap();
            }
            assert_eq!(bits32(&one.c), bits32(&three.c));
            assert_eq!(bits32(&one.f), bits32(&three.f));
            assert_eq!(one.t, three.t);
            let (a, b) = (one.total(), three.total());
            assert_eq!(a.n, b.n);
            for (x, y) in [
                (&a.l, &b.l),
                (&a.m, &b.m),
                (&a.csum, &b.csum),
                (&a.ysum, &b.ysum),
                (&a.dsq, &b.dsq),
                (&a.mlp, &b.mlp),
                (&a.lsum, &b.lsum),
            ] {
                assert_eq!(bits(x), bits(y), "{lanes} lanes");
            }
        }

        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits32(&one.c), bits32(&two.c));
        assert_eq!(bits32(&one.f), bits32(&two.f));
        let (a, b) = (one.total(), two.total());
        assert_eq!(a.n, b.n);
        assert_eq!(a.n, t);
        for (x, y) in [
            (&a.l, &b.l),
            (&a.m, &b.m),
            (&a.csum, &b.csum),
            (&a.ysum, &b.ysum),
            (&a.dsq, &b.dsq),
            (&a.mlp, &b.mlp),
            (&a.lsum, &b.lsum),
        ] {
            assert_eq!(bits(x), bits(y));
        }
        assert!(a.mlp.iter().any(|&v| v != 0.0));
    }
}
