"""Bounded-memory MiniMe prototype, tested on a synthetic session with ground truth.

Footprints come from short time windows; traces and the spatial update come from sums
streamed over the whole session, so the state kept between frames does not grow with ``T``
(the browser design in ``docs/browser-options.md``):

* ``Session``: the synthetic session (truth) and its batch MiniMe run (the baseline).
* ``time_windows``, ``window_units_subset``, ``window_units_fresh``: MiniMe's window 2 on one
  time window, either on a time subset of the baseline factors (fast screen, but the tile
  bases saw the whole session) or on a fresh MiniMe run of that window's frames only.
* ``stitch``: one unit list from every window's units, by footprint cosine similarity.
* ``FrameStream``: preprocessed, motion-corrected frames as pass 1 makes them, a batch at a
  time, with the baseline's shifts.
* ``TraceModel``, ``stream_pass``: per-frame least-squares traces with fixed footprints plus
  the running sums the spatial update needs, per segment.
* ``TileBasis``, ``merge_basis``: a per-tile spatial basis the frames are projected onto
  before the fit (the denoised movie the batch traces see), from pass-2 factors or folded
  window by window with a rank cap.
* ``spatial_from_stats``, ``merge_from_stats``, ``segment_stability``: ``cnmf.update_spatial``
  and the loop's unit merge on those sums, without frames.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.signal import butter, sosfilt

import minime
from minime import cnmf, corr, init, io, motion, pipeline, places, preprocess
from minime.factors import Factors
from minime.params import Params
from minime.report import StageTimer, peak_rss
from minime.score import row_corr, score_run
from minime.spatial_kernel import solve_nnqp

DEFAULT_SESSION = Path("/Volumes/Extreme SSD/synth_v4_grin30")

# experiment runs must not show up in `minime serve`'s list of recent sessions
places.remember = lambda *a, **k: None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """A synthetic session folder (``raw/``, ``meta.json``, ``truth.npz``) and a batch run."""

    root: Path = DEFAULT_SESSION
    baseline: str = "baseline"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def run_dir(self) -> Path:
        return self.root / "processed" / self.baseline

    def out(self, name: str) -> Path:
        """``processed/stream/<name>``, created (raw data stays read-only)."""
        d = self.root / "processed" / "stream" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shape(self) -> tuple[int, int, int]:
        """``(H, W, T)``."""
        m = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return int(m["height"]), int(m["width"]), int(m["frames"])

    def params(self) -> Params:
        return Params.from_json(self.run_dir / "params.json")

    def factors(self) -> Factors:
        return Factors.load(self.run_dir / "factors")

    def sidecar(self) -> motion.MotionSidecar:
        return motion.load_sidecar(self.run_dir / "motion")

    def baseline_score(self) -> dict:
        return json.loads((self.run_dir / "score.json").read_text(encoding="utf-8"))

    def baseline_units(self, stage: str) -> tuple[sps.csr_matrix, np.ndarray | None]:
        """Footprints (and final traces for ``stage="cnmf"``) of the baseline; ``stage`` is
        ``"init"`` (after initA and the first merge) or ``"cnmf"`` (final)."""
        A = sps.csr_matrix(sps.load_npz(self.run_dir / stage / "A.npz"))
        C = np.load(self.run_dir / "cnmf" / "YrA.npy") if stage == "cnmf" else None
        return A, C


def score(s: Session, A: sps.csr_matrix, traces: np.ndarray | None, shifts: np.ndarray) -> dict:
    """``score_run`` against the truth; ``traces=None`` scores the footprints only."""
    H, W, T = s.shape()
    if traces is None:
        traces = np.zeros((A.shape[0], T), dtype=np.float32)  # trace metrics come out None
    return score_run(A, traces, H, W, s.meta_path, shifts)


def brief(sc: dict) -> dict:
    """The headline numbers of a score."""
    keys = ("n_units", "recall", "precision", "f1", "recall_detectable", "trace_corr_median")
    out = {k: sc.get(k) for k in keys}
    out["recall_by_category"] = {k: v["recall"] for k, v in sc["recall_by_category"].items()}
    return out


# ---------------------------------------------------------------------------
# Footprints from time windows
# ---------------------------------------------------------------------------


@dataclass
class WindowUnits:
    t0: int
    t1: int
    A: sps.csr_matrix  # (K_w, P) final footprints of the window's own CNMF
    bright: np.ndarray  # (K_w,) peak of each unit's trace in the window
    seconds: float

    def save(self, path: Path) -> None:
        sps.save_npz(path.with_suffix(".A.npz"), self.A)
        np.savez(path, t=[self.t0, self.t1], bright=self.bright, seconds=self.seconds)

    @classmethod
    def load(cls, path: Path) -> WindowUnits:
        with np.load(path) as z:
            t0, t1 = (int(v) for v in z["t"])
            bright, seconds = z["bright"], float(z["seconds"])
        A = sps.csr_matrix(sps.load_npz(path.with_suffix(".A.npz")))
        return cls(t0, t1, A, bright, seconds)


def time_windows(T: int, length: int, edges: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Consecutive ``[t0, t1)`` windows of about ``length`` frames covering ``T``.

    A last piece shorter than half a window joins the previous window. With ``edges`` every
    bound snaps to the nearest edge (block edges for ``Factors.subset``, segment starts for
    ``window_units_fresh``).
    """
    bounds = [*range(0, T, length), T]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < length // 2:
        del bounds[-2]
    if edges is not None:
        e = np.asarray(edges)
        bounds = sorted({int(e[np.argmin(np.abs(e - b))]) for b in bounds})
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _empty_units(t0: int, t1: int, P: int, seconds: float, why: Exception) -> WindowUnits:
    warnings.warn(f"window {t0}-{t1}: no units ({why})", stacklevel=2)
    return WindowUnits(t0, t1, sps.csr_matrix((0, P), dtype=np.float32), np.zeros(0), seconds)


def window_units_subset(fac: Factors, params: Params, t0: int, t1: int) -> WindowUnits:
    """Window 2 on frames ``[t0, t1)`` of the baseline factors (``Factors.subset``)."""
    t = time.perf_counter()
    try:
        res = pipeline.run_window2(fac.subset(t0, t1), params, StageTimer()).cnmf
    except ValueError as e:  # no seeds / all units dropped in a quiet window
        return _empty_units(t0, t1, fac.P, time.perf_counter() - t, e)
    bright = res.YrA.max(axis=1) if res.YrA.shape[0] else np.zeros(0)
    return WindowUnits(t0, t1, res.A, bright, time.perf_counter() - t)


def segment_starts(info: io.RawInfo) -> np.ndarray:
    """First frame of every raw segment, plus ``T``."""
    return np.cumsum([0, *info.segment_frames])


def window_raw_dir(info: io.RawInfo, t0: int, t1: int, work: Path) -> Path:
    """A folder of symlinks to the raw segments of frames ``[t0, t1)`` (segment bounds)."""
    starts = segment_starts(info)
    i0, i1 = int(np.searchsorted(starts, t0)), int(np.searchsorted(starts, t1))
    if starts[i0] != t0 or starts[i1] != t1:
        raise ValueError(f"window {t0}-{t1} does not start and end on raw segment bounds")
    d = work / f"raw_{t0}_{t1}"
    d.mkdir(parents=True, exist_ok=True)
    for j, seg in enumerate(info.segments[i0:i1]):
        link = d / f"{j}{seg.suffix}"
        if not link.exists():
            link.symlink_to(seg.resolve())
    return d


def _seed_sidecar(
    out: Path, raw: Path, sc: motion.MotionSidecar, t0: int, t1: int, params: Params
) -> None:
    """Give the window's run the baseline shifts of its frames, so every window's units sit
    in the same template coordinates (``pipeline.run`` reuses a valid ``motion/`` sidecar)."""
    motion.save_sidecar(
        out / "motion",
        motion.MotionSidecar(
            shifts=sc.shifts[t0:t1],
            template=sc.template,
            quality=sc.quality[t0:t1],
            params=params.motion_params(),
            raw_hash=io.hash_files(io.list_segments(raw)),
            meta={"source": f"baseline shifts {t0}-{t1}"},
            fill=sc.fill,  # the session's fill image, not one from the window's frames
        ),
    )
    # marks the folder as a minime run, so pipeline.run may wipe it after reading the sidecar
    (out / "params.json").write_text(json.dumps({"minime_version": minime.__version__}))


def window_units_fresh(
    info: io.RawInfo,
    sc: motion.MotionSidecar,
    params: Params,
    t0: int,
    t1: int,
    work: Path,
) -> WindowUnits:
    """A whole MiniMe run (pass 1, pass 2, window 2) on frames ``[t0, t1)`` only."""
    t = time.perf_counter()
    raw = window_raw_dir(info, t0, t1, work)
    out = work / f"run_{t0}_{t1}"
    _seed_sidecar(out, raw, sc, t0, t1, params)
    try:
        pipeline.run(raw, params, run_name=out.name, out_root=out.parent)
    except ValueError as e:
        return _empty_units(t0, t1, info.height * info.width, time.perf_counter() - t, e)
    A = sps.csr_matrix(sps.load_npz(out / "cnmf" / "A.npz"))
    YrA = np.load(out / "cnmf" / "YrA.npy")
    bright = YrA.max(axis=1) if YrA.shape[0] else np.zeros(0)
    return WindowUnits(t0, t1, A, bright, time.perf_counter() - t)


def window_mean_noise(runs: Path, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Session mean image and noise from the ``fresh`` windows' pass-2 factors (``runs/``;
    the first ``n`` windows only with ``n``).

    The mean is the frame-weighted mean of the windows' means (the same frames, so equal to
    the session's up to rounding). The noise is the quadratic mean of the windows' noise,
    ``sqrt(mean_w sigma_w^2)``: equal to the session's on stationary noise, and like the
    session's whole-trace FFT it scales with the share of frames a pixel has signal in (motion
    borders), where a geometric mean would follow the windows with none to zero.
    """
    dirs = sorted(runs.glob("run_*_*"), key=lambda d: int(d.name.split("_")[1]))[:n]
    if not dirs:
        raise FileNotFoundError(f"no run_<t0>_<t1> folders in {runs}")
    mean = sq = None
    T = 0
    for d in dirs:
        f = Factors.load(d / "factors")
        t = int(d.name.split("_")[2]) - int(d.name.split("_")[1])
        m, sg = f.mean_flat().astype(np.float64), f.sigma_flat().astype(np.float64)
        mean = t * m if mean is None else mean + t * m
        sq = sg**2 if sq is None else sq + sg**2
        T += t
    return mean / T, np.sqrt(sq / len(dirs))


def _roots(parent: np.ndarray) -> np.ndarray:
    """Root of every node of a ``brightest_parent`` forest."""
    root = parent.copy()
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            return root
        root = nxt


def _row_max_one(A: sps.csr_matrix) -> sps.csr_matrix:
    m = A.max(axis=1).toarray().ravel()
    m[m == 0] = 1.0
    return sps.csr_matrix(sps.diags(1.0 / m) @ A)


def stitch(
    units: list[WindowUnits],
    cos_thres: float,
    min_dist: float = 0.0,
    shape: tuple[int, int] | None = None,
) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """One footprint per cell from the units of every window.

    Units of different windows whose footprints (rows scaled to maximum 1) have cosine
    similarity above ``cos_thres`` are linked. Each unit joins its brightest linked neighbour
    that is brighter than itself (``corr.brightest_parent``, the seed merge's rule, so a
    chain cannot climb from one cell to a neighbouring cell through dim units), and every
    tree is averaged into one footprint. Units of the same window are never linked: window 2
    already decided to keep them apart.

    With ``min_dist > 0`` (``shape`` = (H, W)) groups whose mean footprints have centroids
    closer than ``min_dist`` px are joined (each to its brightest such group brighter than
    itself), as the batch CNMF loop's unit merge does with ``merge_min_dist``
    (``_close_groups``): two cells cannot sit that close. Each
    window's CNMF applies that rule inside the window; without it across windows, a cell
    whose footprint one window estimated worse (cosine <= ``cos_thres``) stays a second unit,
    and such units pile up with the number of windows.

    Returns ``(A (K, P) float32 rows with maximum 1, bright (K,), n_windows (K,))``.
    """
    A = _row_max_one(sps.vstack([u.A for u in units]).tocsr().astype(np.float64))
    win = np.concatenate([np.full(u.A.shape[0], i) for i, u in enumerate(units)])
    bright = np.concatenate([u.bright for u in units]).astype(np.float64)
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    An = sps.csr_matrix(sps.diags(1.0 / norms) @ A)
    S = sps.triu(An @ An.T, k=1).tocoo()
    keep = (S.data > cos_thres) & (win[S.row] != win[S.col])
    edges = np.stack([S.row[keep], S.col[keep]], axis=1).astype(np.int64)
    _, g = np.unique(_roots(corr.brightest_parent(edges, bright)), return_inverse=True)
    if min_dist > 0:
        if shape is None:
            raise ValueError("min_dist needs shape=(H, W)")
        g = _close_groups(A, g, bright, min_dist, shape)[g]
    n_g = int(g.max()) + 1
    A_st = _group_means(A, g).astype(np.float32)
    bright_g = np.zeros(n_g)
    np.maximum.at(bright_g, g, bright)
    n_windows = np.bincount(np.unique(np.stack([g, win]), axis=1)[0], minlength=n_g)
    return A_st, bright_g, n_windows


def _group_means(A: sps.csr_matrix, g: np.ndarray) -> sps.csr_matrix:
    """Mean footprint of every group ``g`` of the rows of ``A``, rows scaled to maximum 1."""
    n_g, N = int(g.max()) + 1, g.size
    M = sps.csr_matrix((1.0 / np.bincount(g)[g], (g, np.arange(N))), shape=(n_g, N))
    return _row_max_one(sps.csr_matrix(M @ A))


def _close_groups(
    A: sps.csr_matrix, g: np.ndarray, bright: np.ndarray, min_dist: float, shape: tuple[int, int]
) -> np.ndarray:
    """New label of every group: groups whose mean footprints' centroids are closer than
    ``min_dist`` px are adjacent (``init.unit_merge``'s ``min_dist`` rule), and each group joins
    its brightest adjacent group brighter than itself (``corr.brightest_parent`` on the groups'
    brightest units), as the links do: in a dense field connected components percolate
    through chains of close groups into neighbouring cells."""
    n_g = int(g.max()) + 1
    bg = np.zeros(n_g)
    np.maximum.at(bg, g, bright)
    cent = init.centroids(_group_means(A, g), *shape)
    d = np.hypot(cent[:, None, 0] - cent[None, :, 0], cent[:, None, 1] - cent[None, :, 1])
    with np.errstate(invalid="ignore"):
        close = np.triu(d < min_dist, 1)  # NaN centroids never compare close
    edges = np.stack(np.nonzero(close), axis=1).astype(np.int64)
    return np.unique(_roots(corr.brightest_parent(edges, bg)), return_inverse=True)[1]


# ---------------------------------------------------------------------------
# Streaming frames and traces
# ---------------------------------------------------------------------------


class FrameStream:
    """Preprocessed, motion-corrected frames ``(n, P) float32`` as pass 1 makes them, one
    ``batch`` at a time; iterating yields ``(t0, frames)``. Each iteration rereads the raw
    files, as a browser would reread the file with ``File.slice``. With ``motion_fill =
    "template"`` the session's ``fill`` image (the sidecar's) goes where a frame has no data."""

    def __init__(
        self,
        info: io.RawInfo,
        shifts: np.ndarray,
        params: Params,
        batch: int = 100,
        fill: np.ndarray | None = None,
    ) -> None:
        if shifts.shape[0] != info.frames:
            raise ValueError(f"{shifts.shape[0]} shifts for {info.frames} frames")
        if params.motion_fill == "template" and fill is None:
            raise ValueError("motion_fill is 'template' but no fill image was given")
        self.info, self.shifts, self.params, self.batch = info, shifts, params, batch
        self.fill = fill if params.motion_fill == "template" else None

    def _prep(self, raw: np.ndarray, sh: np.ndarray) -> np.ndarray:
        p = self.params
        rep = self.fill is not None
        blk = preprocess.denoise(raw, p.median_ksize)
        if p.motion_before_tophat:
            blk = preprocess.tophat(motion.apply_shifts(blk, sh, replicate=rep), p.tophat_wnd)
        else:
            blk = motion.apply_shifts(preprocess.tophat(blk, p.tophat_wnd), sh, replicate=rep)
        if rep:
            blk = motion.fill_outside(blk, sh, self.fill)
        return blk.reshape(blk.shape[0], -1).astype(np.float32)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        t = 0
        for seg in self.info.segments:
            for raw in io.iter_avi(seg, self.batch, self.info.bgr):
                yield t, self._prep(raw, self.shifts[t : t + raw.shape[0]])
                t += raw.shape[0]


def _normal_solver(G: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """``B -> G^-1 B`` from one factorisation (``init._solve_normal``'s rule)."""
    if np.linalg.cond(G) <= init.COND_MAX:
        try:
            cf = cho_factor(G, lower=True, check_finite=False)
        except LinAlgError:
            pass
        else:
            return lambda B: cho_solve(cf, B, check_finite=False)
    Gi = np.linalg.pinv(G, rcond=init.PINV_RCOND, hermitian=True)
    return lambda B: Gi @ B


@dataclass
class TraceModel:
    """Fixed footprints ``A`` and a rank-1 background ``b``: every frame's least-squares
    ``(c_t, f_t)`` on ``X = [A; b]`` comes from one factorisation of ``X X^T``."""

    A: sps.csr_matrix  # (K, P) float64
    b: np.ndarray  # (P,) float64
    X: sps.csr_matrix  # (K + 1, P)
    solve_normal: Callable[[np.ndarray], np.ndarray]

    @classmethod
    def build(cls, A: sps.csr_matrix, ymean: np.ndarray, background: str = "fit") -> TraceModel:
        """``background="fit"``: ``b = clip(ymean - A^T cbar, 0)``, ``cbar`` the least-squares
        fit of the mean image (``cnmf``'s default background from the mean image alone). The
        fit gives every unit the whole local mean, so ``b`` has a hole at each cell and the
        background's fluctuations under a cell go into its trace. ``background="mean"``:
        ``b = ymean``, the cells' resting levels stay in ``b`` and follow ``f``."""
        A64 = sps.csr_matrix(A, dtype=np.float64)
        ymean = np.asarray(ymean, dtype=np.float64)
        if background == "mean":
            b = np.clip(ymean, 0, None)
        elif background == "fit":
            G = (A64 @ A64.T).toarray()
            cbar = _normal_solver(0.5 * (G + G.T))(A64 @ ymean)
            b = np.clip(ymean - A64.T @ cbar, 0, None)
        else:
            raise ValueError(f"background {background!r}: 'fit' or 'mean'")
        X = sps.csr_matrix(sps.vstack([A64, sps.csr_matrix(b[None, :])]))
        GX = (X @ X.T).toarray()
        return cls(A64, b, X, _normal_solver(0.5 * (GX + GX.T)))

    def solve(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(C (n, K) float64 unclipped, f (n,))`` for frames ``Y (n, P)``."""
        sol = self.solve_normal(np.asarray(self.X @ Y.T))  # (K + 1, n)
        return sol[:-1].T, sol[-1]


@dataclass
class Support:
    """The (pixel, unit) pairs ``cnmf.update_spatial`` solves for, in its order."""

    active: np.ndarray  # (n_active,) pixels with at least one allowed unit
    indptr: np.ndarray  # (n_active + 1,) int64
    indices: np.ndarray  # (nnz,) int64 unit of each pair
    pix: np.ndarray  # (nnz,) int64 pixel of each pair

    @classmethod
    def build(cls, A: sps.csr_matrix, H: int, W: int, dl_wnd: int) -> Support:
        sub = cnmf.dilate_footprints(A, H, W, dl_wnd)
        subT = sps.csr_matrix(sub.T)
        counts = np.diff(subT.indptr)
        active = np.flatnonzero(counts)
        indptr = np.zeros(active.size + 1, dtype=np.int64)
        np.cumsum(counts[active], out=indptr[1:])
        indices = subT.indices.astype(np.int64)
        return cls(active, indptr, indices, np.repeat(active, counts[active]).astype(np.int64))


@dataclass
class SegmentStats:
    """Running sums over the frames of one segment: everything the spatial update, the
    unit merge and the noise estimate need. Size O(nnz + K^2 + P), whatever its length."""

    n: int
    L: np.ndarray  # (nnz,) sum_t y_t[pix] c_t[unit] on the support pairs
    M: np.ndarray  # (K, K) sum_t c_t c_t^T
    csum: np.ndarray  # (K,) sum_t c_t
    ysum: np.ndarray  # (P,) sum_t y_t
    dsq: np.ndarray  # (P,) sum_t (y_t - y_{t-1})^2
    Mlp: np.ndarray  # (K, K) sum_t l_t l_t^T of the low-passed traces l_t (``LowPass``)
    lsum: np.ndarray  # (K,) sum_t l_t

    SUMS = ("L", "M", "csum", "ysum", "dsq", "Mlp", "lsum")

    @classmethod
    def zeros(cls, nnz: int, K: int, P: int) -> SegmentStats:
        z = np.zeros
        return cls(0, z(nnz), z((K, K)), z(K), z(P), z(P), z((K, K)), z(K))

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, k).nbytes for k in self.SUMS)


def total(stats: list[SegmentStats]) -> SegmentStats:
    out = SegmentStats.zeros(stats[0].L.size, stats[0].csum.size, stats[0].ysum.size)
    for s in stats:
        out.n += s.n
        for k in SegmentStats.SUMS:
            getattr(out, k).__iadd__(getattr(s, k))
    return out


@dataclass
class PassResult:
    C: np.ndarray  # (K, T) float32 traces, clipped at 0 (the YrA analogue)
    f: np.ndarray  # (T,) float32 background trace
    stats: list[SegmentStats]
    seconds: float
    state_bytes: int  # stats + support + model: what a browser keeps between frames
    timing: dict[str, float] = field(default_factory=dict)


class LowPass:
    """Causal 4th-order Butterworth low-pass of the traces, run batch by batch with its
    state carried over: the streaming stand-in for ``corr.lowpass`` (a brick-wall FFT
    filter over the whole session) that the batch unit merge correlates. ``noise_freq`` is
    the cutoff as a fraction of the frame rate, as in ``Params.noise_freq``."""

    def __init__(self, noise_freq: float, K: int) -> None:
        self.sos = butter(4, 2 * noise_freq, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, K))

    def __call__(self, C: np.ndarray) -> np.ndarray:
        out, self.zi = sosfilt(self.sos, C, axis=0, zi=self.zi)
        return out


def stream_pass(
    frames: FrameStream,
    model: TraceModel,
    sup: Support,
    T: int,
    seg_frames: int,
    noise_freq: float,
    chunk: int = 50,
    project: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> PassResult:
    """One read of the movie: traces with fixed footprints and per-segment sums.

    ``project(t0, Y)``, if given, replaces each batch of frames before the fit and the sums
    (``WindowProjector``); the noise sum ``dsq`` stays on the raw frames.
    """
    if seg_frames % frames.batch:
        raise ValueError(f"segment length {seg_frames} is not a multiple of {frames.batch}")
    K, P = model.A.shape
    n_seg = -(-T // seg_frames)
    stats = [SegmentStats.zeros(sup.pix.size, K, P) for _ in range(n_seg)]
    C_out = np.zeros((K, T), dtype=np.float32)
    f_out = np.zeros(T, dtype=np.float32)
    lowpass = LowPass(noise_freq, K)
    prev: np.ndarray | None = None
    timing = {"read_prep": 0.0, "project": 0.0, "solve": 0.0, "sums": 0.0}
    t_start = t = time.perf_counter()
    for t0, Y_raw in frames:
        n = Y_raw.shape[0]
        now = time.perf_counter()
        timing["read_prep"] += now - t
        Y = Y_raw if project is None else project(t0, Y_raw)
        t = time.perf_counter()
        timing["project"] += t - now
        now = t
        C, fb = model.solve(Y)
        Cc = np.clip(C, 0, None)
        C_out[:, t0 : t0 + n] = Cc.T
        f_out[t0 : t0 + n] = fb
        t = time.perf_counter()
        timing["solve"] += t - now
        if t0 % seg_frames + n > seg_frames:
            raise ValueError(f"frames {t0}-{t0 + n} straddle a {seg_frames}-frame segment")
        st = stats[t0 // seg_frames]
        st.n += n
        st.M += Cc.T @ Cc
        st.csum += Cc.sum(axis=0)
        lp = lowpass(Cc)
        st.Mlp += lp.T @ lp
        st.lsum += lp.sum(axis=0)
        st.ysum += Y.sum(axis=0, dtype=np.float64)
        d = np.diff(Y_raw, axis=0)
        st.dsq += np.einsum("tp,tp->p", d, d, dtype=np.float64)
        if prev is not None:
            st.dsq += (Y_raw[0] - prev).astype(np.float64) ** 2
        prev = Y_raw[-1].copy()
        for c0 in range(0, n, chunk):
            st.L += np.einsum(
                "tp,tp->p", Y[c0 : c0 + chunk][:, sup.pix], Cc[c0 : c0 + chunk][:, sup.indices]
            )
        now = time.perf_counter()
        timing["sums"] += now - t
        t = now
    state = sum(s.nbytes for s in stats) + sum(
        a.nbytes for a in (sup.active, sup.indptr, sup.indices, sup.pix, model.b)
    )
    state += model.X.data.nbytes + model.X.indices.nbytes + (K + 1) ** 2 * 8 + lowpass.zi.nbytes
    return PassResult(C_out, f_out, stats, time.perf_counter() - t_start, state, timing)


# ---------------------------------------------------------------------------
# Per-tile spatial basis (the denoised movie)
# ---------------------------------------------------------------------------


def orthonormal(W: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column span of ``W`` (QR in float64), float32."""
    if W.shape[1] == 0:
        return np.zeros(W.shape, dtype=np.float32)
    q = np.linalg.qr(np.asarray(W, dtype=np.float64))[0]
    return np.ascontiguousarray(q, dtype=np.float32)


@dataclass
class TileBasis:
    """Per-tile orthonormal spatial bases: a frame's tile ``y`` is replaced by its
    projection ``mean + U U^T (y - mean)``, the low-rank movie pass 2 fits, one frame at a
    time. State ``sum_b P_b (r_b + 1)`` float32, independent of ``T``."""

    rows: list[np.ndarray]  # flat pixel indices of each tile
    mean: list[np.ndarray]  # (P_b,) float32
    U: list[np.ndarray]  # (P_b, r_b) float32, orthonormal columns

    @classmethod
    def from_factors(cls, fac: Factors) -> TileBasis:
        """The pass-2 basis of ``fac``: ``U`` spans ``W`` (``W = U s`` of the rSVD), so
        ``U U^T (y_t - mean)`` is the factors' own reconstruction ``W V_t^T`` of frame ``t``."""
        rows = [fac.tile_rows(b) for b in range(len(fac.tiles))]
        return cls(rows, [tf.mean for tf in fac.tiles], [orthonormal(tf.W) for tf in fac.tiles])

    @property
    def ranks(self) -> np.ndarray:
        return np.array([u.shape[1] for u in self.U])

    @property
    def nbytes(self) -> int:
        return sum(u.nbytes + m.nbytes for u, m in zip(self.U, self.mean, strict=True))

    def __call__(self, Y: np.ndarray) -> np.ndarray:
        """Projected frames ``(n, P) float32`` of ``Y (n, P)``."""
        out = np.empty_like(Y)
        for rows, m, U in zip(self.rows, self.mean, self.U, strict=True):
            Yb = Y[:, rows] - m
            out[:, rows] = (Yb @ U) @ U.T + m
        return out


def merge_basis(
    acc: tuple[np.ndarray, np.ndarray, int] | None,
    W: np.ndarray,
    mean: np.ndarray,
    n: int,
    cap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fold one window's scaled basis ``W`` (columns ``U s``, mean removed) into the running
    ``(W_acc, mean_acc, n_acc)`` of one tile (incremental PCA with mean update, Ross et al.
    2008): the SVD of ``[W_acc, W, sqrt(n_acc n / (n_acc + n)) (mean - mean_acc)]`` keeps
    its ``cap`` leading columns, so the state is ``(P_b, cap)`` whatever the window count."""
    W = np.asarray(W, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if acc is None:
        M, mean_new, n_new = W, mean, n
    else:
        W_acc, mean_acc, n_acc = acc
        n_new = n_acc + n
        mean_new = (n_acc * mean_acc + n * mean) / n_new
        extra = np.sqrt(n_acc * n / n_new) * (mean - mean_acc)
        M = np.concatenate([W_acc, W, extra[:, None]], axis=1)
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    k = min(cap, int((s > s[0] * 1e-7).sum())) if s.size else 0
    return u[:, :k] * s[:k], mean_new, n_new


def basis_from_windows(
    runs: Path, mode: str, cap: int
) -> tuple[list[tuple[int, int]], list[TileBasis]]:
    """Bases from the ``fresh`` windows' pass-2 factors (``runs/run_<t0>_<t1>/factors``).

    ``mode="windows"``: one ``TileBasis`` per window (its own factors). ``mode="merged"``:
    every window folded into one basis with ``merge_basis`` at ``cap`` columns per tile,
    returned once per window for the same interface.
    """
    dirs = sorted(runs.glob("run_*_*"), key=lambda d: int(d.name.split("_")[1]))
    spans = [(int(d.name.split("_")[1]), int(d.name.split("_")[2])) for d in dirs]
    if mode == "windows":
        return spans, [TileBasis.from_factors(Factors.load(d / "factors")) for d in dirs]
    acc: list[tuple[np.ndarray, np.ndarray, int] | None] = []
    rows: list[np.ndarray] = []
    for d, (t0, t1) in zip(dirs, spans, strict=True):
        fac = Factors.load(d / "factors")
        if not acc:
            acc = [None] * len(fac.tiles)
            rows = [fac.tile_rows(b) for b in range(len(fac.tiles))]
        for b, tf in enumerate(fac.tiles):
            acc[b] = merge_basis(acc[b], tf.W, tf.mean, t1 - t0, cap)
    basis = TileBasis(
        rows,
        [a[1].astype(np.float32) for a in acc],  # type: ignore[index]
        [orthonormal(a[0]) for a in acc],  # type: ignore[index]
    )
    return spans, [basis] * len(spans)


class WindowProjector:
    """``(t0, Y) -> projected Y`` with the basis of the window frame ``t0`` falls in."""

    def __init__(self, spans: list[tuple[int, int]], bases: list[TileBasis]) -> None:
        self.starts = np.array([t0 for t0, _ in spans])
        self.bases = bases

    def __call__(self, t0: int, Y: np.ndarray) -> np.ndarray:
        return self.bases[int(np.searchsorted(self.starts, t0, side="right")) - 1](Y)


# ---------------------------------------------------------------------------
# Spatial update and merge from the sums
# ---------------------------------------------------------------------------


def diff_noise(st: SegmentStats) -> np.ndarray:
    """Per-pixel noise std from successive frame differences, ``sqrt(mean(dy^2) / 2)``: a
    streaming stand-in for the factors' FFT noise (overestimates where cells fire often)."""
    return np.sqrt(st.dsq / max(2 * (st.n - 1), 1))


def _centred(st: SegmentStats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cov (K, K), cbar (K,), ybar (P,))`` of the traces and frames of ``st``."""
    cbar, ybar = st.csum / st.n, st.ysum / st.n
    return st.M - st.n * np.outer(cbar, cbar), cbar, ybar


def spatial_from_stats(
    A: sps.csr_matrix,
    sup: Support,
    st: SegmentStats,
    sn: np.ndarray,
    sparse_snr: float,
    size_thres: tuple[float | None, float | None],
) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """``cnmf.update_spatial`` from the sums of ``st``: same centring, unit-norm scaling,
    penalty ``sqrt(n) * sparse_snr * sn`` and post-processing, no frame read.

    ``Y Cc^T = (L - n ybar cbar^T) / scale`` on the support pairs and ``Cc Cc^T = cov /
    (scale scale^T)``, with ``scale`` the norms of the centred traces (``column_norms``).
    """
    K, P = A.shape
    cov, cbar, ybar = _centred(st)
    scale = np.sqrt(np.clip(np.diag(cov), 0, None))
    scale[scale == 0] = 1.0
    G = cov / np.outer(scale, scale)
    rows = np.repeat(np.arange(sup.active.size), np.diff(sup.indptr))
    B = np.zeros((sup.active.size, K))
    B[rows, sup.indices] = (st.L - st.n * ybar[sup.pix] * cbar[sup.indices]) / scale[sup.indices]
    lam = np.sqrt(st.n) * sparse_snr * np.asarray(sn, dtype=np.float64)[sup.active]
    values = solve_nnqp(G, B, lam, sup.indptr, sup.indices) / scale[sup.indices]
    A_new = sps.csr_matrix((values, (sup.indices, sup.pix)), shape=(K, P), dtype=np.float64)
    A_new.eliminate_zeros()
    mask = cnmf._size_mask(np.diff(A_new.indptr), size_thres)
    if not mask.any():
        raise ValueError("all units dropped by size_thres")
    A_new = A_new[np.flatnonzero(mask)]
    norm_fac = A_new.max(axis=1).toarray().ravel()
    A_new = sps.csr_matrix(sps.diags(1.0 / norm_fac) @ A_new, dtype=np.float32)
    return A_new, mask, norm_fac.astype(np.float32)


def merge_from_stats(
    A: sps.csr_matrix,
    st: SegmentStats,
    mask: np.ndarray,
    thres_corr: float,
    lowpassed: bool = True,
) -> sps.csr_matrix:
    """The loop's unit merge with trace correlations from the sums: overlapping units of
    ``A`` (the units ``mask`` kept) whose traces correlate above ``thres_corr`` are averaged.
    ``lowpassed`` correlates the low-passed traces, as the batch merge does; otherwise the
    raw per-frame traces."""
    if lowpassed:
        lbar = st.lsum / st.n
        cov = (st.Mlp - st.n * np.outer(lbar, lbar))[np.ix_(mask, mask)]
    else:
        cov = _centred(st)[0][np.ix_(mask, mask)]
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    rho = cov / np.outer(sd, sd)
    overlap = (A @ A.T).toarray() > 0
    np.fill_diagonal(overlap, False)
    labels = corr.merge_labels(sps.csr_matrix(overlap & (rho > thres_corr)))
    return sps.csr_matrix(init._group_mean_matrix(labels) @ A, dtype=np.float32)


def segment_stability(
    A_global: sps.csr_matrix,
    mask: np.ndarray,
    A: sps.csr_matrix,
    sup: Support,
    stats: list[SegmentStats],
    sn: np.ndarray,
    sparse_snr: float,
    H: int,
    W: int,
) -> dict[str, np.ndarray]:
    """Footprints from each segment's sums alone vs the global update ``A_global``.

    Returns ``cos (n_seg, K')`` cosine similarity and ``shift (n_seg, K')`` centroid
    distance in px per unit of ``A_global`` (NaN where a segment dropped the unit): how much
    footprints change over the session, i.e. whether fixed footprints are good enough.
    """
    kept = np.flatnonzero(mask)
    Kg = A_global.shape[0]
    cos = np.full((len(stats), Kg), np.nan)
    shift = np.full((len(stats), Kg), np.nan)
    g = A_global.astype(np.float64)
    gn = np.sqrt(np.asarray(g.multiply(g).sum(axis=1)).ravel())
    gc = init.centroids(A_global, H, W)
    for i, st in enumerate(stats):
        try:
            A_s, m_s, _ = spatial_from_stats(A, sup, st, sn, sparse_snr, (None, None))
        except ValueError:
            continue
        both = np.flatnonzero(m_s[kept])  # global units this segment kept too
        rows = np.searchsorted(np.flatnonzero(m_s), kept[both])
        s = A_s[rows].astype(np.float64)
        sn_ = np.sqrt(np.asarray(s.multiply(s).sum(axis=1)).ravel())
        dot = np.asarray(s.multiply(g[both]).sum(axis=1)).ravel()
        cos[i, both] = dot / np.maximum(sn_ * gn[both], 1e-12)
        shift[i, both] = np.hypot(*(init.centroids(A_s[rows], H, W) - gc[both]).T)
    return {"cos": cos, "shift": shift}


def trace_agreement(C: np.ndarray, C_ref: np.ndarray) -> float:
    """Median row correlation of two (K, T) trace sets of the same units."""
    return float(np.nanmedian(row_corr(np.asarray(C, np.float64), np.asarray(C_ref, np.float64))))


def rss_mb() -> float | None:
    r = peak_rss()
    return None if r is None else r / 2**20
