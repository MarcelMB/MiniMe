"""Score a pipeline's units (footprints + traces) against the synthetic ground truth.

Used for both the minime run and the exported Minian reference. Depends only on
numpy, scipy and pandas (plus the synthetic ground truth in ``minime.testing.synth``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from minime.testing import synth

BASELINE_F0 = 20.0  # unbleached baseline; score_run uses setup.F0 (per frame)
MAX_DIST_PX = 5.0
SELF_CHECK_PX = 1.5


# --- matching --------------------------------------------------------------------------


def match_units(
    unit_centroids: np.ndarray, true_centroids: np.ndarray, max_dist: float = MAX_DIST_PX
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching by increasing distance (pairs farther than max_dist dropped)."""
    u = np.asarray(unit_centroids, float).reshape(-1, 2)
    t = np.asarray(true_centroids, float).reshape(-1, 2)
    empty = np.zeros(0, np.intp)
    if len(u) == 0 or len(t) == 0:
        return empty, empty.copy()
    d = np.hypot(u[:, None, 0] - t[None, :, 0], u[:, None, 1] - t[None, :, 1])
    iu, it = np.nonzero(d <= max_dist)
    order = np.argsort(d[iu, it], kind="stable")
    used_u: set[int] = set()
    used_t: set[int] = set()
    ui: list[int] = []
    ti: list[int] = []
    for k in order:
        a, b = int(iu[k]), int(it[k])
        if a in used_u or b in used_t:
            continue
        used_u.add(a)
        used_t.add(b)
        ui.append(a)
        ti.append(b)
    return np.array(ui, np.intp), np.array(ti, np.intp)


def summarise(ui: np.ndarray, ti: np.ndarray, n_units: int, n_true: int) -> dict:
    n = len(ui)
    recall = n / n_true if n_true else 0.0
    precision = n / n_units if n_units else 0.0
    f1 = 2 * recall * precision / (recall + precision) if n else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "n_matched": n}


# --- footprints and motion -------------------------------------------------------------


def footprint_centroids(A: sparse.spmatrix, H: int, W: int) -> np.ndarray:
    """Intensity-weighted (y, x) centroid of each row of a (K, H*W) row-major footprint matrix."""
    A = sparse.csr_matrix(A)
    K = A.shape[0]
    out = np.full((K, 2), np.nan)
    w = np.abs(A.data).astype(np.float64)
    rows = np.repeat(np.arange(K), np.diff(A.indptr))
    tot = np.bincount(rows, weights=w, minlength=K)
    ys = np.bincount(rows, weights=w * (A.indices // W), minlength=K)
    xs = np.bincount(rows, weights=w * (A.indices % W), minlength=K)
    ok = tot > 0
    out[ok, 0] = ys[ok] / tot[ok]
    out[ok, 1] = xs[ok] / tot[ok]
    return out


def motion_fit(est: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Offset c with ``est = c + sign * true``, motion RMSE (px) and sign.

    Same as ``bench_opt.motion_fit`` of the minian_v2.0 prototype.
    """
    best = None
    for sign in (-1, 1):
        d = est - sign * true
        c = d.mean(axis=0)
        rmse = float(np.sqrt(((d - c) ** 2).sum(axis=1).mean()))
        if best is None or rmse < best[1]:
            best = (c, rmse, sign)
    return best  # type: ignore[return-value]


def _median_dist(unit_c: np.ndarray, true_c: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    ok = np.isfinite(unit_c).all(axis=1)
    idx = np.flatnonzero(ok)
    ui, ti = match_units(unit_c[ok], true_c)
    ui = idx[ui]
    if len(ui) == 0:
        return float("inf"), ui, ti
    d = np.hypot(*(unit_c[ui] - true_c[ti]).T)
    return float(np.median(d)), ui, ti


# --- scoring ---------------------------------------------------------------------------


def _truth_traces(setup: synth.SynthSetup) -> tuple[np.ndarray, np.ndarray]:
    """Truth traces above F0 in recorded frame order, and which recorded frames show a source
    frame (dropped frames are black and excluded from the correlation)."""
    above = (setup.C - setup.F0).astype(np.float32)
    fm = setup.frame_map
    ok = fm >= 0
    out = np.zeros_like(above)
    out[:, ok] = above[:, fm[ok]]
    return out, ok


def _trace_corr(traces: np.ndarray, truth: np.ndarray) -> np.ndarray:
    a = traces - traces.mean(axis=1, keepdims=True)
    b = truth - truth.mean(axis=1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / den


def score_run(
    A: sparse.spmatrix,
    traces: np.ndarray,
    H: int,
    W: int,
    meta_path: Path,
    shifts: np.ndarray,
) -> dict:
    """Score units against the truth of the session described by ``meta_path``.

    Units live in template coordinates = frame coordinates + a constant. With the fit
    ``shifts = c + sign * true_shifts`` and a pipeline that aligns frames by ``shifts``,
    that constant is ``sign * c``. The sign of the warp convention is verified with the
    units themselves: both signs are evaluated and the one with the smaller median matched
    distance kept. ``alignment_ok`` is False when even the better one is >= 1.5 px (or there
    are no matches), i.e. the offset could not be verified; a warning goes to stderr.
    ``offset_sign_used`` is the multiplier applied to the fitted ``c`` (so the offset actually
    added to the true centroids is
    ``offset_sign_used * c``, reported as ``align_offset_px``).
    """
    meta = json.loads(Path(meta_path).read_text())
    options = synth.SynthOptions.from_dict(meta.get("options"))
    setup = synth.synth_setup(meta["height"], meta["width"], meta["frames"], meta["seed"], options)
    true_c = synth.true_centroids(setup)
    traces = np.asarray(traces, np.float32)
    if traces.shape[1] != setup.C.shape[1]:
        raise ValueError(f"traces have {traces.shape[1]} frames, truth has {setup.C.shape[1]}")

    c, _, sign = motion_fit(np.asarray(shifts, float), setup.shifts.astype(float))
    unit_c = footprint_centroids(A, H, W)
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    for mult in (sign, -sign):  # always evaluate both; ties keep the derived sign
        med, ui, ti = _median_dist(unit_c, true_c + mult * c)
        if best is None or med < best[0]:
            best = (med, mult, ui, ti)
    assert best is not None
    med, mult, ui, ti = best

    n_units, n_true = A.shape[0], len(true_c)
    out = summarise(ui, ti, n_units, n_true)
    truth, ok = _truth_traces(setup)
    corr = _trace_corr(traces[ui][:, ok], truth[ti][:, ok]) if len(ui) else np.array([np.nan])
    out.update(
        n_units=int(n_units),
        n_true=int(n_true),
        trace_corr_median=float(np.nanmedian(corr)) if np.isfinite(corr).any() else None,
        trace_corr_q25=float(np.nanquantile(corr, 0.25)) if np.isfinite(corr).any() else None,
        align_offset_px=[float(v) for v in mult * c],
        offset_sign_used=int(mult),
        median_match_dist_px=None if not np.isfinite(med) else med,
        alignment_ok=bool(med < SELF_CHECK_PX),
    )
    if not out["alignment_ok"]:
        print(
            f"WARNING: alignment self-check failed (median matched distance {med:.2f} px, "
            f"{len(ui)} matches); scores may be unreliable",
            file=sys.stderr,
        )
    return out


# --- loading and CLI helpers -----------------------------------------------------------


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def score_pipeline_dir(run_dir: Path, raw_dir: Path | None = None) -> dict:
    """Score a minime run directory (``cnmf/``, ``motion/``, ``factors/``)."""
    run_dir = Path(run_dir)
    if raw_dir is None:
        raw_dir = Path(_read_json(run_dir / "params.json")["raw_dir"])
    meta = _read_json(run_dir / "factors" / "meta.json")
    A = sparse.load_npz(run_dir / "cnmf" / "A.npz")
    traces = np.load(run_dir / "cnmf" / "YrA.npy")
    shifts = np.load(run_dir / "motion" / "arrays.npz")["shifts"]
    return score_run(
        A, traces, int(meta["H"]), int(meta["W"]), Path(raw_dir).parent / "meta.json", shifts
    )


def score_minian_npz(npz_path: Path, raw_dir: Path) -> dict:
    """Score a ``minian_units.npz`` written by ``bench/minian_export.py``."""
    meta = _read_json(Path(raw_dir).parent / "meta.json")
    z = np.load(npz_path)
    A = sparse.csr_matrix(z["A"])
    return score_run(
        A,
        z["C"],
        int(meta["height"]),
        int(meta["width"]),
        Path(raw_dir).parent / "meta.json",
        z["shifts"],
    )
