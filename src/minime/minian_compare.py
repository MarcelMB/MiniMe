"""A Minian result loaded, aligned and matched to a minime run (no ground truth).

Minian's coordinates are mapped onto minime's by phase-correlating the two max projections;
units are matched one-to-one by footprint centroid (``score.match_units``) and matched traces
are correlated.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse

from minime import preprocess
from minime.score import MAX_DIST_PX, footprint_centroids, match_units

# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "factors" / "meta.json").read_text())
    H, W = int(meta["H"]), int(meta["W"])
    return {
        "A": sparse.load_npz(run_dir / "cnmf" / "A.npz").tocsr(),
        "C": np.load(run_dir / "cnmf" / "YrA.npy"),
        "max_proj": np.load(run_dir / "factors" / "max_proj.npy").astype(np.float32),
        "H": H,
        "W": W,
        "report": json.loads((run_dir / "report.json").read_text()),
    }


def load_minian(npz: Path) -> dict:
    z = np.load(npz)
    return {
        "A": sparse.csr_matrix(z["A"]),
        "C": z["C"],
        "max_proj": z["max_proj"].astype(np.float32),
    }


def projection_offset(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, float]:
    """(dy, dx) to add to ``mov`` coordinates to land on ``ref``, and the peak response."""
    a, b = (preprocess.bandpass(x[None], 1.0, 6.0)[0] for x in (ref, mov))
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
    return np.array([dy, dx]), float(resp)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def footprint_sizes(A: sparse.csr_matrix, frac: float) -> np.ndarray:
    """Pixels per footprint above ``frac`` of its own maximum."""
    d = A.toarray()
    return (d > frac * d.max(axis=1, keepdims=True)).sum(axis=1)


def row_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (x * y).sum(axis=1) / den, np.nan)


def compare(run: dict, mn: dict) -> dict:
    H, W = run["H"], run["W"]
    off, resp = projection_offset(run["max_proj"], mn["max_proj"])
    cc = footprint_centroids(run["A"], H, W)
    mc = footprint_centroids(mn["A"], H, W) + off
    ui, mi = match_units(cc, mc, MAX_DIST_PX)
    T = min(run["C"].shape[1], mn["C"].shape[1])
    corr = row_corr(run["C"][ui, :T], mn["C"][mi, :T]) if len(ui) else np.zeros(0)
    dist = np.hypot(*(cc[ui] - mc[mi]).T) if len(ui) else np.zeros(0)
    return {
        "offset_px": off,
        "offset_response": resp,
        "minime_centroids": cc,
        "minian_centroids": mc,
        "ui": ui,
        "mi": mi,
        "corr": corr,
        "dist": dist,
        "T": T,
        "size10": (footprint_sizes(run["A"], 0.1), footprint_sizes(mn["A"], 0.1)),
        "size01": (footprint_sizes(run["A"], 0.01), footprint_sizes(mn["A"], 0.01)),
    }


def summary(run: dict, mn: dict, res: dict) -> dict:
    n_c, n_m, n = run["A"].shape[0], mn["A"].shape[0], len(res["ui"])
    s10, s01 = res["size10"], res["size01"]
    rep = run["report"]
    return {
        "minime_units": n_c,
        "minian_units": n_m,
        "matched": n,
        "frac_minian_found": n / n_m if n_m else float("nan"),
        "frac_minime_matched": n / n_c if n_c else float("nan"),
        "trace_corr_median": float(np.nanmedian(res["corr"])) if n else float("nan"),
        "trace_corr_q25": float(np.nanpercentile(res["corr"], 25)) if n else float("nan"),
        "match_dist_median_px": float(np.median(res["dist"])) if n else float("nan"),
        "offset_px": [float(v) for v in res["offset_px"]],
        "offset_response": res["offset_response"],
        "footprint_px_gt10pct_median": {
            "minime": float(np.median(s10[0])),
            "minian": float(np.median(s10[1])),
        },
        "footprint_px_gt1pct_median": {
            "minime": float(np.median(s01[0])),
            "minian": float(np.median(s01[1])),
        },
        "minime_wall_s": rep.get("wall_total"),
        "frames": res["T"],
    }
