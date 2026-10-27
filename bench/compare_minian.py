"""Compare a minime run with a Minian result on a real session (no ground truth).

    .venv/bin/python bench/compare_minian.py RUN_DIR MINIAN_NPZ [--fps 20] [--label NAME]

``MINIAN_NPZ`` holds ``A (K, H*W)``, ``C (K, T)`` and ``max_proj (H, W)`` (written by
``bench/minian_export.py --final MINIAN_DIR`` under the Minian venv). The two pipelines
register to different references, so Minian's coordinates are mapped onto minime's by
phase-correlating the two max projections. Units are matched one-to-one by footprint
centroid (``score.match_units``, 5 px) and matched traces are correlated (minime ``YrA``
vs Minian ``C``; neither is deconvolved).

Writes ``RUN_DIR/compare_minian.json`` and ``RUN_DIR/compare_minian.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: bench/ is not packaged
from bench.score import MAX_DIST_PX, footprint_centroids, match_units  # noqa: E402
from minime import preprocess  # noqa: E402

MINIME_COLOR = "#2a78d6"  # categorical slot 1 (blue)
MINIAN_COLOR = "#eb6834"  # categorical slot 2 (orange)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


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


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def _contours(ax, A: sparse.csr_matrix, H: int, W: int, off: np.ndarray, color: str) -> None:
    d = A.toarray().reshape(-1, H, W)
    yy, xx = np.arange(H) + off[0], np.arange(W) + off[1]
    for f in d:
        if f.max() > 0:
            ax.contour(xx, yy, f / f.max(), levels=[0.3], colors=color, linewidths=0.8)


def _example_rows(corr: np.ndarray, k: int = 4) -> list[int]:
    """Indices into the matched list at the best, median, 25th percentile and worst corr."""
    order = np.argsort(corr)
    picks = [order[-1], order[len(order) // 2], order[len(order) // 4], order[0]]
    return list(dict.fromkeys(int(i) for i in picks))[:k]


def plot(run: dict, mn: dict, res: dict, s: dict, fps: float, label: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = run["H"], run["W"]
    fig = plt.figure(figsize=(15, 9.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.28)
    fig.suptitle(label, x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    legend = [
        Line2D([], [], color=MINIME_COLOR, lw=2, label=f"MiniMe ({s['minime_units']} units)"),
        Line2D([], [], color=MINIAN_COLOR, lw=2, label=f"Minian ({s['minian_units']} units)"),
    ]

    ax = fig.add_subplot(gs[0, 0])
    mp = run["max_proj"]
    ax.imshow(mp, cmap="gray", vmin=np.percentile(mp, 1), vmax=np.percentile(mp, 99.7))
    _contours(ax, run["A"], H, W, np.zeros(2), MINIME_COLOR)
    _contours(ax, mn["A"], H, W, res["offset_px"], MINIAN_COLOR)
    ax.set_title(
        "Footprints (outline at 30 % of peak) on MiniMe max projection", fontsize=9, color=INK
    )
    ax.set_axis_off()
    ax.legend(handles=legend, loc="lower left", fontsize=7, framealpha=0.85)

    ax = fig.add_subplot(gs[0, 1])
    s10 = res["size10"]
    hi = max(np.percentile(s10[0], 99), np.percentile(s10[1], 99))
    bins = np.linspace(0, hi, 40)
    ax.hist(s10[1], bins, histtype="step", lw=2, color=MINIAN_COLOR)
    ax.hist(s10[0], bins, histtype="step", lw=2, color=MINIME_COLOR)
    for v, c in ((np.median(s10[0]), MINIME_COLOR), (np.median(s10[1]), MINIAN_COLOR)):
        ax.axvline(v, color=c, lw=1, ls="--")
    ax.set_xlabel("footprint size: pixels above 10 % of peak")
    ax.set_ylabel("units")
    ax.set_title(
        f"Footprint size  (median MiniMe {np.median(s10[0]):.0f} px, "
        f"Minian {np.median(s10[1]):.0f} px)",
        fontsize=9,
        color=INK,
    )
    ax.legend(handles=legend, fontsize=7, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[0, 2])
    if len(res["corr"]):
        ax.hist(
            res["corr"],
            np.linspace(min(0.0, np.nanmin(res["corr"])), 1, 30),
            color=MINIME_COLOR,
            edgecolor="white",
            linewidth=1,
        )
        ax.axvline(s["trace_corr_median"], color=INK, lw=1, ls="--")
        ax.text(
            s["trace_corr_median"],
            ax.get_ylim()[1] * 0.95,
            f" median {s['trace_corr_median']:.2f}",
            fontsize=8,
            color=INK,
            va="top",
        )
    ax.set_xlabel("trace correlation (MiniMe YrA vs Minian C)")
    ax.set_ylabel("matched pairs")
    ax.set_title(
        f"Matched units: {s['matched']}  (within {MAX_DIST_PX:.0f} px)", fontsize=9, color=INK
    )
    _style(ax)

    ax = fig.add_subplot(gs[1, :2])
    rows = _example_rows(res["corr"]) if len(res["corr"]) else []
    T = res["T"]
    w = int(min(T, 3 * 60 * fps))
    a = int(T // 2 - w // 2)
    t = (a + np.arange(w)) / fps / 60
    for k, r in enumerate(rows):
        for tr, c in (
            (run["C"][res["ui"][r], a : a + w], MINIME_COLOR),
            (mn["C"][res["mi"][r], a : a + w], MINIAN_COLOR),
        ):
            z = (tr - np.median(tr)) / (np.ptp(tr) or 1)
            ax.plot(t, z - 1.3 * k, color=c, lw=1)
        ax.text(
            t[0],
            -1.3 * k + 0.75,
            f"pair {k + 1}: r = {res['corr'][r]:.2f}",
            fontsize=8,
            color=MUTED,
        )
    ax.set_yticks([])
    ax.set_xlabel("time (min)")
    ax.set_title(
        "Example matched traces (best, median, 25th percentile, worst correlation), "
        "scaled per trace",
        fontsize=9,
        color=INK,
    )
    ax.legend(handles=legend, fontsize=7, frameon=False, loc="upper right")
    _style(ax)
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    lines = [
        ("Units", f"MiniMe {s['minime_units']}   Minian {s['minian_units']}"),
        (
            "Matched",
            f"{s['matched']}  ({100 * s['frac_minian_found']:.0f} % of Minian, "
            f"{100 * s['frac_minime_matched']:.0f} % of MiniMe)",
        ),
        ("Trace corr", f"median {s['trace_corr_median']:.2f}, 25th pct {s['trace_corr_q25']:.2f}"),
        ("Match distance", f"median {s['match_dist_median_px']:.1f} px"),
        (
            "Footprint >1 % peak",
            f"MiniMe {s['footprint_px_gt1pct_median']['minime']:.0f} px, "
            f"Minian {s['footprint_px_gt1pct_median']['minian']:.0f} px",
        ),
        (
            "Alignment",
            f"offset ({s['offset_px'][0]:.1f}, {s['offset_px'][1]:.1f}) px, "
            f"response {s['offset_response']:.2f}",
        ),
        ("MiniMe wall time", f"{s['minime_wall_s']:.0f} s for {s['frames']} frames"),
    ]
    for i, (k, v) in enumerate(lines):
        ax.text(0, 0.95 - i * 0.12, k, fontsize=9, color=MUTED, transform=ax.transAxes)
        ax.text(0, 0.95 - i * 0.12 - 0.05, v, fontsize=10, color=INK, transform=ax.transAxes)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("minian_npz", type=Path)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    run, mn = load_run(a.run_dir), load_minian(a.minian_npz)
    res = compare(run, mn)
    s = summary(run, mn, res)
    (a.run_dir / "compare_minian.json").write_text(json.dumps(s, indent=1) + "\n")
    label = a.label or f"{a.run_dir.name} vs Minian"
    plot(run, mn, res, s, a.fps, label, a.run_dir / "compare_minian.png")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
