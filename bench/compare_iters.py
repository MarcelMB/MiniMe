"""Compare two caanfast runs of one session that differ in CNMF iterations (e.g. n_iter 1 vs 2).

    .venv/bin/python bench/compare_iters.py RUN_A RUN_B [--frac 0.1] [--out FILE.json]

Units of the two runs are matched one-to-one by footprint centroid (``score.match_units``,
5 px; both runs share the motion template, so no offset is needed). For matched pairs it
reports footprint IoU (pixels above ``frac`` of each unit's peak), weighted IoU
(``sum min / sum max`` of the peak-normalised footprints) and Pearson correlation of the
``C`` and ``YrA`` traces. Pairs are split by whether RUN_A's unit was formed by RUN_A's last
unit merge: such a unit is the mean of its members' peak-normalised rows, so its maximum is
below 1, while a unit straight from the spatial update has maximum 1. With RUN_A the
n_iter=1 run this separates the units merged in iteration 0 from the rest; the rest is
further split into units whose footprint overlaps a merged unit and isolated ones.

Also reports the cnmf stage seconds and unit counts from ``report.json`` and, when the
session is synthetic (``<raw>/../meta.json`` with a ``seed``), the ``bench/score.py`` score of
each run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: bench/ is not packaged
from bench.score import (  # noqa: E402
    MAX_DIST_PX,
    footprint_centroids,
    match_units,
    score_pipeline_dir,
)

MERGED_MAX_TOL = 1e-4  # a row maximum below 1 - tol marks a unit formed by unit_merge

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict:
    """Footprints, traces, frame size and report of one run directory."""
    meta = json.loads((run_dir / "factors" / "meta.json").read_text())
    return {
        "dir": run_dir,
        "A": sparse.load_npz(run_dir / "cnmf" / "A.npz").tocsr(),
        "C": np.load(run_dir / "cnmf" / "C.npy"),
        "YrA": np.load(run_dir / "cnmf" / "YrA.npy"),
        "H": int(meta["H"]),
        "W": int(meta["W"]),
        "report": json.loads((run_dir / "report.json").read_text()),
        "params": json.loads((run_dir / "params.json").read_text()),
    }


def stage_seconds(report: dict, stage: str) -> float:
    """Wall seconds of ``stage`` in a ``report.json`` (summed if it appears twice)."""
    return float(sum(r["seconds"] for r in report["stages"] if r["stage"] == stage))


def synth_meta(run: dict) -> Path | None:
    """``<raw>/../meta.json`` if the run's session is synthetic, else None."""
    p = Path(run["params"].get("raw_dir", "")).parent / "meta.json"
    if p.is_file() and "seed" in json.loads(p.read_text()):
        return p
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def row_max(A: sparse.csr_matrix) -> np.ndarray:
    return A.max(axis=1).toarray().ravel()


def touches(A: sparse.csr_matrix, sel: np.ndarray) -> np.ndarray:
    """(K,) bool: the unit's footprint shares a pixel with another unit in ``sel``."""
    B = sparse.csr_matrix(A > 0, dtype=np.float32)
    ov = (B @ B[np.flatnonzero(sel)].T).toarray() > 0
    ov[np.flatnonzero(sel), np.arange(int(sel.sum()))] = False  # a unit does not touch itself
    return ov.any(axis=1)


def footprint_ious(
    Aa: sparse.csr_matrix, Ab: sparse.csr_matrix, ia: np.ndarray, ib: np.ndarray, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Binary IoU (above ``frac`` of peak) and weighted IoU of the pairs ``(ia[k], ib[k])``."""
    a = Aa[ia].toarray().astype(np.float64)
    b = Ab[ib].toarray().astype(np.float64)
    a /= a.max(axis=1, keepdims=True)
    b /= b.max(axis=1, keepdims=True)
    ma, mb = a > frac, b > frac
    iou = (ma & mb).sum(axis=1) / np.maximum((ma | mb).sum(axis=1), 1)
    wiou = np.minimum(a, b).sum(axis=1) / np.maximum(a, b).sum(axis=1)
    return iou, wiou


def pair_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two (n, T) arrays (NaN for a flat row)."""
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(axis=1) / den


def quantiles(v: np.ndarray) -> dict:
    """min / q05 / median / mean of the finite values, and their count."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "q05": float(np.quantile(v, 0.05)),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
    }


def group_stats(metrics: dict[str, np.ndarray], sel: np.ndarray) -> dict:
    return {k: quantiles(v[sel]) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(ra: dict, rb: dict, frac: float = 0.1, max_dist: float = MAX_DIST_PX) -> dict:
    """The comparison dict for runs ``ra`` (reference, e.g. n_iter=1) and ``rb``."""
    if (ra["H"], ra["W"]) != (rb["H"], rb["W"]):
        raise ValueError("runs have different frame sizes")
    ca = footprint_centroids(ra["A"], ra["H"], ra["W"])
    cb = footprint_centroids(rb["A"], rb["H"], rb["W"])
    ia, ib = match_units(ca, cb, max_dist)
    iou, wiou = footprint_ious(ra["A"], rb["A"], ia, ib, frac)
    metrics = {
        "centroid_shift_px": np.hypot(*(ca[ia] - cb[ib]).T),
        "iou": iou,
        "weighted_iou": wiou,
        "corr_C": pair_corr(ra["C"][ia], rb["C"][ib]),
        "corr_YrA": pair_corr(ra["YrA"][ia], rb["YrA"][ib]),
    }
    merged_a = row_max(ra["A"]) < 1 - MERGED_MAX_TOL
    sel_m = merged_a[ia]
    sel_t = touches(ra["A"], merged_a)[ia] & ~sel_m
    out = {
        "run_a": str(ra["dir"]),
        "run_b": str(rb["dir"]),
        "n_iter": [ra["params"].get("n_iter"), rb["params"].get("n_iter")],
        "cnmf_seconds": [stage_seconds(r["report"], "cnmf") for r in (ra, rb)],
        "wall_total": [float(r["report"]["wall_total"]) for r in (ra, rb)],
        "counts": [r["report"]["counts"] for r in (ra, rb)],
        "n_units": [int(ra["A"].shape[0]), int(rb["A"].shape[0])],
        "n_matched": int(ia.size),
        "unmatched": [int(ra["A"].shape[0] - ia.size), int(rb["A"].shape[0] - ib.size)],
        "n_merged_in_a": int(merged_a.sum()),
        "iou_frac": frac,
        "all": group_stats(metrics, np.ones(ia.size, bool)),
        "merged_in_a": group_stats(metrics, sel_m),
        "not_merged_in_a": group_stats(metrics, ~sel_m),
        "not_merged_touching": group_stats(metrics, sel_t),
        "not_merged_isolated": group_stats(metrics, ~sel_m & ~sel_t),
    }
    meta = synth_meta(ra)
    if meta is not None:
        out["score"] = [score_pipeline_dir(r["dir"]) for r in (ra, rb)]
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(s: dict, key: str) -> str:
    q = s[key]
    if q.get("n", 0) == 0:
        return "-"
    return f"{q['median']:.3f} (q05 {q['q05']:.3f}, min {q['min']:.3f})"


def summary_lines(res: dict) -> list[str]:
    a, b = res["n_iter"]
    lines = [
        f"n_iter {a} vs {b}: cnmf {res['cnmf_seconds'][0]:.1f} s vs "
        f"{res['cnmf_seconds'][1]:.1f} s, wall {res['wall_total'][0]:.1f} s vs "
        f"{res['wall_total'][1]:.1f} s",
        f"units {res['n_units'][0]} vs {res['n_units'][1]}, matched {res['n_matched']}, "
        f"unmatched {res['unmatched'][0]} / {res['unmatched'][1]}, "
        f"merged by A's last merge {res['n_merged_in_a']}",
    ]
    groups = ("all", "merged_in_a", "not_merged_in_a", "not_merged_touching")
    for grp in (*groups, "not_merged_isolated"):
        g = res[grp]
        lines.append(f"[{grp}] n={g['iou'].get('n', 0)}")
        for key in ("iou", "weighted_iou", "corr_C", "corr_YrA", "centroid_shift_px"):
            lines.append(f"  {key:18s} median {_fmt(g, key)}")
    for name, s in zip(("A", "B"), res.get("score", []), strict=False):
        lines.append(
            f"score {name}: recall {s['recall']:.3f} precision {s['precision']:.3f} "
            f"F1 {s['f1']:.3f} trace corr {s['trace_corr_median']:.4f} / "
            f"{s['trace_corr_q25']:.4f} (median / q25), units {s['n_units']}"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_a", type=Path, help="reference run (e.g. n_iter=1)")
    ap.add_argument("run_b", type=Path, help="run to compare (e.g. n_iter=2)")
    ap.add_argument("--frac", type=float, default=0.1, help="IoU support: fraction of peak")
    ap.add_argument("--out", type=Path, default=None, help="write the comparison JSON here")
    args = ap.parse_args(argv)
    res = compare(load_run(args.run_a), load_run(args.run_b), args.frac)
    print("\n".join(summary_lines(res)))
    if args.out is not None:
        args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
