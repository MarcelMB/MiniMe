"""Test 1: footprints from short windows, stitched into one unit list.

For each window length, MiniMe's window 2 runs on every window on its own, the units of
all windows are stitched by footprint similarity, and the stitched footprints are scored
against the truth (recall per category: are late and dim cells found?). Only footprints are
scored here; traces come from ``t2_stream.py``.

    python experiments/streaming/t1_discovery.py --mode subset --windows 1000 3000 6000
    python experiments/streaming/t1_discovery.py --mode fresh --windows 3000

``subset`` runs window 2 on time slices of the baseline factors (seconds per window; the
tile bases were fitted to the whole session, so this flatters late cells). ``fresh`` runs
the whole pipeline on each window's raw files only (the honest version, minutes).

Writes ``processed/stream/t1_<mode>/w<length>/`` (window units, ``stitched.npz``,
``score.json``) and ``processed/stream/t1_<mode>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import stream as st

from minime import io


def run_windows(
    s: st.Session, mode: str, length: int, out: Path, n: int | None = None
) -> list[st.WindowUnits]:
    """Units of every window, or of the first ``n`` (each saved, so an interrupted run resumes)."""
    params = s.params()
    H, W, T = s.shape()
    if mode == "subset":
        fac = s.factors()
        wins = st.time_windows(T, length, fac.block_edges)
    else:
        info, sc = io.probe(s.raw), s.sidecar()
        wins = st.time_windows(T, length, st.segment_starts(info))
    wins = wins[:n]
    units = []
    for t0, t1 in wins:
        path = out / f"win_{t0:06d}_{t1:06d}.npz"
        if path.exists():
            units.append(st.WindowUnits.load(path))
            continue
        if mode == "subset":
            u = st.window_units_subset(fac, params, t0, t1)
        else:
            u = st.window_units_fresh(info, sc, params, t0, t1, out / "runs")
        u.save(path)
        units.append(u)
        print(f"  {t0}-{t1}: {u.A.shape[0]} units, {u.seconds:.0f} s", flush=True)
    return units


def evaluate(
    s: st.Session, units: list[st.WindowUnits], cos_thres: float, shifts: np.ndarray, out: Path
) -> dict:
    """Score the plain union of all window units and the stitched footprints."""
    union = sps.vstack([u.A for u in units]).tocsr()
    A, bright, n_win = st.stitch(units, cos_thres)
    sps.save_npz(out / "stitched_A.npz", A)
    np.savez(out / "stitched.npz", bright=bright, n_windows=n_win)
    sc = st.score(s, A, None, shifts)
    (out / "score.json").write_text(json.dumps(sc, indent=1))
    return {
        "windows": len(units),
        "units_per_window_median": float(np.median([u.A.shape[0] for u in units])),
        "union": st.brief(st.score(s, union, None, shifts)),
        "stitched": st.brief(sc),
        "found_in_n_windows": np.bincount(n_win).tolist(),
        "window_seconds": float(sum(u.seconds for u in units)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--mode", choices=("subset", "fresh"), default="subset")
    ap.add_argument("--windows", type=int, nargs="+", default=[1000, 3000, 6000])
    ap.add_argument("--cos", type=float, default=0.7, help="stitch threshold (footprint cosine)")
    ap.add_argument("--baseline", default="baseline", help="the batch run to start from")
    ap.add_argument("--name", help="output folder (default t1_<mode>)")
    ap.add_argument("--n", type=int, help="first n windows only")
    a = ap.parse_args()

    s = st.Session(a.session, baseline=a.baseline)
    name = a.name or f"t1_{a.mode}"
    root = s.out(name)
    shifts = s.sidecar().shifts
    base = s.baseline_score() if (s.run_dir / "score.json").exists() else None
    summary = {"baseline": base and st.brief(base), "cos": a.cos, "runs": {}}
    for length in a.windows:
        print(f"window {length} frames ({a.mode})", flush=True)
        out = s.out(f"{name}/w{length}")
        units = run_windows(s, a.mode, length, out, a.n)
        if not any(u.A.shape[0] for u in units):
            print("  no units in any window")
            continue
        r = evaluate(s, units, a.cos, shifts, out)
        summary["runs"][str(length)] = r
        b = r["stitched"]
        print(
            f"  stitched {b['n_units']} units: recall {b['recall']:.3f}, precision "
            f"{b['precision']:.3f}, F1 {b['f1']:.3f}, late {b['recall_by_category'].get('late')}",
            flush=True,
        )
        (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"summary: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
