"""Compare the Rust init stage (native or browser) with ``ref_init.py``'s reference.

    python experiments/streaming/check_init.py REF_DIR RUST_DIR [--own-factors]

``REF_DIR`` is ``processed/stream/ref_init_<t0>_<t1>/``; ``RUST_DIR`` holds the Rust
core's outputs. The decisions must be identical: the units ``initA`` keeps and each
footprint's support (CSR ``indptr`` and ``indices``), the overlapping unit pairs, the
merge groups (labels in order of first appearance, and their sizes) and the merged
footprints' supports.

Float values get tolerances. On Python's factors (the native run, ``minime-core init``):
the window correlations and trace correlations are float64 bilinear forms and FFTs summed
in another order, 1e-9; a footprint value is the float32 of such a correlation, so it may
round to the neighbouring float32: 1.2e-7 (one ulp below 1). The traces solve
``(A A^T) C = A Y``, where ``A Y`` is float32 per tile: Rust copies scipy's sum order but
not Accelerate's sgemm kernel, a few float32 ulp, times ``cond(G)`` (13.7 here): 1e-5 of
each trace's largest value. Centroids average footprint values: 1e-5 px.
``--own-factors`` (the browser run, which reads its own pass 2) adds pass 2's differences,
up to 5.6e-5 relative in a tile's reconstruction (``check_pass2.py``): 1e-4 on
correlations and footprint values (as ``check_seeds.py`` allows for correlations), 1e-3
on traces (the reconstruction error over a footprint, times ``cond(G)``) and 1e-3 px on
centroids. The decisions above must still come out identical; the margin of the closest
decision to its threshold (``inita_corr`` on the window correlations, ``init_merge_corr``
on the overlapping pairs' trace correlations) is printed next to the observed
differences, with the difference at that decision itself.

A window pixel that is constant over the window (zero energy) has ``norm2`` = 0 up to the
rSVD's rounding (``check_seeds.py``), so its correlation is rounding over rounding (two
such pixels: Python -0.05 and 0.06, Chrome 0.11 and 0.23). Those pixels are listed apart:
they must stay below ``inita_corr`` in both runs, which keeps them out of every footprint.
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from check_seeds import flat_pixels

TOL_SAME = {"corr": 1e-9, "data": 1.2e-7, "C": 1e-5, "cent": 1e-5}
TOL_OWN = {"corr": 1e-4, "data": 1e-4, "C": 1e-3, "cent": 1e-3}
EXACT = (
    "initA_indptr",
    "initA_indices",
    "unit_pairs",
    "merge_labels",
    "merge_counts",
    "A_merge_indptr",
    "A_merge_indices",
)


def max_abs(g: np.ndarray, r: np.ndarray) -> float:
    if g.shape != r.shape:
        return np.inf
    return float(np.abs(g.astype(np.float64) - r.astype(np.float64)).max()) if r.size else 0.0


def trace_rel(g: np.ndarray, r: np.ndarray) -> float:
    """Largest difference relative to each reference trace's largest absolute value."""
    if g.shape != r.shape:
        return np.inf
    g, r = g.astype(np.float64), r.astype(np.float64)
    scale = np.maximum(np.abs(r).max(axis=1, keepdims=True), 1e-30)
    return float((np.abs(g - r) / scale).max()) if r.size else 0.0


def window_constant(ref: Path, meta: dict) -> np.ndarray:
    """Which ``initA_window_corr`` entries belong to a constant (zero-energy) pixel."""
    H, W, wnd = meta["H"], meta["W"], meta["inita_wnd"]
    flat = np.load(ref / "inputs" / "fac_energy.npy") == 0  # tile order
    const = np.zeros(H * W, bool)
    const[flat_pixels(ref, H, W)[flat]] = True
    dy, dx = np.mgrid[-wnd : wnd + 1, -wnd : wnd + 1]
    inside = dy**2 + dx**2 <= wnd**2
    dy, dx = dy[inside], dx[inside]
    out = []
    for h, w in np.load(ref / "inputs" / "init_seeds.npy"):
        hh, ww = h + dy, w + dx
        ok = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
        out.append(const[hh[ok] * W + ww[ok]])
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    ap.add_argument("--own-factors", action="store_true", help="the run used its own pass 2")
    a = ap.parse_args()
    ref = a.ref / "ref"
    tol = TOL_OWN if a.own_factors else TOL_SAME
    meta = json.loads((a.ref / "inputs" / "init.json").read_text())
    load = lambda d, n: np.load(d / f"{n}.npy")  # noqa: E731
    ok = True

    def report(name: str, d: float, t: float, what: str) -> None:
        nonlocal ok
        good = d <= t
        print(f"{name:18s} {'ok  ' if good else 'FAIL'} {what} {d:.3g} (tol {t:g})")
        ok &= good

    for name in EXACT:
        r, g = load(ref, name), load(a.rust, name)
        good = r.shape == g.shape and np.array_equal(r, g)
        diff = int((r != g).sum()) if r.shape == g.shape else f"shape {g.shape} vs {r.shape}"
        print(f"{name:18s} {'ok  ' if good else 'FAIL'} {r.shape}, {diff} differ")
        ok &= good

    win_r, win_g = load(ref, "initA_window_corr"), load(a.rust, "initA_window_corr")
    const = window_constant(a.ref, meta)
    if win_g.shape != win_r.shape:
        win_g = np.full_like(win_r, np.inf)
    report("initA_window_corr", max_abs(win_g[~const], win_r[~const]), tol["corr"], "max |diff|")
    cmax = float(np.maximum(win_r[const], win_g[const]).max()) if const.any() else -np.inf
    good = cmax < meta["inita_corr"]
    print(
        f"{'  constant pixels':18s} {'ok  ' if good else 'FAIL'} {int(const.sum())} window "
        f"entries, largest correlation in either run {cmax:.3g} (must stay < inita_corr)"
    )
    ok &= good
    for name in ("initA_data", "A_merge_data"):
        report(name, max_abs(load(a.rust, name), load(ref, name)), tol["data"], "max |diff|")
    for name in ("C_initC", "C_merge"):
        report(name, trace_rel(load(a.rust, name), load(ref, name)), tol["C"], "max rel diff")
    pc_r, pc_g = load(ref, "unit_corr"), load(a.rust, "unit_corr")
    report("unit_corr", max_abs(pc_g, pc_r), tol["corr"], "max |diff|")
    for name in ("initA_centroids", "merge_centroids"):
        report(name, max_abs(load(a.rust, name), load(ref, name)), tol["cent"], "max |diff| px")

    dist = np.where(const, np.inf, np.abs(win_r - meta["inita_corr"]))
    i = int(np.argmin(dist))
    d_all = max_abs(win_g[~const], win_r[~const])
    near = int((dist <= d_all).sum())
    print(
        f"margin initA: closest window pixel {dist[i]:.3g} from inita_corr "
        f"{meta['inita_corr']}, its diff {abs(win_g[i] - win_r[i]):.3g}; largest diff "
        f"{d_all:.3g}, {near} pixel(s) within it of the threshold"
    )
    if pc_r.size:
        j = int(np.argmin(np.abs(pc_r - meta["init_merge_corr"])))
        print(
            f"margin merge: closest overlapping pair {abs(pc_r[j] - meta['init_merge_corr']):.3g}"
            f" from init_merge_corr {meta['init_merge_corr']}, its diff "
            f"{abs(pc_g[j] - pc_r[j]):.3g}; largest diff {max_abs(pc_g, pc_r):.3g}"
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
