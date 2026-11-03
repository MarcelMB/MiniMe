"""Compare the Rust CNMF loop (native or browser) with ``ref_cnmf.py``'s reference.

    python experiments/streaming/check_cnmf.py REF_DIR RUST_DIR [--own-factors]

``REF_DIR`` is ``processed/stream/ref_cnmf_<t0>_<t1>/``; ``RUST_DIR`` holds the Rust
core's outputs. The decisions must be identical, per iteration: the dilated supports
(active pixels, allowed units), which (pixel, unit) pairs the NNQP leaves at zero, each
unit's support size and the size filter's mask, the footprints' supports after the spatial
update, the overlapping unit pairs, the merge groups and the merged supports; and the
final supports and unit count.

Float values get tolerances, relative to the largest reference value of the array (per
trace for ``C``-like arrays). On Python's factors (the native run, ``minime-core cnmf``):
``B`` (``YCt``) is float32 in Accelerate's order, which the port reproduces, so the first
iteration's ``B`` is exact; later steps are float64 sums in other orders (1e-15), and the
NNQP stops on a relative duality gap of 1e-10, so a pixel whose stop falls on another
sweep may move by its own convergence error, ~1e-5 of the footprint maximum (the reference's
largest KKT residual of a non-zero is 8.5e-5 in ``B`` units): 1e-4 on ``w``, footprint
values and ``norm_fac``; the traces and background (float64 sums of float32 products,
then float32) 1e-5; ``B`` of later iterations follows the traces: 1e-5; the trace
correlations of the merge are float64 FFTs of those float32 traces, which differ by a few
float32 ulp (2e-7 of a trace's maximum here), so they move by about as much: 1e-6.
``--own-factors`` (the browser run, which reads its own pass 2 and init) adds pass 2's
reconstruction differences (≤ 5.6e-5 relative, ``check_pass2.py``) and init's trace
differences (≤ 5.7e-6, ``check_init.py``): 1e-3 on everything but ``lam`` (the noise,
≤ 1.2e-6 relative) and the correlations (1e-4, as ``check_init.py``).

Next to the observed differences the closest decisions are printed: the (pixel, unit) pair
of the NNQP nearest the zero/non-zero boundary (a zero's KKT value ``(G w)_k - b_k + lam``,
a non-zero's own value ``w_k``, both in ``B`` units) with the ``B`` difference at it and the
number of pairs whose margin is below the largest ``B`` difference; the unit nearest the
size threshold; the overlapping pair nearest ``loop_merge_corr``. Exits 1 if any check
fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOL_SAME = {
    "B0": 0.0,
    "B": 1e-5,
    "w": 1e-4,
    "data": 1e-4,
    "C": 1e-5,
    "bg": 1e-5,
    "corr": 1e-6,
    "lam": 0.0,
}
TOL_OWN = {
    "B0": 1e-3,
    "B": 1e-3,
    "w": 1e-3,
    "data": 1e-3,
    "C": 1e-3,
    "bg": 1e-3,
    "corr": 1e-4,
    "lam": 1.2e-6,
}


def rel(g: np.ndarray, r: np.ndarray) -> float:
    """Largest difference relative to the reference's largest absolute value."""
    if g.shape != r.shape:
        return np.inf
    if not r.size:
        return 0.0
    g, r = g.astype(np.float64), r.astype(np.float64)
    return float(np.abs(g - r).max() / max(np.abs(r).max(), 1e-300))


def trace_rel(g: np.ndarray, r: np.ndarray) -> float:
    """Largest difference relative to each reference trace's largest absolute value."""
    if g.shape != r.shape:
        return np.inf
    g, r = g.astype(np.float64), r.astype(np.float64)
    scale = np.maximum(np.abs(r).max(axis=1, keepdims=True), 1e-30)
    return float((np.abs(g - r) / scale).max()) if r.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    ap.add_argument("--own-factors", action="store_true", help="the run used its own pass 2")
    a = ap.parse_args()
    ref = a.ref / "ref"
    tol = TOL_OWN if a.own_factors else TOL_SAME
    meta = json.loads((a.ref / "inputs" / "cnmf.json").read_text())
    load = lambda d, n: np.load(d / f"{n}.npy")  # noqa: E731
    ok = True

    def exact(name: str, r: np.ndarray, g: np.ndarray) -> None:
        nonlocal ok
        good = r.shape == g.shape and np.array_equal(r, g)
        diff = int((r != g).sum()) if r.shape == g.shape else f"shape {g.shape} vs {r.shape}"
        print(f"{name:22s} {'ok  ' if good else 'FAIL'} {r.shape}, {diff} differ")
        ok &= good

    def report(name: str, d: float, t: float, what: str = "max rel diff") -> None:
        nonlocal ok
        good = d <= t
        print(f"{name:22s} {'ok  ' if good else 'FAIL'} {what} {d:.3g} (tol {t:g})")
        ok &= good

    def same(name: str) -> None:
        exact(name, load(ref, name), load(a.rust, name))

    margins = []
    for name in ("b_init", "f_init"):
        report(name, rel(load(a.rust, name), load(ref, name)), tol["bg"])
    for i in range(meta["n_iter"]):
        print(f"--- iteration {i}")
        for part in ("active", "indptr", "units"):
            same(f"sub_{i}_{part}")
        w_r, w_g = load(ref, f"w_{i}"), load(a.rust, f"w_{i}")
        exact(f"w_{i} == 0", w_r == 0, w_g == 0)
        for name in (f"nnz_{i}", f"mask_{i}", f"spatial_{i}_indptr", f"spatial_{i}_indices"):
            same(name)
        B_r, B_g = load(ref, f"B_{i}"), load(a.rust, f"B_{i}")
        report(f"B_{i}", rel(B_g, B_r), tol["B0" if i == 0 else "B"])
        report(f"lam_{i}", rel(load(a.rust, f"lam_{i}"), load(ref, f"lam_{i}")), tol["lam"])
        report(f"w_{i}", rel(w_g, w_r), tol["w"])
        for name in (f"spatial_{i}_data", f"norm_fac_{i}"):
            report(name, rel(load(a.rust, name), load(ref, name)), tol["data"])
        for name in (f"C_sp_{i}", f"YrA_{i}"):
            report(name, trace_rel(load(a.rust, name), load(ref, name)), tol["C"])
        for name in (f"b_{i}", f"f_{i}"):
            report(name, rel(load(a.rust, name), load(ref, name)), tol["bg"])
        for name in (f"unit_pairs_{i}", f"merge_labels_{i}", f"merge_counts_{i}"):
            same(name)
        pc_r, pc_g = load(ref, f"unit_corr_{i}"), load(a.rust, f"unit_corr_{i}")
        d = float(np.abs(pc_g - pc_r).max()) if pc_r.shape == pc_g.shape and pc_r.size else 0.0
        report(
            f"unit_corr_{i}", d if pc_r.shape == pc_g.shape else np.inf, tol["corr"], "max |diff|"
        )
        for name in (f"merge_{i}_indptr", f"merge_{i}_indices"):
            same(name)
        report(
            f"merge_{i}_data",
            rel(load(a.rust, f"merge_{i}_data"), load(ref, f"merge_{i}_data")),
            tol["data"],
        )

        # margins of this iteration's decisions
        kkt = load(ref, f"kkt_{i}")
        m = np.where(w_r == 0, kkt, w_r)
        dB = np.abs(B_g - B_r) if B_g.shape == B_r.shape else np.full(B_r.shape, np.inf)
        j = int(np.argmin(m))
        dmax = float(dB.max())
        kind = "zero (KKT)" if w_r[j] == 0 else "non-zero (w)"
        margins.append(
            f"iteration {i} NNQP: closest pair a {kind} {m[j]:.3g} from the boundary, its B diff "
            f"{dB[j]:.3g}; largest B diff {dmax:.3g}, {int((m <= dmax).sum())} pair(s) within it"
        )
        nnz = load(ref, f"nnz_{i}")
        lo, hi = meta["size_thres"]
        dist = np.full(nnz.shape, np.inf)
        if lo:
            dist = np.minimum(dist, np.abs(nnz - lo))
        if hi:
            dist = np.minimum(dist, np.abs(nnz - hi))
        margins.append(
            f"iteration {i} size: closest unit {dist.min():.0f} pixel(s) from size_thres "
            f"{meta['size_thres']}"
        )
        if pc_r.size:
            thr = meta["loop_merge_corr"]
            k = int(np.argmin(np.abs(pc_r - thr)))
            margins.append(
                f"iteration {i} merge: closest overlapping pair {abs(pc_r[k] - thr):.3g} from "
                f"loop_merge_corr {thr}, its diff {abs(pc_g[k] - pc_r[k]):.3g}; "
                f"largest diff {d:.3g}"
            )
    print("--- result")
    for name in ("A_indptr", "A_indices"):
        same(name)
    report("A_data", rel(load(a.rust, "A_data"), load(ref, "A_data")), tol["data"])
    for name in ("C", "YrA"):
        report(name, trace_rel(load(a.rust, name), load(ref, name)), tol["C"])
    for name in ("b", "f"):
        report(name, rel(load(a.rust, name), load(ref, name)), tol["bg"])
    s_r = json.loads((ref / "summary.json").read_text())
    own = a.rust / "summary_cnmf.json"  # the browser run's (its summary.json is pass 1's)
    s_g = json.loads((own if own.exists() else a.rust / "summary.json").read_text())
    n_r, n_g = s_r["units_final"], s_g["units_final"]
    good = n_r == n_g
    print(f"{'units_final':22s} {'ok  ' if good else 'FAIL'} {n_g} vs {n_r}")
    ok &= good
    for line in margins:
        print("margin " + line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
