"""Compare the Rust seed stage (native or browser) with ``ref_seeds.py``'s reference.

    python experiments/streaming/check_seeds.py REF_DIR RUST_DIR [--own-factors]

``REF_DIR`` is ``processed/stream/ref_seeds_<t0>_<t1>/``; ``RUST_DIR`` holds the Rust
core's outputs. The seed sets must be identical: the initial seeds and their window
counts, the PNR mask, the opt-in GMM mask, the merge pairs, the merge mask and the merged
positions, and the k-means++ draws (MT19937, bit for bit).

Float values get tolerances. On Python's own factors (the native run, ``minime-core
seeds``): the traces are float32 sums whose order Rust copies (sequential FMAs) but whose
BLAS kernel it cannot, so a few ulp in a few rows: 1e-6 relative on PNR. The mixtures are
fitted to those values (and use a BLAS gemv for ``resp.T @ x``), so their parameters move
by the PNR ulps averaged over the seeds: 1e-9 relative. The low-pass is another FFT and the
Gram matrices and bilinear forms sum in another order: float64 rounding, 1e-9.
``--own-factors`` (the browser run, whose seeds read its own pass 2) adds pass 2's own
differences, up to 5.6e-5 relative in a tile's reconstruction (``check_pass2.py``), which
a dim pixel's trace can amplify: 1e-3 relative on PNR and ``norm2``, 1e-4 on the mixture
parameters and correlations. The decisions above must still come out identical.

``norm2`` of a pixel that is constant over the window (zero energy, e.g. a dead pixel) is
zero up to the rounding of the rSVD's ``||W_i||^2`` (Python 2.8e-11, Rust 8e-27 for two
such pixels), so a relative error means nothing there: those pixels are compared with the
same tolerance relative to the median ``norm2`` instead. Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from minime.tiles import tile_grid

TOL_SAME = {"pnr": 1e-6, "gmm": 1e-9, "corr": 1e-9, "norm2": 1e-9}
TOL_OWN = {"pnr": 1e-3, "gmm": 1e-4, "corr": 1e-4, "norm2": 1e-3}
EXACT = (
    "seeds_init",
    "mask_pnr",
    "mask_gmm",
    "merge_pairs",
    "mask_mrg",
    "seeds_merged",
    "gmm_draws",
)


def json_width(ref: Path) -> int:
    return int(json.loads((ref / "inputs" / "seeds.json").read_text())["W"])


def flat_pixels(ref: Path, H: int, W: int) -> np.ndarray:
    """Flat image index of every pixel in tile order (pass 2's per-pixel arrays)."""
    tile = int(json.loads((ref / "inputs" / "seeds.json").read_text())["tile"])
    return np.concatenate(
        [
            (np.arange(t.y0, t.y1)[:, None] * W + np.arange(t.x0, t.x1)).ravel()
            for t in tile_grid(H, W, tile)
        ]
    )


def rel(g: np.ndarray, r: np.ndarray) -> float:
    g, r = np.asarray(g, np.float64), np.asarray(r, np.float64)
    return float((np.abs(g - r) / np.maximum(np.abs(r), 1e-300)).max()) if r.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    ap.add_argument("--own-factors", action="store_true", help="the run used its own pass 2")
    a = ap.parse_args()
    ref = a.ref / "ref"
    tol = TOL_OWN if a.own_factors else TOL_SAME
    load = lambda d, n: np.load(d / f"{n}.npy")  # noqa: E731
    ok = True

    for name in EXACT:
        r, g = load(ref, name), load(a.rust, name)
        good = r.shape == g.shape and np.array_equal(r, g)
        diff = int((r != g).sum()) if r.shape == g.shape else f"shape {g.shape} vs {r.shape}"
        print(f"{name:12s} {'ok  ' if good else 'FAIL'} {r.shape}, {diff} differ")
        ok &= good

    r, g = load(ref, "pnr"), load(a.rust, "pnr")
    d = rel(g, r) if r.shape == g.shape else np.inf
    good = d <= tol["pnr"]
    print(f"{'pnr':12s} {'ok  ' if good else 'FAIL'} max rel diff {d:.3g}")
    ok &= good

    for name in ("gmm_pnr", "gmm_refine"):
        d = rel(load(a.rust, name), load(ref, name))
        good = d <= tol["gmm"]
        print(
            f"{name:12s} {'ok  ' if good else 'FAIL'} max rel diff {d:.3g} (weights, means, vars)"
        )
        ok &= good

    r, g = load(ref, "merge_corr"), load(a.rust, "merge_corr")
    d = float(np.abs(g - r).max()) if r.shape == g.shape and r.size else np.inf
    good = d <= tol["corr"]
    print(f"{'merge_corr':12s} {'ok  ' if good else 'FAIL'} max |diff| {d:.3g}")
    ok &= good

    r, g = load(ref, "corr_norm2"), load(a.rust, "corr_norm2")
    flat = np.load(a.ref / "inputs" / "fac_energy.npy") == 0  # tile order, as norm2 is not
    const = np.zeros(r.size, bool)
    H, W = r.size // json_width(a.ref), json_width(a.ref)
    const[flat_pixels(a.ref, H, W)[flat]] = True
    d = rel(g[~const], r[~const])
    d0 = float(np.abs(g[const] - r[const]).max() / np.median(r)) if const.any() else 0.0
    good = d <= tol["norm2"] and d0 <= tol["norm2"]
    print(
        f"{'corr_norm2':12s} {'ok  ' if good else 'FAIL'} max rel diff {d:.3g}; "
        f"{int(const.sum())} constant pixels, max |diff| / median {d0:.3g}"
    )
    ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
