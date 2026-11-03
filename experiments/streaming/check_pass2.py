"""Compare the Rust pass 2 (native or browser) with ``ref_pass2.py``'s reference.

    python experiments/streaming/check_pass2.py REF_DIR RUST_DIR

``REF_DIR`` is ``processed/stream/ref_pass2_<t0>_<t1>/``; ``RUST_DIR`` holds the Rust
core's outputs. Per tile the kept rank and the last rSVD rank must be identical, the mean
and energy exact; the rest are float32 GEMMs and decompositions in another order, so they
get tolerances: noise (FFT), singular values, the reconstruction ``W V^T`` (relative
Frobenius error) and the temporal subspace (smallest cosine of the principal angles
between the two ``V``). Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from minime.tiles import tile_grid

TOL = {"sigma": 1e-5, "s": 1e-4, "recon": 1e-4, "subspace": 1e-4}


def split(flat: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    return np.split(flat, np.cumsum(sizes)[:-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    a = ap.parse_args()
    ref = a.ref / "ref"
    params = json.loads((a.ref / "inputs" / "pass2.json").read_text())
    T = params["T"]
    P = [t.pixels for t in tile_grid(params["H"], params["W"], params["tile"])]
    load = lambda d, n: np.load(d / f"{n}.npy")  # noqa: E731
    ok = True

    for name in ("rank", "k"):
        r, g = load(ref, name), load(a.rust, name)
        good = np.array_equal(r, g)
        print(
            f"{name:9s} {'ok  ' if good else 'FAIL'} {int((r != g).sum())} of {r.size} tiles differ"
        )
        ok &= good
    if not ok:
        sys.exit(1)
    rank = load(ref, "rank").tolist()

    for name in ("mean", "energy"):
        d = np.abs(load(a.rust, name).astype(np.float64) - load(ref, name))
        good = d.max() == 0
        print(f"{name:9s} {'ok  ' if good else 'FAIL'} max |diff| {d.max():.3g}")
        ok &= good

    r, g = load(ref, "sigma"), load(a.rust, "sigma")
    d = float((np.abs(g.astype(np.float64) - r) / np.maximum(r, 1e-12)).max())
    good = d <= TOL["sigma"]
    print(f"{'sigma':9s} {'ok  ' if good else 'FAIL'} max rel diff {d:.3g}")
    ok &= good

    r, g = load(ref, "s").astype(np.float64), load(a.rust, "s")
    d = float((np.abs(g - r) / r).max())
    good = d <= TOL["s"]
    print(f"{'s':9s} {'ok  ' if good else 'FAIL'} max rel diff {d:.3g}")
    ok &= good

    sizes_w = [p * k for p, k in zip(P, rank, strict=True)]
    sizes_v = [T * k for k in rank]
    Wr, Wg = split(load(ref, "W"), sizes_w), split(load(a.rust, "W"), sizes_w)
    Vr, Vg = split(load(ref, "V"), sizes_v), split(load(a.rust, "V"), sizes_v)
    recon, angle = np.zeros(len(P)), np.zeros(len(P))
    for i, (p, k) in enumerate(zip(P, rank, strict=True)):
        wr, wg = Wr[i].reshape(p, k).astype(np.float64), Wg[i].reshape(p, k).astype(np.float64)
        vr, vg = Vr[i].reshape(T, k).astype(np.float64), Vg[i].reshape(T, k).astype(np.float64)
        Yr, Yg = wr @ vr.T, wg @ vg.T
        recon[i] = np.linalg.norm(Yg - Yr) / np.linalg.norm(Yr)
        qr, qg = np.linalg.qr(vr)[0], np.linalg.qr(vg)[0]
        angle[i] = 1.0 - np.linalg.svd(qr.T @ qg, compute_uv=False).min()
    for name, v in (("recon", recon), ("subspace", angle)):
        good = v.max() <= TOL[name]
        print(
            f"{name:9s} {'ok  ' if good else 'FAIL'} max {v.max():.3g} (tile {int(v.argmax())}), "
            f"median {np.median(v):.3g}"
        )
        ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
