"""Compare the Rust motion stage (native or browser) with ``ref_motion.py``'s reference.

    python experiments/streaming/check_motion.py REF_DIR RUST_DIR

``REF_DIR`` is ``processed/stream/ref_motion_<t0>_<t1>/``; ``RUST_DIR`` holds the Rust
core's ``.npy`` outputs. The registration and prepared images must be bit-identical; the
template, shifts and quality pass through numpy's float32 FFT, which Rust does not
reproduce, so they get tolerances. Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# (name, largest allowed absolute difference)
CHECKS = [
    ("frames_t", 0.0),
    ("reg", 0.0),
    ("prep", 0.0),
    ("template", None),  # relative, see below
    ("shifts", 1e-3),
    # numpy's float32 dot products are off by ~1e-5 from float64; Rust sums in float64
    ("quality", 5e-5),
]
TEMPLATE_REL = 1e-5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    a = ap.parse_args()
    ref = a.ref / "ref"
    ok = True
    for name, tol in CHECKS:
        r, g = np.load(ref / f"{name}.npy"), np.load(a.rust / f"{name}.npy")
        if r.shape != g.shape:
            print(f"{name:10s} FAIL shape {g.shape} != {r.shape}")
            ok = False
            continue
        d = np.abs(g.astype(np.float64) - r.astype(np.float64))
        dmax = float(d.max()) if d.size else 0.0
        if name == "template":
            scale = float(np.abs(r).max())
            good = dmax <= TEMPLATE_REL * scale
            note = f"max |diff| {dmax:.3g} = {dmax / scale:.2g} of max |ref|"
        else:
            good = dmax <= tol
            note = f"max |diff| {dmax:.3g}, {int((d > 0).sum())} of {d.size} differ"
        if name == "shifts":
            whole = np.abs(np.round(g / 2) - np.round(r / 2)).max()  # integer peak, down = 2
            note += f", peak moved in {int((np.abs(g - r) > 0.5).any(axis=1).sum())} frames"
            note += f" (max whole-px change {whole:g})"
        print(f"{name:10s} {'ok  ' if good else 'FAIL'} {note}")
        ok &= good
    if (ref / "fill.npy").exists():  # template fill: uint8, from the template frames' shifts
        r, g = np.load(ref / "fill.npy"), np.load(a.rust / "fill.npy")
        d = np.abs(g.astype(np.int16) - r.astype(np.int16)) if r.shape == g.shape else None
        # a shift ~5e-6 px off (the FFT) can move one warped pixel by a grey level
        good = d is not None and d.max() <= 1 and (d > 0).sum() <= 1e-4 * d.size
        note = (
            "shape differs"
            if d is None
            else f"max |diff| {d.max()}, {int((d > 0).sum())} px differ"
        )
        print(f"{'fill':10s} {'ok  ' if good else 'FAIL'} {note}")
        ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
