"""Compare the Rust pass 1 (native or browser) with ``ref_pass1.py``'s reference.

    python experiments/streaming/check_pass1.py REF_DIR RUST_DIR

``REF_DIR`` is ``processed/stream/ref_pass1_<t0>_<t1>/``; ``RUST_DIR`` holds the Rust
core's outputs. Max projection, block maxima, block edges, shifts and the tile store
(by CRC-32 per tile) must be identical. Quality passes through an FFT round trip of the
template (numpy float64, Rust float32), so it gets a tolerance. When a tile's CRC differs
and ``RUST_DIR/tiles/`` exists, the differing frames and pixels are reported.
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from minime.tiles import TileStore, tile_grid

# (name, largest allowed absolute difference)
CHECKS = [
    ("block_edges", 0.0),
    ("max_proj", 0.0),
    ("block_max", 0.0),
    ("quality", 5e-5),
    ("tile_crc", 0.0),
]


def tile_detail(a: Path, rust: Path, bad: np.ndarray) -> str:
    """Where the first differing tile differs, from both tile stores."""
    params = json.loads((a / "inputs" / "params.json").read_text())
    grid = tile_grid(params["H"], params["W"], params["tile"])
    k = int(bad[0])
    r = TileStore(a / "ref" / "tiles", grid, 1).read_tile(k)
    g = TileStore(rust / "tiles", grid, 1).read_tile(k)
    if r.shape != g.shape:
        return f"tile {k}: shape {g.shape} != {r.shape}"
    d = np.abs(r.astype(np.int16) - g.astype(np.int16))
    frames = np.flatnonzero(d.reshape(d.shape[0], -1).any(axis=1))
    return (
        f"tile {k}: {int((d > 0).sum())} pixels in {frames.size} frames differ "
        f"(first frame {frames[0] if frames.size else None}, max {int(d.max())})"
    )


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
            print(f"{name:12s} FAIL shape {g.shape} != {r.shape}")
            ok = False
            continue
        d = np.abs(g.astype(np.float64) - r.astype(np.float64))
        dmax = float(d.max()) if d.size else 0.0
        good = dmax <= tol
        note = f"max |diff| {dmax:.3g}, {int((d > 0).sum())} of {d.size} differ"
        if name == "tile_crc" and not good and (a.rust / "tiles").is_dir():
            note += "; " + tile_detail(a.ref, a.rust, np.flatnonzero(d))
        print(f"{name:12s} {'ok  ' if good else 'FAIL'} {note}")
        ok &= good
    shifts = np.load(a.rust / "shifts.npy")
    given = np.load(a.ref / "inputs" / "shifts.npy")
    print(f"{'shifts':12s} max |diff| vs the given shifts {np.abs(shifts - given).max():.3g}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
