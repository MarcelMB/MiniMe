"""Time pass-2 variants on tiles of a kept tile store, one tile at a time, single-threaded.

    .venv/bin/python bench/pass2_variants.py RUN_DIR [--tiles 0 17 ...] [--noise-frames 20000]

``RUN_DIR`` is a caanfast run that kept ``tiles/`` (the uint8 tile store) and ``factors/``.
Each tile is compressed the way a pass-2 worker does it (BLAS on one thread) with:

- ``base``: all frames for the noise
- ``noise_sub``: noise from ``--noise-frames`` frames (``Params.noise_frames``)

and reports per variant: read and compress seconds, rank, ``sigma`` relative to ``base``
(median and max |ratio - 1|) and the captured energy ``sum ||W||^2 / sum energy``. Default
tiles: the 4 highest-rank tiles of the saved factors plus 4 spread over the rest. Writes
``RUN_DIR/pass2_variants.json``.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"  # a pass-2 worker runs single-threaded

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caanfast import compress  # noqa: E402
from caanfast.params import Params  # noqa: E402
from caanfast.tiles import TileStore, tile_grid  # noqa: E402

VARIANTS = {
    "base": {},
    "noise_sub": {"noise_frames": None},  # filled from --noise-frames
}


def default_tiles(ranks: list[int]) -> list[int]:
    order = np.argsort(ranks)[::-1]
    rest = order[4:]
    return sorted({*order[:4].tolist(), *rest[np.linspace(0, len(rest) - 1, 4).astype(int)]})


def n_parts(store_dir: Path) -> int:
    return len(list(store_dir.glob("tile_000.part*")))


def run_tile(store: TileStore, i: int, p: Params, noise_frames: int) -> dict:
    t0 = time.perf_counter()
    raw = store.read_tile(i)
    row: dict = {"tile": int(i), "read_s": time.perf_counter() - t0}
    sig0 = None
    for name, kw in VARIANTS.items():
        kw = {k: (noise_frames if v is None else v) for k, v in kw.items()}
        t0 = time.perf_counter()
        tf = compress.compress_tile(
            raw,
            k0=p.svd_k0,
            n_iter=p.svd_power_iter,
            rho_min=p.rank_rho_min,
            z=p.rank_z,
            seed=i,
            noise_freq=p.noise_freq,
            **kw,
        )
        dt = time.perf_counter() - t0
        sig0 = tf.sigma if sig0 is None else sig0
        dev = np.abs(tf.sigma / np.maximum(sig0, 1e-12) - 1)
        row[name] = {
            "seconds": dt,
            "rank": tf.rank,
            "sigma_dev_median": float(np.median(dev)),
            "sigma_dev_max": float(dev.max()),
            "captured": float((tf.W.astype(np.float64) ** 2).sum() / tf.energy.sum()),
        }
        print(
            f"tile {i:3d} {name:<10s} {dt:7.1f} s  rank {tf.rank:3d}  "
            f"sigma dev {row[name]['sigma_dev_median']:.4f}  "
            f"captured {row[name]['captured']:.4f}",
            flush=True,
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path)
    ap.add_argument("--tiles", type=int, nargs="*")
    ap.add_argument("--noise-frames", type=int, default=20000)
    a = ap.parse_args()
    meta = json.loads((a.run / "factors" / "meta.json").read_text())
    p = Params.from_json(a.run / "params.json")
    grid = tile_grid(meta["H"], meta["W"], meta["tile"])
    store = TileStore(a.run / "tiles", grid, n_parts(a.run / "tiles"))
    tiles = a.tiles or default_tiles(meta["ranks"])
    rows = [run_tile(store, int(i), p, a.noise_frames) for i in tiles]
    tot = {v: sum(r[v]["seconds"] for r in rows) for v in VARIANTS}
    print(
        "total s:",
        {k: round(v, 1) for k, v in tot.items()},
        "read s:",
        round(sum(r["read_s"] for r in rows), 1),
    )
    out = {"tiles": rows, "total_seconds": tot, "noise_frames": a.noise_frames}
    (a.run / "pass2_variants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
