"""Reference outputs of MiniMe's pass 2 on one window, for checking the Rust core.

Runs ``compress.compress_tile`` on every tile of ``ref_pass1.py``'s tile store exactly as
``pass2.pass2_worker`` does (seed = tile index, the session's parameters) and records every
Gaussian draw of the rSVD, so the Rust core can be run on the same test matrices. Writes,
under ``processed/stream/ref_pass2_<t0>_<t1>/``:

* ``inputs/``: ``pass2.json`` (``H``, ``W``, ``T``, ``tile`` and the pass-2 parameters),
  ``omega_NNN.npy`` per tile (every draw of that tile, flat float32, in draw order; each
  draw is a C-ordered (T, k + 10) block) and a copy of ``ref_pass1.py``'s inputs, so one
  browser run does pass 1 and then pass 2 on its own tile store;
* ``ref/``: per tile ``rank.npy`` (kept components) and ``k.npy`` (the last rSVD rank),
  and the flat concatenations, tile after tile, of ``W`` ((P, r) C-order), ``V``
  ((T, r) C-order), ``s`` (the kept singular values), ``mean``, ``sigma`` and ``energy``.

    OMP_NUM_THREADS=1 python experiments/streaming/ref_pass2.py --t0 0 --t1 3000
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from unittest import mock

import numpy as np
import stream as st

from minime import compress
from minime.params import Params
from minime.tiles import TileStore, tile_grid


class RecordingRng:
    """``np.random.Generator`` stand-in that keeps every ``standard_normal`` draw."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.draws: list[np.ndarray] = []

    def standard_normal(self, size: tuple[int, int], dtype: type = np.float64) -> np.ndarray:
        x = self.rng.standard_normal(size, dtype=dtype)
        self.draws.append(x)
        return x


def pass2_meta(params: Params, H: int, W: int, T: int) -> dict:
    return {
        "H": H,
        "W": W,
        "T": T,
        "tile": params.tile,
        "svd_k0": params.svd_k0,
        "svd_power_iter": params.svd_power_iter,
        "rank_rho_min": params.rank_rho_min,
        "rank_z": params.rank_z,
        "noise_freq": params.noise_freq,
        "noise_frames": params.noise_frames,
    }


def compress_recorded(raw: np.ndarray, seed: int, params: Params) -> tuple:
    """``pass2_worker``'s call for one tile, with the draws recorded."""
    rec = RecordingRng(seed)
    with mock.patch.object(compress.np.random, "default_rng", lambda s: rec):
        tf = compress.compress_tile(
            raw,
            k0=params.svd_k0,
            n_iter=params.svd_power_iter,
            rho_min=params.rank_rho_min,
            z=params.rank_z,
            seed=seed,
            noise_freq=params.noise_freq,
            noise_frames=params.noise_frames,
        )
    k = rec.draws[-1].shape[1] - compress.N_OVERSAMPLES
    return tf, rec.draws, k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=3000)
    a = ap.parse_args()

    s = st.Session(a.session)
    params = s.params()
    p1 = s.out(f"ref_pass1_{a.t0}_{a.t1}")
    meta1 = json.loads((p1 / "inputs" / "params.json").read_text())
    H, W = meta1["H"], meta1["W"]
    grid = tile_grid(H, W, params.tile)
    store = TileStore(p1 / "ref" / "tiles", grid, 1)
    out = s.out(f"ref_pass2_{a.t0}_{a.t1}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)

    cols: dict[str, list[np.ndarray]] = {n: [] for n in ("W", "V", "s", "mean", "sigma", "energy")}
    rank, ks, secs = [], [], []
    T = 0
    for t in grid:
        raw = store.read_tile(t.index)
        T = raw.shape[0]
        s0 = time.perf_counter()
        tf, draws, k = compress_recorded(raw, t.index, params)
        secs.append(time.perf_counter() - s0)
        np.save(inp / f"omega_{t.index:03d}.npy", np.concatenate([d.ravel() for d in draws]))
        Wn = np.linalg.norm(tf.W, axis=0)
        cols["W"].append(tf.W.ravel())
        cols["V"].append(tf.V.ravel())
        cols["s"].append(Wn.astype(np.float32))
        cols["mean"].append(tf.mean)
        cols["sigma"].append(tf.sigma)
        cols["energy"].append(tf.energy)
        rank.append(tf.rank)
        ks.append(k)
    (inp / "pass2.json").write_text(json.dumps(pass2_meta(params, H, W, T), indent=1))
    for f in (p1 / "inputs").iterdir():
        shutil.copy2(f, inp / f.name)
    for name, parts in cols.items():
        np.save(ref / f"{name}.npy", np.concatenate(parts))
    np.save(ref / "rank.npy", np.asarray(rank, np.int64))
    np.save(ref / "k.npy", np.asarray(ks, np.int64))
    summary = {
        "frames": T,
        "tiles": len(grid),
        "rank_total": int(sum(rank)),
        "rank_max": int(max(rank)),
        "k_max": int(max(ks)),
        "seconds": float(sum(secs)),
        "seconds_max_tile": float(max(secs)),
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
