"""Pass 2: per-tile streamed rSVD in parallel workers, and assembly of ``Factors``."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import numpy as np

from cafast.compress import compress_tile
from cafast.factors import Factors, TileFactors
from cafast.io import RawInfo
from cafast.params import Params
from cafast.pass1 import Pass1Result
from cafast.tiles import Tile, TileStore

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _split(n: int, parts: int) -> list[range]:
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [range(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def pass2_worker(
    store_root: str, grid: list[Tile], n_parts: int, indices: range, params: Params
) -> list[TileFactors]:
    """Compress the tiles ``indices`` (seed = tile index)."""
    store = TileStore(Path(store_root), grid, n_parts)
    return [
        compress_tile(
            store.read_tile(i),
            k0=params.svd_k0,
            n_iter=params.svd_power_iter,
            rho_min=params.rank_rho_min,
            z=params.rank_z,
            seed=i,
        )
        for i in indices
    ]


def run_pass2(
    store: TileStore, grid: list[Tile], params: Params, workers: int
) -> list[TileFactors]:
    """Compress every tile with ``workers`` single-threaded processes; grid order."""
    ranges = _split(len(grid), workers)
    jobs = [(str(store.root), grid, store.n_parts, r, params) for r in ranges]
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(len(jobs)) as pool:
            res = pool.starmap(pass2_worker, jobs, chunksize=1)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return [tf for chunk in res for tf in chunk]


def build_factors(
    info: RawInfo, grid: list[Tile], p1: Pass1Result, tiles: list[TileFactors], params: Params
) -> Factors:
    return Factors(
        H=info.height,
        W_=info.width,
        T=info.frames,
        grid=grid,
        tiles=tiles,
        max_proj=p1.max_proj,
        block_max=p1.block_max,
        block_edges=p1.block_edges,
        meta={"params": params.asdict(), "pass1_seconds": p1.seconds},
    )
