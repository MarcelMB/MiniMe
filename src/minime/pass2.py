"""Pass 2: per-tile streamed rSVD in parallel workers, and assembly of ``Factors``."""

from __future__ import annotations

from pathlib import Path

from minime.compress import compress_tile
from minime.factors import Factors, TileFactors
from minime.io import RawInfo
from minime.params import Params
from minime.pass1 import Pass1Result, _split, run_workers
from minime.tiles import Tile, TileStore


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
            noise_freq=params.noise_freq,
            noise_frames=params.noise_frames,
        )
        for i in indices
    ]


def run_pass2(
    store: TileStore, grid: list[Tile], params: Params, workers: int
) -> list[TileFactors]:
    """Compress every tile with ``workers`` single-threaded processes; grid order."""
    ranges = _split(len(grid), workers)
    jobs = [(str(store.root), grid, store.n_parts, r, params) for r in ranges]
    res = run_workers(pass2_worker, jobs)
    return [tf for chunk in res for tf in chunk]


def build_factors(
    info: RawInfo, grid: list[Tile], p1: Pass1Result, tiles: list[TileFactors], params: Params
) -> Factors:
    return Factors(
        H=info.height,
        W_=info.width,
        T=p1.frames,
        grid=grid,
        tiles=tiles,
        max_proj=p1.max_proj,
        block_max=p1.block_max,
        block_edges=p1.block_edges,
        meta={"params": params.asdict(), "pass1_seconds": p1.seconds},
    )
