"""Pass 2: per-tile streamed rSVD (in time windows on long sessions) in parallel workers, and
assembly of ``Factors``.

With an output directory each worker saves a tile's factors as soon as it is compressed and
keeps nothing: held until the end, the ``V`` of finished tiles (T x rank float32, 23 GB on a
24 h session) piled up in the workers and swapped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minime.compress import compress_tile_windows
from minime.factors import Factors, TileFactors, load_tile, save_tile
from minime.io import RawInfo
from minime.params import Params
from minime.pass1 import Pass1Result, _split, run_workers
from minime.tiles import Tile, TileStore


def pass2_worker(
    store_root: str,
    grid: list[Tile],
    n_parts: int,
    indices: range,
    params: Params,
    out_dir: str | None = None,
    valid: np.ndarray | None = None,
) -> list[TileFactors]:
    """Compress the tiles ``indices`` (seed = tile index), in time windows of at most
    ``params.svd_window`` frames. With ``out_dir`` each tile is saved there (``save_tile``)
    and the list is empty. ``valid`` (H, W bool): pixels kept, the others are set to 0."""
    store = TileStore(Path(store_root), grid, n_parts)
    out: list[TileFactors] = []
    for i in indices:
        tf = _compress(store, i, params, _tile_mask(grid[i], valid))
        if out_dir is None:
            out.append(tf)
        else:
            save_tile(Path(out_dir), i, tf)
    return out


def _tile_mask(t: Tile, valid: np.ndarray | None) -> np.ndarray | None:
    """Pixels of tile ``t`` outside ``valid`` (None when there are none)."""
    if valid is None:
        return None
    drop = ~valid[t.y0 : t.y1, t.x0 : t.x1]
    return drop if drop.any() else None


def _reader(store: TileStore, i: int, drop: np.ndarray | None):
    """``read(t0, t1)`` of tile ``i``'s frames with the ``drop`` pixels set to 0."""

    def read(t0: int, t1: int) -> np.ndarray:
        frames = store.read_frames(i, t0, t1)
        if drop is not None:
            frames = np.array(frames)
            frames[:, drop] = 0
        return frames

    return read


def _compress(
    store: TileStore, i: int, params: Params, drop: np.ndarray | None = None
) -> TileFactors:
    """Factors of tile ``i`` (seed = tile index)."""
    return compress_tile_windows(
        _reader(store, i, drop),
        store.tile_frames(i),
        params.svd_window,
        k0=params.svd_k0,
        n_iter=params.svd_power_iter,
        rho_min=params.rank_rho_min,
        z=params.rank_z,
        seed=i,
        noise_freq=params.noise_freq,
        noise_frames=params.noise_frames,
    )


def run_pass2(
    store: TileStore,
    grid: list[Tile],
    params: Params,
    workers: int,
    out_dir: Path | None = None,
    valid: np.ndarray | None = None,
) -> list[TileFactors]:
    """Compress every tile with ``workers`` single-threaded processes; grid order.

    With ``out_dir`` the workers save the tiles there and the result reads each ``V``
    memory-mapped from its file. ``valid`` (H, W bool): pixels kept, the others are zeroed
    before compression (``pipeline.border_mask``)."""
    ranges = _split(len(grid), workers)
    od = None if out_dir is None else str(out_dir)
    jobs = [(str(store.root), grid, store.n_parts, r, params, od, valid) for r in ranges]
    res = run_workers(pass2_worker, jobs)
    if out_dir is None:
        return [tf for chunk in res for tf in chunk]
    return [load_tile(out_dir, i, mmap=True) for i in range(len(grid))]


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
