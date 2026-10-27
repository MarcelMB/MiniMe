"""Per-tile low-rank factors and the ``Factors`` container every later stage reads.

A tile with ``P_b`` pixels and ``T`` frames is stored as ``Y_b ~ W_b V_b^T + mean_b 1^T``
with ``W_b (P_b, r)``, ``V_b (T, r)``. ``Y`` is never rebuilt in full; the helpers below
(``AtY``, ``YCt``, ``reconstruct_chunk``, ``traces``) are the only access paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps

from minime.tiles import Tile, tile_grid

FORMAT_VERSION = 1


@dataclass
class TileFactors:
    W: np.ndarray  # (P, r) float32, left factors scaled by singular values
    V: np.ndarray  # (T, r) float32, unit-norm temporal components
    mean: np.ndarray  # (P,) float32
    sigma: np.ndarray  # (P,) float32, per-pixel noise std
    energy: np.ndarray  # (P,) float64, sum_t (y - mean)^2

    @property
    def rank(self) -> int:
        return self.W.shape[1]

    @property
    def P(self) -> int:
        return self.W.shape[0]

    @property
    def T(self) -> int:
        return self.V.shape[0]


@dataclass
class Factors:
    H: int
    W_: int  # frame width (the factor matrices are named W)
    T: int
    grid: list[Tile]
    tiles: list[TileFactors]
    max_proj: np.ndarray
    block_max: np.ndarray
    block_edges: np.ndarray
    meta: dict[str, Any]
    tile_of: np.ndarray = field(init=False, repr=False)  # (P,) int32
    loc_of: np.ndarray = field(init=False, repr=False)  # (P,) int32

    def __post_init__(self) -> None:
        self.tile_of = np.empty(self.P, dtype=np.int32)
        self.loc_of = np.empty(self.P, dtype=np.int32)
        for b in range(len(self.grid)):
            rows = self.tile_rows(b)
            self.tile_of[rows] = b
            self.loc_of[rows] = np.arange(rows.size, dtype=np.int32)

    @property
    def P(self) -> int:
        return self.H * self.W_

    def pixel_index(self, h: int, w: int) -> int:
        return int(h) * self.W_ + int(w)

    def tile_rows(self, b: int) -> np.ndarray:
        """Flat (row-major) pixel indices of tile ``b`` in the tile's own row-major order."""
        t = self.grid[b]
        ys = np.arange(t.y0, t.y1)[:, None]
        xs = np.arange(t.x0, t.x1)[None, :]
        return (ys * self.W_ + xs).ravel().astype(np.int64)

    # ---- algebra on the factors ---------------------------------------------------

    def reconstruct_chunk(self, b: int, t0: int, t1: int) -> np.ndarray:
        """``W V[t0:t1]^T + mean[:, None]`` of tile ``b``: (P_b, t1 - t0) float32."""
        tf = self.tiles[b]
        return (tf.W @ tf.V[t0:t1].T + tf.mean[:, None]).astype(np.float32, copy=False)

    def traces(self, pixels: np.ndarray) -> np.ndarray:
        """Denoised traces including the mean of flat ``pixels``: (n, T) float32."""
        pixels = np.asarray(pixels, dtype=np.int64)
        out = np.empty((pixels.size, self.T), dtype=np.float32)
        tile_idx = self.tile_of[pixels]
        for b in np.unique(tile_idx):
            sel = np.flatnonzero(tile_idx == b)
            loc = self.loc_of[pixels[sel]]
            tf = self.tiles[int(b)]
            out[sel] = tf.W[loc] @ tf.V.T + tf.mean[loc, None]
        return out

    def AtY(self, A: sps.csr_matrix) -> np.ndarray:
        """``A Y`` for a sparse ``A`` (K, P): (K, T) float64 accumulated tile by tile."""
        Acsc = sps.csc_matrix(A)
        out = np.zeros((Acsc.shape[0], self.T), dtype=np.float64)
        for b, tf in enumerate(self.tiles):
            Ab = Acsc[:, self.tile_rows(b)]
            if Ab.nnz == 0:
                continue
            out += (Ab @ tf.W) @ tf.V.T
            out += (Ab @ tf.mean.astype(np.float64))[:, None]
        return out

    def YCt(self, Cc: np.ndarray) -> np.ndarray:
        """``Y Cc^T`` as (P, K) float32; ``Cc`` rows must sum to zero (mean term dropped)."""
        Ct = np.ascontiguousarray(np.asarray(Cc, dtype=np.float32).T)  # (T, K)
        out = np.empty((self.P, Ct.shape[1]), dtype=np.float32)
        for b, tf in enumerate(self.tiles):
            out[self.tile_rows(b)] = tf.W @ (tf.V.T @ Ct)
        return out

    # ---- per-pixel statistics ---------------------------------------------------

    def _flat(self, name: str, dtype: type) -> np.ndarray:
        out = np.empty(self.P, dtype=dtype)
        for b, tf in enumerate(self.tiles):
            out[self.tile_rows(b)] = getattr(tf, name)
        return out

    def sigma_flat(self) -> np.ndarray:
        return self._flat("sigma", np.float32)

    def mean_flat(self) -> np.ndarray:
        return self._flat("mean", np.float32)

    def energy_flat(self) -> np.ndarray:
        return self._flat("energy", np.float64)

    # ---- disk -------------------------------------------------------------------

    def save(self, dir: Path) -> None:
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        for b, tf in enumerate(self.tiles):
            np.savez(
                dir / f"tile_{b:03d}.npz",
                W=tf.W,
                V=tf.V,
                mean=tf.mean,
                sigma=tf.sigma,
                energy=tf.energy,
            )
        np.save(dir / "max_proj.npy", self.max_proj)
        np.save(dir / "block_max.npy", self.block_max)
        np.save(dir / "block_edges.npy", self.block_edges)
        # largest tile side; tile_grid(H, W, tile) rebuilds the grid (also when H, W < tile)
        tile = max(max(t.y1 - t.y0, t.x1 - t.x0) for t in self.grid)
        meta = {
            "format_version": FORMAT_VERSION,
            "H": self.H,
            "W": self.W_,
            "T": self.T,
            "tile": tile,
            "n_tiles": len(self.tiles),
            "ranks": [tf.rank for tf in self.tiles],
            "meta": self.meta,
        }
        (dir / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, dir: Path) -> Factors:
        dir = Path(dir)
        m = json.loads((dir / "meta.json").read_text())
        if m["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported factors format {m['format_version']}")
        grid = tile_grid(m["H"], m["W"], m["tile"])
        if len(grid) != m["n_tiles"]:
            raise ValueError(f"grid has {len(grid)} tiles, meta.json says {m['n_tiles']}")
        tiles = []
        for b in range(m["n_tiles"]):
            with np.load(dir / f"tile_{b:03d}.npz") as z:
                tiles.append(TileFactors(z["W"], z["V"], z["mean"], z["sigma"], z["energy"]))
        return cls(
            m["H"],
            m["W"],
            m["T"],
            grid,
            tiles,
            np.load(dir / "max_proj.npy"),
            np.load(dir / "block_max.npy"),
            np.load(dir / "block_edges.npy"),
            m["meta"],
        )
