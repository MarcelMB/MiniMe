"""Per-tile low-rank factors and the ``Factors`` container every later stage reads.

A tile with ``P_b`` pixels and ``T`` frames is stored as ``Y_b ~ W_b V_b^T + mean_b 1^T``
with ``W_b (P_b, r)``, ``V_b (T, r)``. ``Y`` is never rebuilt in full; the helpers below
(``AtY``, ``YCt``, ``reconstruct_chunk``, ``traces``) are the only access paths.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps

from minime.tiles import Tile, tile_grid

# 2: each tile's V in its own .npy (memory-mappable); 1: V inside the tile's .npz
FORMAT_VERSION = 2


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


# frames per block when a product over the whole session (``A Y``, ``Y C^T``) is formed
# piecewise: a (K, T) float64 array of 1569 units on a 24 h session is 22 GB
TIME_CHUNK = 65_536


def _tile_vtc(tf: TileFactors, t0: int, t1: int, Ct: np.ndarray) -> np.ndarray:
    """``V[t0:t1]^T Ct`` of one tile, (r, K) float32."""
    return tf.V[t0:t1].T @ Ct


def _tile_vtc_f64(tf: TileFactors, t0: int, t1: int, Ct: np.ndarray) -> np.ndarray:
    """``_tile_vtc`` in float64: the fallback when the float32 product is not finite. On
    macOS 26 (M1 Max) Accelerate's float32 products with ~513-575 output columns
    intermittently returned NaN; a 4 h session failed the spatial update with K = 527 units
    and passed on rerunning the same input.
    """
    return tf.V[t0:t1].astype(np.float64).T @ Ct.astype(np.float64)


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
    # a time subset (``subset``): its first frame in the session and the session's frames,
    # so the seed windows stay the session's own
    t0: int = 0
    session_T: int | None = None
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

    def subset(self, t0: int, t1: int) -> Factors:
        """Frames ``[t0, t1)`` as factors of their own; ``t0`` and ``t1`` must be block edges.

        Each tile's ``V`` slice is re-orthonormalised (``V_s = Q R``, ``W -> W R^T``, same
        product), so ``energy`` = low-rank part + the residual's share of the frames holds as
        on full factors. ``max_proj`` is the max over the slice's blocks; ``mean`` and
        ``sigma`` are kept.
        """
        e = self.block_edges
        i0, i1 = np.searchsorted(e, [t0, t1])
        if not 0 <= t0 < t1 <= self.T or e[i0] != t0 or i1 >= len(e) or e[i1] != t1:
            raise ValueError(f"frames [{t0}, {t1}) do not start and end on stored block edges")
        tiles = []
        for tf in self.tiles:
            q, r = np.linalg.qr(tf.V[t0:t1].astype(np.float64))
            W = tf.W.astype(np.float64)
            resid = np.maximum(tf.energy - (W**2).sum(axis=1), 0.0)
            Ws = W @ r.T
            energy = (Ws**2).sum(axis=1) + resid * (t1 - t0) / self.T
            tiles.append(
                TileFactors(Ws.astype(np.float32), q.astype(np.float32), tf.mean, tf.sigma, energy)
            )
        block_max = self.block_max[i0:i1]
        return Factors(
            self.H,
            self.W_,
            t1 - t0,
            self.grid,
            tiles,
            block_max.max(axis=0),
            block_max,
            e[i0 : i1 + 1] - t0,
            self.meta,
            self.t0 + t0,
            self.session_T or self.T,
        )

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

    def _aty_parts(self, A: sps.csr_matrix) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """``(tile, A_b W_b, A_b mean_b)`` of every tile that ``A`` touches."""
        Acsc = sps.csc_matrix(A)
        parts = []
        for b, tf in enumerate(self.tiles):
            Ab = Acsc[:, self.tile_rows(b)]
            if Ab.nnz:
                parts.append((b, Ab @ tf.W, Ab @ tf.mean.astype(np.float64)))
        return parts

    def _aty(self, parts: list, K: int, t0: int, t1: int) -> np.ndarray:
        out = np.zeros((K, t1 - t0), dtype=np.float64)
        for b, AW, Am in parts:
            out += AW @ self.tiles[b].V[t0:t1].T
            out += Am[:, None]
        return out

    def AtY(self, A: sps.csr_matrix, t0: int = 0, t1: int | None = None) -> np.ndarray:
        """``A Y[:, t0:t1]`` for a sparse ``A`` (K, P): (K, t1 - t0) float64, tile by tile."""
        return self._aty(self._aty_parts(A), A.shape[0], t0, self.T if t1 is None else t1)

    def AtY_blocks(
        self, A: sps.csr_matrix, chunk: int = TIME_CHUNK
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """``(t0, t1, A Y[:, t0:t1])`` over the session, ``chunk`` frames at a time: for
        callers that reduce ``A Y`` block by block instead of holding all of it."""
        parts = self._aty_parts(A)
        for t0 in range(0, self.T, chunk):
            t1 = min(self.T, t0 + chunk)
            yield t0, t1, self._aty(parts, A.shape[0], t0, t1)

    def YCt(
        self,
        C: np.ndarray,
        center: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        chunk: int = TIME_CHUNK,
    ) -> np.ndarray:
        """``Y Cc^T`` as (P, K) float32 with ``Cc = (C - center) / scale`` row by row (each
        default: none). ``Cc`` rows must sum to zero (mean term dropped). ``Cc`` is formed
        ``chunk`` frames at a time (float32 products, float64 sums), never all of it.
        """
        K = C.shape[0]
        VtC = [np.zeros((tf.rank, K), dtype=np.float64) for tf in self.tiles]
        for t0 in range(0, self.T, chunk):
            t1 = min(self.T, t0 + chunk)
            Cc = np.asarray(C[:, t0:t1], dtype=np.float64)
            if center is not None:
                Cc = Cc - np.asarray(center, dtype=np.float64)[:, None]
            if scale is not None:
                Cc = Cc / np.asarray(scale, dtype=np.float64)[:, None]
            Ct = np.ascontiguousarray(Cc.T, dtype=np.float32)  # (t1 - t0, K)
            for b, tf in enumerate(self.tiles):
                part = _tile_vtc(tf, t0, t1, Ct)
                if not np.isfinite(part).all():
                    warnings.warn(
                        f"YCt: float32 product of tile {b} (frames {t0}:{t1}) is not finite; "
                        "recomputed in float64",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    part = _tile_vtc_f64(tf, t0, t1, Ct)
                VtC[b] += part
        out = np.empty((self.P, K), dtype=np.float32)
        for b, tf in enumerate(self.tiles):
            out[self.tile_rows(b)] = tf.W.astype(np.float64) @ VtC[b]
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
                dir / f"tile_{b:03d}.npz", W=tf.W, mean=tf.mean, sigma=tf.sigma, energy=tf.energy
            )
            np.save(dir / f"tile_{b:03d}.V.npy", tf.V)
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
        (dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, dir: Path, mmap: bool = False) -> Factors:
        """Factors saved by ``save``; ``mmap``: each tile's ``V`` is a read-only memory map
        of its file instead of an array in RAM (``V`` is T x rank per tile: 23 GB on a 24 h
        session). Format 1 (``V`` inside the ``.npz``) is always read into RAM."""
        dir = Path(dir)
        m = json.loads((dir / "meta.json").read_text(encoding="utf-8"))
        if m["format_version"] not in (1, FORMAT_VERSION):
            raise ValueError(f"unsupported factors format {m['format_version']}")
        grid = tile_grid(m["H"], m["W"], m["tile"])
        if len(grid) != m["n_tiles"]:
            raise ValueError(f"grid has {len(grid)} tiles, meta.json says {m['n_tiles']}")
        tiles = []
        for b in range(m["n_tiles"]):
            with np.load(dir / f"tile_{b:03d}.npz") as z:
                V = (
                    z["V"]
                    if m["format_version"] == 1
                    else np.load(dir / f"tile_{b:03d}.V.npy", mmap_mode="r" if mmap else None)
                )
                tiles.append(TileFactors(z["W"], V, z["mean"], z["sigma"], z["energy"]))
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
