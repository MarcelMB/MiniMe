"""Tile grid and the on-disk uint8 tile store (one file per tile and frame part)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class Tile:
    index: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def pixels(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def tile_grid(h: int, w: int, tile: int = 64) -> list[Tile]:
    out = []
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            out.append(Tile(len(out), y0, min(h, y0 + tile), x0, min(w, x0 + tile)))
    return out


def pack_tiles(frames: np.ndarray, grid: list[Tile]) -> np.ndarray:
    """(n, h, w) -> one flat uint8 buffer, tile after tile, each tile (n, th, tw) C-order."""
    return np.concatenate([frames[:, t.y0 : t.y1, t.x0 : t.x1].ravel() for t in grid])


def tile_offsets(n: int, grid: list[Tile]) -> list[tuple[int, int]]:
    """(start, stop) byte ranges of each tile inside ``pack_tiles`` output for n frames."""
    out, o = [], 0
    for t in grid:
        out.append((o, o + n * t.pixels))
        o += n * t.pixels
    return out


class TileStore:
    """Directory of ``tile_NNN.partPP`` files: tile-major uint8 copy of the movie."""

    def __init__(self, root: Path, grid: list[Tile], n_parts: int) -> None:
        self.root = Path(root)
        self.grid = grid
        self.n_parts = n_parts

    def part_path(self, tile: int, part: int) -> Path:
        return self.root / f"tile_{tile:03d}.part{part:02d}"

    def open_part(self, part: int) -> list[BinaryIO]:
        """Open one write-binary file per tile (grid order) for ``part``; caller closes."""
        self.root.mkdir(parents=True, exist_ok=True)
        return [open(self.part_path(t.index, part), "wb") for t in self.grid]

    def read_tile(self, tile: int) -> np.ndarray:
        """All frames of one tile, (T, th, tw) uint8, parts concatenated in order."""
        t = self.grid[tile]
        paths = [self.part_path(tile, p) for p in range(self.n_parts)]
        raw = np.empty(sum(p.stat().st_size for p in paths), dtype=np.uint8)
        o = 0
        buf = memoryview(raw)
        for path in paths:  # read straight into one buffer: no per-part arrays, no concatenate
            with open(path, "rb", buffering=0) as fh:
                while n := fh.readinto(buf[o:]):  # a read() returns at most ~2 GB on macOS
                    o += n
        return raw[:o].reshape(-1, *t.shape)

    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("tile_*.part*"))

    def remove(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
