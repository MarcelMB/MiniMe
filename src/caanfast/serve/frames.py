"""Frame ``t`` of a run as raw, preprocessed (tile store) or reconstruction, scaled to uint8."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from caanfast import io
from caanfast.factors import Factors
from caanfast.serve.runs import RunView

SOURCES = ("raw", "pre", "recon")


def raw_frame(info: io.RawInfo, t: int) -> np.ndarray:
    """Frame ``t`` of the raw session, decoded by seeking inside its segment."""
    edges = np.cumsum([0, *info.segment_frames])
    if not 0 <= t < edges[-1]:
        raise IndexError(f"frame {t} is beyond the {edges[-1]} raw frames")
    s = int(np.searchsorted(edges, t, side="right") - 1)
    k = t - int(edges[s])
    cap = io._open(info.segments[s], info.bgr)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = cap.read()
    finally:
        cap.release()
    if not ok:
        raise OSError(f"cannot read frame {k} of {info.segments[s]}")
    return io._gray(f, info.bgr).astype(np.float32)


def store_frame(store: Path, fac: Factors, t: int) -> np.ndarray | None:
    """Frame ``t`` of the preprocessed uint8 tile store, or None if the store is gone."""
    parts = sorted(Path(store).glob("tile_000.part*"))
    if not parts:
        return None
    n_per = [p.stat().st_size // fac.grid[0].pixels for p in parts]
    edges = np.cumsum([0, *n_per])
    if not 0 <= t < edges[-1]:
        raise IndexError(f"frame {t} is beyond the {edges[-1]} stored frames")
    p = int(np.searchsorted(edges, t, side="right") - 1)
    k = t - int(edges[p])
    out = np.zeros((fac.H, fac.W_), np.float32)
    for tl in fac.grid:
        mm = np.memmap(Path(store) / f"tile_{tl.index:03d}.part{p:02d}", np.uint8, "r")
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = mm.reshape(-1, *tl.shape)[k]
    return out


def recon_frame(fac: Factors, t: int) -> np.ndarray:
    """Frame ``t`` rebuilt from the per-tile low-rank factors."""
    out = np.zeros((fac.H, fac.W_), np.float32)
    for b, tl in enumerate(fac.grid):
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = fac.reconstruct_chunk(b, t, t + 1)[:, 0].reshape(
            tl.shape
        )
    return out


def tile_map(fac: Factors, values: list[float]) -> np.ndarray:
    """(H, W) image with each tile filled with its value (e.g. its rank)."""
    out = np.zeros((fac.H, fac.W_), np.float32)
    for tl, v in zip(fac.grid, values, strict=True):
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = v
    return out


class FrameSource:
    """uint8 frames with a fixed per-run contrast and a small LRU of recent frames."""

    def __init__(self, run: RunView, capacity: int = 256) -> None:
        self.run = run
        self.capacity = capacity
        self._lru: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._contrast_lock = threading.Lock()

    def available(self, source: str) -> bool:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r} (one of {', '.join(SOURCES)})")
        return {"raw": self.run.has_raw, "pre": self.run.has_store, "recon": True}[source]

    def _float(self, source: str, t: int) -> np.ndarray:
        if source == "raw":
            return raw_frame(self.run.raw_info, t)
        if source == "pre":
            f = store_frame(self.run.path / "tiles", self.run.factors, t)
            if f is None:
                raise LookupError("the tile store of this run was not kept")
            return f
        return recon_frame(self.run.factors, t)

    def _check(self, source: str, t: int) -> None:
        if not self.available(source):
            why = {
                "raw": f"raw videos not found in {self.run.raw_dir}",
                "pre": "the tile store was not kept (run with --keep-store)",
            }[source]
            raise LookupError(why)
        if not 0 <= t < self.run.T:
            raise IndexError(f"frame {t} out of range 0..{self.run.T - 1}")

    def has_contrast(self, source: str) -> bool:
        return source in self.run.cache.load_json("contrast.json", {})

    def contrast(self, source: str) -> tuple[float, float]:
        """1st / 99.9th percentile of <= 50 frames spread over the run (cached)."""
        doc = self.run.cache.load_json("contrast.json", {})
        if source not in doc:
            with self._contrast_lock:
                doc = self.run.cache.load_json("contrast.json", {})
                if source not in doc:
                    self._check(source, 0)
                    ts = np.linspace(0, self.run.T - 1, min(50, self.run.T)).astype(int)
                    vals = np.stack([self._float(source, int(t)) for t in ts])
                    lo, hi = (float(x) for x in np.percentile(vals, [1, 99.9]))
                    doc[source] = [lo, hi if hi > lo else lo + 1.0]
                    self.run.cache.save_json("contrast.json", doc)
        lo, hi = doc[source]
        return float(lo), float(hi)

    def frame(self, source: str, t: int) -> np.ndarray:
        self._check(source, t)
        key = (source, t)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
        lo, hi = self.contrast(source)
        f = self._float(source, t)
        u8 = np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        with self._lock:
            self._lru[key] = u8
            while len(self._lru) > self.capacity:
                self._lru.popitem(last=False)
        return u8
