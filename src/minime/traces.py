"""(K, T) trace arrays of window 2 that move to disk when they are large.

initC, unit merge and CNMF hold a few units-by-frames float32 arrays at once (1569 units on
a 24 h session: 10.8 GB each). Every stage reads them a block of frames at a time, so inside
``spill_to(dir)`` an array above ``RAM_FRACTION`` of the RAM is a memory map of an unnamed
temporary file in ``dir`` instead: macOS / Linux / Windows write its pages back to that file
under memory pressure, not to swap, and delete the file when the array is freed.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from minime.factors import TIME_CHUNK

RAM_FRACTION = 1 / 16  # 2 GB on a 32 GB Mac: a 4 h session stays in RAM, 8 h initC spills

_spill_dir: Path | None = None


def total_ram() -> int:
    """Physical memory in bytes."""
    if sys.platform == "win32":

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong)
                for n in (
                    "ullTotalPhys",
                    "ullAvailPhys",
                    "ullTotalPageFile",
                    "ullAvailPageFile",
                    "ullTotalVirtual",
                    "ullAvailVirtual",
                    "ullAvailExtendedVirtual",
                )
            ]

        s = _Status(dwLength=ctypes.sizeof(_Status))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.ullTotalPhys)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


@contextmanager
def spill_to(directory: Path | None) -> Iterator[None]:
    """Arrays from ``empty`` inside this block spill to ``directory`` (None: never)."""
    global _spill_dir
    prev, _spill_dir = _spill_dir, None if directory is None else Path(directory)
    try:
        yield
    finally:
        _spill_dir = prev


def empty(shape: tuple[int, int], dtype: np.dtype | type = np.float32) -> np.ndarray:
    """``np.empty(shape, dtype)``, or a memory map of a temporary file when spilling."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if _spill_dir is None or nbytes <= RAM_FRACTION * total_ram():
        return np.empty(shape, dtype=dtype)
    _spill_dir.mkdir(parents=True, exist_ok=True)
    # the map keeps its own handle: the file is gone (POSIX) or deleted on the last close
    # (Windows) once both the map and this handle are closed
    with tempfile.TemporaryFile(dir=_spill_dir, prefix="minime_traces_") as fh:
        return np.memmap(fh, dtype=dtype, mode="w+", shape=shape)


def scaled_rows(
    X: np.ndarray, rows: np.ndarray, scale: np.ndarray, chunk: int = TIME_CHUNK
) -> np.ndarray:
    """``X[rows] * scale[:, None]`` in float32, ``chunk`` frames at a time."""
    out = empty((rows.size, X.shape[1]))
    s = np.asarray(scale, dtype=np.float32)[:, None]
    for t0 in range(0, X.shape[1], chunk):
        out[:, t0 : t0 + chunk] = X[rows, t0 : t0 + chunk] * s
    return out


def all_finite(X: np.ndarray, chunk: int = TIME_CHUNK) -> bool:
    """``np.isfinite(X).all()`` without a mask of all of a (K, T) ``X``."""
    X = np.asarray(X)
    if X.ndim != 2:
        return bool(np.isfinite(X).all())
    return all(np.isfinite(X[:, t0 : t0 + chunk]).all() for t0 in range(0, X.shape[1], chunk))
