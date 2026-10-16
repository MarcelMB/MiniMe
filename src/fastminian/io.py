"""Raw AVI access: segment listing, frame reading, folder probe and content hash."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def list_segments(raw_dir: Path) -> list[Path]:
    """``*.avi`` files of ``raw_dir`` sorted by integer stem."""
    return sorted(Path(raw_dir).glob("*.avi"), key=lambda p: int(p.stem))


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return cap


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else frame[..., 0]


def read_avi(path: Path) -> np.ndarray:
    """All frames of one AVI as (n, h, w) uint8."""
    cap = _open(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(_gray(f))
    cap.release()
    return np.stack(frames)


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


@dataclass(frozen=True)
class RawInfo:
    height: int
    width: int
    frames: int
    segment_frames: list[int]
    segments: list[Path]
    fps: float | None


def probe(raw_dir: Path) -> RawInfo:
    """Frame size, per-segment frame counts and fps (from ``raw_dir.parent/meta.json``)."""
    segments = list_segments(raw_dir)
    if not segments:
        raise FileNotFoundError(f"no *.avi segments in {raw_dir}")
    cap = _open(segments[0])
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"cannot read a frame from {segments[0]}")
    height, width = _gray(first).shape
    counts = [frame_count(s) for s in segments]
    meta = Path(raw_dir).parent / "meta.json"
    fps = float(json.loads(meta.read_text())["fps"]) if meta.exists() else None
    return RawInfo(height, width, sum(counts), counts, segments, fps)


def hash_files(paths: list[Path], sample_mb: int = 1, n_samples: int = 16) -> str:
    """Sampled content hash of the raw video files.

    For each file: its name, its size and ``n_samples`` blocks of ``sample_mb`` MiB spread
    evenly over the file (first and last block included); files no larger than all the
    blocks together are hashed whole. About 16 MB of reads per file whatever its size.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f"sampled:{sample_mb}:{n_samples}".encode())
    for p in sorted(str(Path(x)) for x in paths):
        size = os.path.getsize(p)
        h.update(os.path.basename(p).encode())
        h.update(size.to_bytes(8, "little"))
        with open(p, "rb") as f:
            block = sample_mb << 20
            if size <= block * n_samples:
                h.update(f.read())
                continue
            for off in np.linspace(0, size - block, n_samples).astype(np.int64):
                f.seek(int(off))
                h.update(f.read(block))
    return h.hexdigest()
