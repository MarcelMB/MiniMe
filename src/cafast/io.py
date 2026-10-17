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


def read_avi(path: Path, max_frames: int | None = None) -> np.ndarray:
    """The first ``max_frames`` (default all) frames of one AVI as (n, h, w) uint8.

    Frames go straight into one preallocated array sized by the container's frame count
    (frames beyond it are appended): keeping the decoded 3-channel frames alive until a
    final stack cost 4x the segment size (8.4 GB for a 1.6 GB single-file session).
    """
    limit = frame_count(path) if max_frames is None else min(frame_count(path), max_frames)
    cap = _open(path)
    out: np.ndarray | None = None
    extra: list[np.ndarray] = []
    n = 0
    while max_frames is None or n < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        g = _gray(f)
        if out is None:
            out = np.empty((max(limit, 1), *g.shape), dtype=np.uint8)
        if n < out.shape[0]:
            out[n] = g
        else:
            extra.append(g.copy())
        n += 1
    cap.release()
    if out is None:
        raise OSError(f"no frames could be read from {path}")
    return np.concatenate([out[: min(n, out.shape[0])], *([np.stack(extra)] if extra else [])])


def frame_count(path: Path) -> int:
    """Frame count from the container header (``CAP_PROP_FRAME_COUNT``), not decoded."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def decoded_frame_count(path: Path) -> int:
    """Number of frames that actually decode: a truncated file has fewer than its header says."""
    cap = _open(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


@dataclass(frozen=True)
class RawInfo:
    height: int
    width: int
    frames: int
    segment_frames: list[int]  # header counts; the last segment's is decoded (see probe)
    segments: list[Path]
    fps: float | None


def probe(raw_dir: Path) -> RawInfo:
    """Frame size, per-segment frame counts and fps (from ``raw_dir.parent/meta.json``).

    Counts come from the container header, except for the last segment, which is decoded
    in full: a recording that stops abruptly leaves a truncated last file whose header
    still claims the full count. Pass 1 checks every other segment's decoded count.
    """
    segments = list_segments(raw_dir)
    if not segments:
        raise FileNotFoundError(f"no *.avi segments in {raw_dir}")
    cap = _open(segments[0])
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"cannot read a frame from {segments[0]}")
    height, width = _gray(first).shape
    counts = [frame_count(s) for s in segments[:-1]] + [decoded_frame_count(segments[-1])]
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
