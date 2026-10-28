"""Raw AVI access: segment listing, frame reading, folder probe and content hash."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def list_segments(raw_dir: Path) -> list[Path]:
    """``N.avi`` files of ``raw_dir`` sorted by integer stem.

    Other ``.avi`` names are skipped: macOS ``._0.avi`` files, a ``behavior.avi``.
    """
    segs = [p for p in Path(raw_dir).glob("*.avi") if p.stem.isdigit()]
    return sorted(segs, key=lambda p: int(p.stem))


# a codec whose raw decode differs from its BGR->gray decode by more than this is read via BGR
CODEC_TOL_MEAN = 0.5
CODEC_TOL_MAX = 4


def _open(path: Path, bgr: bool = False) -> cv2.VideoCapture:
    """Capture of ``path``: raw decoder output (``CONVERT_RGB=0``), or BGR when ``bgr``."""
    cap = cv2.VideoCapture(str(path))
    if not bgr:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return cap


def _gray(frame: np.ndarray, bgr: bool = False) -> np.ndarray:
    """2-D uint8 frame: channel 0 of a raw decode, or the BGR->gray conversion when ``bgr``."""
    if frame.ndim == 2:
        return frame
    if bgr:
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(frame, code)
    return frame[..., 0]


def needs_bgr(raw: np.ndarray, via_bgr: np.ndarray) -> bool:
    """True if the raw decode (``CONVERT_RGB=0``, channel 0) of a frame cannot be trusted.

    ``raw`` and ``via_bgr`` are the same frame decoded both ways (2-D gray). GREY, FFV1 and
    MJPG agree exactly. HuffYUV (bgr0) returns an 8UC1 frame of the right shape holding
    the wrong pixels (mean |diff| 53-88); XVID and gray-HuffYUV return the limited-range Y
    plane (mean |diff| ~9 against the full-range BGR->gray). The BGR path is then used.
    """
    if raw.shape != via_bgr.shape:
        return True
    d = np.abs(raw.astype(np.int16) - via_bgr.astype(np.int16))
    return bool(d.mean() > CODEC_TOL_MEAN or d.max() > CODEC_TOL_MAX)


def first_frame(path: Path, bgr: bool) -> np.ndarray:
    """Frame 0 of ``path`` as 2-D uint8, decoded raw or via BGR->gray."""
    cap = _open(path, bgr)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"cannot read a frame from {path}")
    return _gray(frame, bgr)


def read_avi(path: Path, max_frames: int | None = None, bgr: bool = False) -> np.ndarray:
    """The first ``max_frames`` (default all) frames of one AVI as (n, h, w) uint8.

    ``bgr``: decode through BGR->gray (``RawInfo.bgr``, decided by ``probe``).

    Frames go straight into one preallocated array sized by the container's frame count
    (frames beyond it are appended): keeping the decoded 3-channel frames alive until a
    final stack cost 4x the segment size (8.4 GB for a 1.6 GB single-file session).
    """
    limit = frame_count(path) if max_frames is None else min(frame_count(path), max_frames)
    cap = _open(path, bgr)
    out: np.ndarray | None = None
    extra: list[np.ndarray] = []
    n = 0
    while max_frames is None or n < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        g = _gray(f, bgr)
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


def iter_avi(path: Path, batch: int = 1000, bgr: bool = False) -> Iterator[np.ndarray]:
    """Yield the frames of one AVI as (n, h, w) uint8 batches of ``batch`` frames (last shorter).

    Memory stays at one batch whatever the file size (a single-file session no longer has to
    fit in RAM). The total of the yielded frames is the decoded frame count.
    """
    cap = _open(path, bgr)
    try:
        buf: np.ndarray | None = None
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            g = _gray(f, bgr)
            if buf is None:
                buf = np.empty((batch, *g.shape), dtype=np.uint8)
            buf[n] = g
            n += 1
            if n == batch:
                yield buf
                buf, n = None, 0
        if buf is not None and n:
            yield buf[:n]
    finally:
        cap.release()


def frame_count(path: Path) -> int:
    """Frame count from the container header (``CAP_PROP_FRAME_COUNT``), not decoded."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def decoded_frame_count(path: Path, bgr: bool = False) -> int:
    """Number of frames that actually decode: a truncated file has fewer than its header says."""
    cap = _open(path, bgr)
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
    bgr: bool = False  # decode via BGR->gray: the raw decode of this codec is wrong


def probe(raw_dir: Path) -> RawInfo:
    """Frame size, per-segment frame counts and fps (from ``raw_dir.parent/meta.json``).

    Counts come from the container header, except for the last segment, which is decoded
    in full: a recording that stops abruptly leaves a truncated last file whose header
    still claims the full count. Pass 1 checks every other segment's decoded count.

    Frame 0 is decoded raw and via BGR->gray; if they disagree (``needs_bgr``) the whole
    session is read via BGR->gray (``RawInfo.bgr``). So is a codec whose raw decode returns
    no frame at all (palette rawvideo, as written by Fiji/ImageJ).
    """
    segments = list_segments(raw_dir)
    if not segments:
        raise FileNotFoundError(f"no *.avi segments in {raw_dir}")
    via_bgr = first_frame(segments[0], True)
    try:
        bgr = needs_bgr(first_frame(segments[0], False), via_bgr)
    except OSError:
        bgr = True
    height, width = (via_bgr if bgr else first_frame(segments[0], False)).shape
    counts = [frame_count(s) for s in segments[:-1]]
    counts.append(decoded_frame_count(segments[-1], bgr))
    meta = Path(os.path.abspath(raw_dir)).parent / "meta.json"
    fps = float(json.loads(meta.read_text())["fps"]) if meta.exists() else None
    return RawInfo(height, width, sum(counts), counts, segments, fps, bgr)


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
