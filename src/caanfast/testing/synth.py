"""Synthetic miniscope session with ground truth.

Cells are 15 x 15 Gaussian patches with spiking exp-decay traces on a smooth vignette
background with slow drift, Gaussian noise and integer random-walk rigid motion, uint8.
The RNG call order (positions, traces, motion steps, then per-chunk noise) is fixed so a
given ``(h, w, T, seed)`` always reproduces the same movie.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MOTION_MARGIN = 8  # px of canvas around the FOV for synthetic motion
GEN_CHUNK = 50  # frames generated at once (bounds memory)


# ---------------------------------------------------------------------------
# Synthetic movie
# ---------------------------------------------------------------------------


def _gauss_patch(size: int = 15, sigma: float = 3.0) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma**2))
    return (g / g.max()).astype(np.float32)


def _traces(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Spikes (rate ~0.3 Hz at 30 fps) convolved with an exp decay (tau 0.5 s)."""
    spikes = (rng.random((n, T)) < 0.01).astype(np.float32) * rng.uniform(0.5, 1.5, (n, 1)).astype(
        np.float32
    )
    decay = np.float32(np.exp(-1.0 / 15.0))
    C = np.empty_like(spikes)
    acc = np.zeros(n, dtype=np.float32)
    for t in range(T):
        acc = acc * decay + spikes[:, t]
        C[:, t] = acc
    return C


def synth_movie(h: int, w: int, T: int, seed: int = 0) -> np.ndarray:
    """(T, h, w) uint8 movie with cells, background, noise and rigid motion."""
    return np.concatenate([c for _, c in synth_chunks(h, w, T, seed)])


def synth_chunks(
    h: int, w: int, T: int, seed: int = 0, chunk: int = GEN_CHUNK
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(t0, (n, h, w) uint8)`` chunks of the synthetic movie, bounded memory."""
    rng = np.random.default_rng(seed)
    m = MOTION_MARGIN
    ch, cw = h + 2 * m, w + 2 * m
    patch = _gauss_patch()
    ps = patch.shape[0]
    n_cells = max(1, int(ch * cw / 900))
    ys = rng.integers(0, ch - ps, n_cells)
    xs = rng.integers(0, cw - ps, n_cells)
    C = 20.0 + _traces(n_cells, T, rng) * 60.0  # baseline F0 keeps cells visible
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    vign = 40.0 + 30.0 * np.exp(
        -((yy - ch / 2) ** 2 + (xx - cw / 2) ** 2) / (2 * (0.4 * max(ch, cw)) ** 2)
    )
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * np.arange(T) / (30 * 60))
    steps = rng.integers(-1, 2, (T, 2))
    shifts = np.clip(np.cumsum(steps, axis=0), -m, m)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        n = t1 - t0
        canvas = vign[None] * drift[t0:t1, None, None].astype(np.float32)
        canvas = canvas + rng.standard_normal((n, ch, cw), dtype=np.float32) * 4.0
        for k in range(n_cells):
            canvas[:, ys[k] : ys[k] + ps, xs[k] : xs[k] + ps] += (
                C[k, t0:t1, None, None] * patch[None]
            )
        out = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            dy, dx = shifts[t0 + i]
            out[i] = np.clip(canvas[i, m + dy : m + dy + h, m + dx : m + dx + w], 0, 255)
        yield t0, out


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthSetup:
    ys: np.ndarray
    xs: np.ndarray
    C: np.ndarray  # (n_cells, T) fluorescence incl. the F0 baseline
    vign: np.ndarray  # (ch, cw) background
    drift: np.ndarray  # (T,) background gain
    shifts: np.ndarray  # (T, 2) integer (dy, dx) crop offsets of each frame
    patch: np.ndarray
    margin: int


def synth_setup(h: int, w: int, T: int, seed: int = 0) -> SynthSetup:
    """Everything ``synth_chunks`` draws before its frame loop, in the same order."""
    rng = np.random.default_rng(seed)
    m = MOTION_MARGIN
    ch, cw = h + 2 * m, w + 2 * m
    patch = _gauss_patch()
    ps = patch.shape[0]
    n_cells = max(1, int(ch * cw / 900))
    ys = rng.integers(0, ch - ps, n_cells)
    xs = rng.integers(0, cw - ps, n_cells)
    C = 20.0 + _traces(n_cells, T, rng) * 60.0
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    vign = 40.0 + 30.0 * np.exp(
        -((yy - ch / 2) ** 2 + (xx - cw / 2) ** 2) / (2 * (0.4 * max(ch, cw)) ** 2)
    )
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * np.arange(T) / (30 * 60))
    steps = rng.integers(-1, 2, (T, 2))
    shifts = np.clip(np.cumsum(steps, axis=0), -m, m)
    return SynthSetup(ys, xs, C, vign, drift, shifts, patch, m)


def truth_chunks(
    setup: SynthSetup, h: int, w: int, chunk: int = 100
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(t0, clean, cells)`` per frame chunk.

    ``clean`` (n, h, w) uint8: the recorded frames without noise (same clip and uint8
    cast as the generator). ``cells`` (n, h, w) float32: only the cells' activity above
    their F0 baseline, i.e. the signal CNMF is after. Both in recorded (moving) frame
    coordinates; ``setup.shifts`` gives each frame's offset.
    """
    s, m = setup, setup.margin
    ps = s.patch.shape[0]
    T = s.C.shape[1]
    ch, cw = h + 2 * m, w + 2 * m
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        n = t1 - t0
        canvas = s.vign[None] * s.drift[t0:t1, None, None].astype(np.float32)
        cells = np.zeros((n, ch, cw), dtype=np.float32)
        for k in range(s.ys.size):
            y, x = s.ys[k], s.xs[k]
            canvas[:, y : y + ps, x : x + ps] += s.C[k, t0:t1, None, None] * s.patch[None]
            cells[:, y : y + ps, x : x + ps] += (
                (s.C[k, t0:t1, None, None] - 20.0) * s.patch[None]
            ).astype(np.float32)
        clean = np.empty((n, h, w), dtype=np.uint8)
        cell_out = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            dy, dx = s.shifts[t0 + i]
            sl = (slice(m + dy, m + dy + h), slice(m + dx, m + dx + w))
            clean[i] = np.clip(canvas[i][sl], 0, 255)
            cell_out[i] = cells[i][sl]
        yield t0, clean, cell_out


def true_centroids(setup: SynthSetup) -> np.ndarray:
    """(n_cells, 2) float (y, x) cell centres in the frame coordinates of an unshifted frame."""
    half = setup.patch.shape[0] // 2
    return np.stack(
        [setup.ys + half - setup.margin, setup.xs + half - setup.margin], axis=1
    ).astype(float)


# ---------------------------------------------------------------------------
# Session writer
# ---------------------------------------------------------------------------


def write_session(
    out_dir: Path,
    minutes: float,
    fps: int = 30,
    segment: int = 1000,
    seed: int = 0,
    side: int = 600,
) -> dict:
    """Write ``N.avi`` segments (uncompressed, fourcc GREY) to ``out_dir`` and ``meta.json``
    to ``out_dir.parent``; returns the metadata dict."""
    T = int(round(minutes * 60 * fps))
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"GREY")
    t0 = time.perf_counter()
    seg = -1
    for f0, chunk in synth_chunks(side, side, T, seed, chunk=segment):
        seg = f0 // segment
        writer = cv2.VideoWriter(str(out_dir / f"{seg}.avi"), fourcc, fps, (side, side), False)
        for frame in chunk:
            writer.write(frame)
        writer.release()
    meta = {
        "frames": T,
        "height": side,
        "width": side,
        "fps": fps,
        "segment_frames": segment,
        "segments": seg + 1,
        "fourcc": "GREY",
        "seed": seed,
        "bytes_raw": T * side * side,
        "generate_seconds": time.perf_counter() - t0,
    }
    (out_dir.parent / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
