"""Trace overview, windows and the CaTune-style low-pass (cutoff in Hz)."""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_LEN = 8  # shorter traces are returned unchanged (as CaLab's MIN_SPECTRUM_LEN)


def lowpass_gain(n: int, fs: float, f_lp: float) -> np.ndarray:
    """rfft gain of CaLab's low-pass: 1 below f_lp/2, cosine taper to 0 at 1.5 f_lp.

    Same curve as ``BandpassFilter.build_gain_curve`` in CaLab's ``crates/solver/src/
    filter.rs`` with only the low-pass half on; there the cutoff comes from tau_rise, here
    it is given in Hz. ``f_lp`` is clamped to Nyquist first, as there.
    """
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f_lp = min(float(f_lp), fs / 2.0)
    w = 0.5 * f_lp
    g = np.ones_like(f)
    band = (f >= f_lp - w) & (f < f_lp + w)
    g[band] = 0.5 * (1.0 + np.cos(np.pi * (f[band] - (f_lp - w)) / (2.0 * w)))
    g[f >= f_lp + w] = 0.0
    return g


def lowpass(x: np.ndarray, fs: float, f_lp: float) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.size < MIN_LEN:
        return x.copy()
    if f_lp <= 0 or fs <= 0:
        raise ValueError(f"cutoff and frame rate must be > 0 (got {f_lp} Hz, {fs} Hz)")
    y = np.fft.irfft(np.fft.rfft(x) * lowpass_gain(x.size, fs, f_lp), n=x.size)
    return y.astype(np.float32)


def window(
    trace: np.ndarray,
    start: int,
    stop: int,
    fs: float | None = None,
    lowpass_hz: float | None = None,
    margin_s: float = 5.0,
) -> np.ndarray:
    """``trace[start:stop]``, low-passed with ``margin_s`` of extra trace on each side."""
    T = len(trace)
    if not 0 <= start < stop <= T:
        raise IndexError(f"window {start}:{stop} outside 0:{T}")
    if lowpass_hz is None:
        return np.asarray(trace[start:stop], np.float32)
    if fs is None:
        raise ValueError("the low-pass needs the frame rate of the recording")
    m = int(round(margin_s * fs))
    a, b = max(0, start - m), min(T, stop + m)
    return lowpass(np.asarray(trace[a:b], np.float32), fs, lowpass_hz)[start - a : stop - a]


def overview(C: np.ndarray, bins: int) -> np.ndarray:
    """(n, 2, bins) float32 min / max of each trace over ``bins`` equal spans of frames."""
    n, T = C.shape
    bins = max(1, min(int(bins), T))
    starts = np.linspace(0, T, bins + 1).astype(int)[:-1]
    out = np.empty((n, 2, bins), np.float32)
    for i in range(0, n, 64):
        blk = np.asarray(C[i : i + 64], np.float32)
        out[i : i + 64, 0] = np.minimum.reduceat(blk, starts, axis=1)
        out[i : i + 64, 1] = np.maximum.reduceat(blk, starts, axis=1)
    return out


def zooms(
    peak: np.ndarray,
    T: int,
    fps: float | None,
    n_units: int = 10,
    wnd_min: float = 5.0,
    per_hour: int = 3,
) -> dict[str, Any]:
    """The ``n_units`` brightest units and ``wnd_min``-minute windows, ``per_hour`` per hour."""
    units = [int(k) for k in np.argsort(np.asarray(peak))[::-1][:n_units]]
    if not fps:
        return {"units": units, "windows": []}
    step = max(1, int(3600 / per_hour * fps))
    wnd = int(wnd_min * 60 * fps)
    windows = [{"start": t0, "stop": min(t0 + wnd, T)} for t0 in range(0, T, step)]
    return {"units": units, "windows": windows}
