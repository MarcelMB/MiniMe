"""Rigid motion traces for synthetic sessions (``synth.write_session(..., shifts=...)``).

Each is an integer (T, 2) (dy, dx) trace of AR(1) jitter (sd 0.6 px) plus one failure mode
of real recordings (the Nov 2026 motion study): a persistent offset after the first seconds,
a slow drift over the session, or sustained jumps (cable tugs).
"""

from __future__ import annotations

import numpy as np

SCENARIOS = ("jitter", "offset", "drift", "jumps")


def jitter(rng: np.random.Generator, T: int, sd: float = 0.6, phi: float = 0.8) -> np.ndarray:
    """(T, 2) AR(1) jitter (frame-to-frame correlated) of standard deviation ``sd`` px."""
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), (T, 2))
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def ramp(T: int, t0: int, n: int) -> np.ndarray:
    """(T,) 0 before ``t0``, rising linearly to 1 over ``n`` frames."""
    return np.clip((np.arange(T) - t0) / max(n, 1), 0, 1)


def levels(T: int, steps: list[tuple[int, tuple[float, float]]], n: int = 6) -> np.ndarray:
    """(T, 2) piecewise-constant positions; each step ramps in over ``n`` frames."""
    x = np.zeros((T, 2))
    prev = np.zeros(2)
    for t0, lv in steps:
        x += ramp(T, t0, n)[:, None] * (np.asarray(lv, float) - prev)[None]
        prev = np.asarray(lv, float)
    return x


def scenario_shifts(name: str, T: int, offset_at: int | None = None, seed: int = 1) -> np.ndarray:
    """Integer (T, 2) trace of scenario ``name`` (``SCENARIOS``).

    offset: the field moves by (-5, -7) px at ``offset_at`` (default T // 30) and stays;
    drift: linear to (8, -10) px over the session; jumps: sustained steps to (6, 4),
    (-4, 9) and (2, -3) px at 1/5, 1/2 and 3/4 of the session. All on top of the jitter.
    """
    x = jitter(np.random.default_rng(seed), T)
    if name == "offset":
        t0 = T // 30 if offset_at is None else offset_at
        x += ramp(T, t0, 20)[:, None] * np.array([-5.0, -7.0])[None]
    elif name == "drift":
        x += (np.arange(T) / max(T - 1, 1))[:, None] * np.array([8.0, -10.0])[None]
    elif name == "jumps":
        x += levels(T, [(T // 5, (6, 4)), (T // 2, (-4, 9)), (3 * T // 4, (2, -3))])
    elif name != "jitter":
        raise ValueError(f"unknown motion scenario {name!r}; choose from {SCENARIOS}")
    return np.rint(x).astype(np.int64)
