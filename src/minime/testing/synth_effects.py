"""Pure building blocks for the synthetic session's optional 1p effects.

Every function takes the RNG it draws from, so ``synth`` can give each effect its own child
stream and switching one effect on never changes another's draws. Formulas follow minisim
(github.com/miniscope/minisim) where it has one; everything is in grey levels, pixels and
frames.
"""

from __future__ import annotations

import cv2
import numpy as np

Generator = np.random.Generator


# ---------------------------------------------------------------------------
# Per-cell variation and time courses
# ---------------------------------------------------------------------------


def lognormal_factors(rng: Generator, cv: float, n: int) -> np.ndarray:
    """(n,) lognormal factors with mean 1 and coefficient of variation ``cv`` (ones if 0)."""
    if cv <= 0:
        return np.ones(n)
    s = np.sqrt(np.log1p(cv * cv))
    return rng.lognormal(-0.5 * s * s, s, n)


def ou_series(rng: Generator, T: int, tau: float) -> np.ndarray:
    """(T,) float32 ``exp(0.4 x - 0.08)`` of a unit-variance Ornstein-Uhlenbeck process."""
    a = np.exp(-1.0 / tau)
    noise = rng.standard_normal(T) * np.sqrt(1.0 - a * a)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    for t in range(1, T):
        x[t] = a * x[t - 1] + noise[t]
    return np.exp(0.4 * x - 0.08).astype(np.float32)


def lowpass(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal first-order low-pass ``y[t] = b y[t-1] + (1 - b) x[t]`` along the last axis."""
    b = np.float32(np.exp(-1.0 / tau))
    y = np.empty(x.shape, np.float32)
    acc = np.zeros(x.shape[:-1], np.float32)
    for t in range(x.shape[-1]):
        acc = acc * b + (1 - b) * x[..., t]
        y[..., t] = acc
    return y


def markov_spikes(
    rng: Generator,
    n: int,
    T: int,
    p_on: float = 0.005,
    p_off: float = 0.3,
    rate_on: float = 0.3,
    rate_off: float = 0.003,
) -> np.ndarray:
    """(n, T) float32 0/1 spikes from a two-state (quiescent/active) gate per cell."""
    active = rng.random(n) < p_on / (p_on + p_off)
    out = np.zeros((n, T), np.float32)
    for t in range(T):
        out[:, t] = rng.random(n) < np.where(active, rate_on, rate_off)
        flip = rng.random(n) < np.where(active, p_off, p_on)
        active = active ^ flip
    return out


def calcium(spikes: np.ndarray, decay_tau: float = 15.0, rise_tau: float = 0.0) -> np.ndarray:
    """(n, T) float32 calcium traces: spikes convolved with an exp decay (peak 1 per spike).

    Without a rise the legacy recursion is used bit for bit; with ``rise_tau`` the kernel is
    the peak-normalised double exponential ``exp(-t/decay) - exp(-t/rise)``.
    """
    spikes = np.asarray(spikes, np.float32)
    n, T = spikes.shape
    if rise_tau <= 0:
        decay = np.float32(np.exp(-1.0 / decay_tau))
        C = np.empty_like(spikes)
        acc = np.zeros(n, dtype=np.float32)
        for t in range(T):
            acc = acc * decay + spikes[:, t]
            C[:, t] = acc
        return C
    t = np.arange(int(np.ceil(5 * decay_tau)) + 1)
    k = np.exp(-t / decay_tau) - np.exp(-t / rise_tau)
    k = (k / k.max()).astype(np.float32)
    return np.stack([np.convolve(s, k)[:T] for s in spikes]).astype(np.float32)


def bleach_curve(T: int, tau: float) -> np.ndarray:
    """(T,) global bleaching envelope ``0.5 + 0.5 exp(-t / tau)``."""
    return 0.5 + 0.5 * np.exp(-np.arange(T) / tau)


# ---------------------------------------------------------------------------
# Spatial fields
# ---------------------------------------------------------------------------


def smooth_field(rng: Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Gaussian-filtered white noise, shifted to [0, 1] with peak 1, float32."""
    f = cv2.GaussianBlur(rng.standard_normal(shape), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    f = f - f.min()
    peak = f.max()
    if peak > 0:
        f = f / peak
    return f.astype(np.float32)


def vessel_mask(
    rng: Generator, shape: tuple[int, int], n: int, density: float = 0.05
) -> np.ndarray:
    """Multiplicative transmission of ``n`` branching vessel trees (ones when ``n == 0``).

    Each tree enters from a random edge with a trunk 2-4.5 % of the canvas side wide (13-28 px
    at 600 px, at least 3) and crosses the canvas in gentle bends (radius >= ~150 px),
    tapering; every 80-200 px it gives off a branch at 30-70 degrees, 40-70 % as wide, that
    runs 60-250 px and tapers faster (two levels deep). Transmission is Beer-Lambert
    ``exp(-density * chord)`` (wide vessels dark, thin ones faint), blurred like an
    out-of-focus surface vessel (sigma 0.4 % of the side, at least 1 px).
    """
    ch, cw = shape
    if n <= 0:
        return np.ones(shape, np.float32)
    side = max(ch, cw)
    chord = np.zeros(shape, np.float32)

    def disc(y: float, x: float, r: float) -> None:
        y0, y1 = max(0, int(y - r) - 1), min(ch, int(y + r) + 2)
        x0, x1 = max(0, int(x - r) - 1), min(cw, int(x + r) + 2)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        c = 2 * np.sqrt(np.maximum(r * r - (yy - y) ** 2 - (xx - x) ** 2, 0))
        np.maximum(chord[y0:y1, x0:x1], c, out=chord[y0:y1, x0:x1])

    for _ in range(n):
        edge = int(rng.integers(4))
        u = rng.uniform(0.1, 0.9)
        y, x, theta = {
            0: (0.0, u * cw, np.pi / 2),  # top edge, heading down
            1: (ch - 1.0, u * cw, -np.pi / 2),
            2: (u * ch, 0.0, 0.0),  # left edge, heading right
            3: (u * ch, cw - 1.0, np.pi),
        }[edge]
        width = max(3.0, rng.uniform(0.02, 0.045) * side)
        # (y, x, heading, width, length budget px, depth)
        stack = [(y, x, theta + rng.uniform(-0.4, 0.4), width, 4.0 * side, 0)]
        while stack:
            y, x, theta, width, budget, depth = stack.pop()
            taper = 0.999 if depth == 0 else 0.994
            kappa, next_branch = 0.0, rng.uniform(80, 200)
            while budget > 0 and width >= 2.0:
                if not (-width <= y < ch + width and -width <= x < cw + width):
                    break
                disc(y, x, width / 2)
                kappa = float(np.clip(kappa + rng.normal(0, 0.0008), -0.006, 0.006))
                theta += 2 * kappa
                y += 2 * np.sin(theta)
                x += 2 * np.cos(theta)
                width *= taper
                budget -= 2
                next_branch -= 2
                if next_branch <= 0 and depth < 2:
                    turn = rng.choice([-1.0, 1.0])
                    child = (
                        y,
                        x,
                        theta + turn * rng.uniform(0.5, 1.2),
                        width * rng.uniform(0.4, 0.7),
                        rng.uniform(60, 250),
                        depth + 1,
                    )
                    stack.append(child)
                    theta -= turn * rng.uniform(0.05, 0.2)
                    width *= rng.uniform(0.88, 0.97)
                    next_branch = rng.uniform(80, 200)
    trans = np.exp(-density * chord).astype(np.float32)
    blurred = cv2.GaussianBlur(trans, (0, 0), max(1.0, 0.004 * side), borderType=cv2.BORDER_REFLECT)
    return np.minimum(blurred, 1.0)


def warp_field(rng: Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """(2, h, w) float32 smooth (dy, dx) displacement field with peak magnitude 1."""
    f = np.stack([smooth_field(rng, shape, sigma) for _ in range(2)])
    f -= f.mean(axis=(1, 2), keepdims=True)
    mag = np.hypot(f[0], f[1]).max()
    return (f / mag).astype(np.float32) if mag > 0 else f


def walk_amplitude(rng: Generator, T: int, target: float) -> np.ndarray:
    """(T,) float32 random walk (step sd 0.1) scaled so the 99th percentile of |amp| is
    ``target``."""
    w = np.cumsum(rng.normal(0, 0.1, T))
    q = np.quantile(np.abs(w), 0.99)
    return (w * (target / q if q > 0 else 0.0)).astype(np.float32)


def warp(frame: np.ndarray, field: np.ndarray, amp: float) -> np.ndarray:
    """``frame`` resampled at ``grid + amp * field`` (bilinear, border replicated), float32."""
    if amp == 0:
        return np.asarray(frame, np.float32)
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        np.asarray(frame, np.float32),
        xx + amp * field[1],
        yy + amp * field[0],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def frame_map(rng: Generator, T: int, dropped: int, duplicated: int) -> np.ndarray:
    """(T,) source frame shown by each recorded frame: -1 = black (dropped), a repeat of the
    previous recorded frame for a duplicated one. Frame 0 is never touched."""
    fm = np.arange(T)
    if T < 2 or (dropped <= 0 and duplicated <= 0):
        return fm
    k = min(max(dropped, 0) + max(duplicated, 0), T - 1)
    picked = rng.choice(np.arange(1, T), k, replace=False)
    drops, dups = picked[: max(dropped, 0)], picked[max(dropped, 0) :]
    fm[drops] = -1
    for t in np.sort(dups):
        fm[t] = fm[t - 1]
    return fm


def grin_aperture(shape: tuple[int, int], radius: float) -> tuple[np.ndarray, np.ndarray]:
    """GRIN lens optics on the sensor: ``(transmission, rim_blur)``, both (h, w) float32.

    With ``R = radius`` x half the frame side and ``r`` the distance from the centre, the
    transmission is a broad Gaussian falloff ``exp(-(r/R)^2 / 1.0)`` cut by a soft lens edge
    (logistic at 1.05 R, 0.08 R wide) over a 2 % floor: the radial profile of a UCLA V4
    recording through a GRIN lens (1, 0.79, 0.42, 0.17, 0.03 at r/R = 0.1, 0.5, 0.85, 1.05,
    1.3). ``rim_blur`` = ``(r/R)^4`` clipped to 1 is the weight of a blurred frame (the lens
    aberrates toward its rim).
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    big_r = radius * min(h, w) / 2
    rr = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / big_r
    edge = 1 / (1 + np.exp((rr - 1.05) / 0.08))
    trans = (np.exp(-(rr**2)) * edge * 0.98 + 0.02).astype(np.float32)
    return trans, np.clip(rr**4, 0, 1).astype(np.float32)
