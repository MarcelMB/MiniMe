"""Synthetic miniscope session with ground truth.

Cells are Gaussian patches with spiking exp-decay traces on a smooth vignette background
with slow drift, Gaussian noise and integer random-walk rigid motion, uint8. Optional 1p
effects (``SynthOptions``: drifting background, neuropil, overlapping cells, late-active
cells, bleaching, vessels, large or non-rigid motion, dropped frames, cell variation,
bursting spikes, a non-calcium blob, an edge band, out-of-focus cells) are off by default.
With the defaults the RNG call order (positions, traces, motion steps, then per-chunk noise)
is the legacy one, so a given ``(h, w, T, seed)`` reproduces the same movie as before the options
existed; each effect draws from its own child stream, so enabling one does not move the
cells or change the noise of the others.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse

from minime.testing import synth_effects as fx

MOTION_MARGIN = 8  # px of canvas around the FOV for synthetic motion
GEN_CHUNK = 50  # frames generated at once (bounds memory)
BASELINE_F0 = 20.0  # grey levels; keeps cells visible
CELL_SIGMA = 3.0  # px
CELL_AMP = 60.0  # grey levels per unit transient
DECAY_TAU = 15.0  # frames (0.5 s at 30 fps)
PATCH_MIN = 15

# child RNG streams, one per effect
(
    _S_SIZE,
    _S_BURST,
    _S_OVERLAP,
    _S_BG,
    _S_NP,
    _S_VESSEL,
    _S_WARP,
    _S_FRAMES,
    _S_HEMO,
    _S_LATE,
    _S_OOF,
    _S_RATE,
) = range(12)
DETECT_BORDER = 5  # px: a cell centred closer to the frame edge is not counted as detectable
DETECT_SPIKES = 2  # a cell with fewer spikes in the session is not counted as detectable
EDGE_PX = 12  # a cell centred closer to the frame edge is in the "edge" category
LATE_RATE = 0.05  # spikes per frame of a late-active cell once it starts
OOF_SIGMA = 3.0  # out-of-focus cell sigma, in units of CELL_SIGMA (median)
OOF_AMP = (0.15, 0.35)  # out-of-focus transient amplitude, fraction of CELL_AMP


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthOptions:
    """Optional 1p effects; every default means "off" (the legacy session)."""

    background_drift: float = 0.0  # grey levels of 3 smooth drifting fields
    neuropil: float = 0.0  # grey levels of 3 cell-scale fields coupled to activity
    overlap: int = 0  # extra cells 0.4-0.8 diameters from an existing cell
    overlap_corr: float = 0.8  # share of the partner's spikes an overlap cell inherits
    late_active: int = 0  # in-frame cells silent before late_active_start * T, bursting after
    late_active_start: float = 0.7
    bleach_tau: float | None = None  # frames; global envelope 0.5 + 0.5 exp(-t / tau)
    vessels: int = 0  # branching vessel trees moving with the tissue
    vessel_density: float = 0.05  # Beer-Lambert density per px of vessel chord (darkness)
    motion_step: int = 1  # max |step| of the rigid random walk, px
    motion_margin: int = MOTION_MARGIN  # canvas margin, px; the walk is clipped to it
    nonrigid: float = 0.0  # px, 99th percentile of the warp amplitude
    dropped: int = 0  # frames replaced by a black frame
    duplicated: int = 0  # frames replaced by the previous recorded frame
    size_cv: float = 0.0  # lognormal CV of the cell sigma
    amp_cv: float = 0.0  # lognormal CV of the transient amplitude
    noise_sd: float = 4.0  # grey levels
    bursts: bool = False  # Markov-gated bursting spikes instead of Bernoulli(0.01)
    rise_tau: float = 0.0  # frames; 0 = instantaneous rise
    hemo: float = 0.0  # grey levels of one broad blob oscillating at 0.1 Hz
    edge_band: int = 0  # px of border multiplied by 0.2 (fixed on the sensor)
    out_of_focus: int = 0  # dim, blurred active cells of other depths; not in the truth
    # calibration to a recording (defaults = the legacy session)
    amp_scale: float = 1.0  # multiplies every cell's transient amplitude (CELL_AMP)
    cell_f0: float = BASELINE_F0  # resting fluorescence of a cell, grey levels
    cell_sigma: float = CELL_SIGMA  # median cell sigma, px
    spike_rate: float = 0.01  # mean spikes per frame of a (non-bursting) cell
    rate_cv: float = 0.0  # lognormal CV of the per-cell spike rate (most cells quiet)
    decay_tau: float = DECAY_TAU  # frames
    background: float = 1.0  # scales the vignetted background (40-70 grey levels)
    shot_noise: float = 0.0  # noise sd sqrt(noise_sd^2 + shot_noise^2 * intensity), on sensor
    aperture: float = 0.0  # GRIN lens: circular aperture radius / half the frame side (0 = off)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> SynthOptions:
        return cls(**(d or {}))


SCENARIOS: dict[str, SynthOptions] = {
    "clean": SynthOptions(),
    "background": SynthOptions(background_drift=15.0),
    "neuropil": SynthOptions(neuropil=10.0),
    "overlap": SynthOptions(overlap=3, overlap_corr=0.5),
    "late_active": SynthOptions(late_active=2),
    "bleach": SynthOptions(bleach_tau=3000.0),
    "vessels": SynthOptions(vessels=2),
    "large_motion": SynthOptions(motion_step=3, motion_margin=24),
    "nonrigid": SynthOptions(nonrigid=2.0),
    "dropped": SynthOptions(dropped=5, duplicated=5),
    "cell_variation": SynthOptions(size_cv=0.3, amp_cv=0.3),
    "low_snr": SynthOptions(noise_sd=12.0),
    "bursts": SynthOptions(bursts=True, rise_tau=2.0),
    "hemo": SynthOptions(hemo=20.0),
    "edge": SynthOptions(edge_band=4),
    "grin": SynthOptions(aperture=0.97),
    "out_of_focus": SynthOptions(out_of_focus=4),
    "realistic": SynthOptions(
        background_drift=15.0,
        neuropil=10.0,
        overlap=3,
        late_active=2,
        bleach_tau=3000.0,
        vessels=2,
        size_cv=0.3,
        amp_cv=0.3,
        bursts=True,
        rise_tau=2.0,
    ),
    # every effect at once, its counts sized for a 600 x 600 V4 frame (~420 cells) and
    # tens of thousands of frames (bleaching to ~0.57 over 54 000 frames)
    "hard": SynthOptions(
        background_drift=15.0,
        neuropil=10.0,
        overlap=40,
        overlap_corr=0.8,
        late_active=20,
        bleach_tau=20000.0,
        vessels=1,
        vessel_density=0.025,  # one faint tree
        motion_step=2,
        motion_margin=16,
        nonrigid=2.0,
        dropped=20,
        duplicated=20,
        size_cv=0.3,
        amp_cv=0.3,
        noise_sd=8.0,
        bursts=True,
        rise_tau=2.0,
        hemo=20.0,
        edge_band=4,
        out_of_focus=80,
    ),
    # a UCLA V4 recording through a GRIN lens, write it at 20 fps (--fps 20 --minutes 30):
    # fitted to V4_24h_15072021 file 100 (mean 46, noise sd 2.9, peak transient 26 grey
    # levels above the mean, background sd 21, 16 % dark outside the lens, pixels above
    # 3 sd 0.3 % of the time at the median and 11 % at the 99th percentile); the hard
    # effects stay on, with one faint vessel
    "grin_v4": SynthOptions(
        background_drift=6.0,
        neuropil=4.0,
        overlap=40,
        overlap_corr=0.8,
        late_active=20,
        bleach_tau=15000.0,
        vessels=1,
        vessel_density=0.012,
        nonrigid=1.0,
        dropped=10,
        duplicated=10,
        size_cv=0.3,
        amp_cv=0.3,
        noise_sd=1.5,
        rise_tau=2.0,
        hemo=4.0,
        out_of_focus=100,
        amp_scale=0.65,
        cell_f0=14.0,
        cell_sigma=4.2,
        spike_rate=0.0012,
        rate_cv=3.0,
        decay_tau=24.0,
        background=1.5,
        shot_noise=0.36,
        aperture=0.97,
    ),
}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthSetup:
    """Everything the generator draws before its frame loop; the ground truth."""

    ys: np.ndarray  # (n,) patch top-left in canvas coordinates
    xs: np.ndarray
    C: np.ndarray  # (n, T) emitted fluorescence incl. the F0 baseline and bleaching
    vign: np.ndarray  # (ch, cw) background
    drift: np.ndarray  # (T,) background gain
    shifts: np.ndarray  # (T, 2) integer (dy, dx) crop offsets of each source frame
    patch: np.ndarray  # the first cell's patch (all patches share its size)
    margin: int
    options: SynthOptions
    sigma: np.ndarray  # (n,) cell sigma, px
    amp: np.ndarray  # (n,) transient amplitude, grey levels
    F0: np.ndarray  # (n, T) baseline (20 x bleach)
    spikes: np.ndarray  # (n, T) spike amplitudes before the calcium kernel
    first_active: np.ndarray  # (n,) first frame with a spike, T if never
    group: np.ndarray  # (n,) overlap group id, -1 without partner
    patches: np.ndarray  # (n, ps, ps) per-cell footprints, peak 1
    bleach: np.ndarray  # (T,) global envelope, ones when off
    bg_fields: np.ndarray  # (3, ch, cw) drifting background components (empty when off)
    bg_time: np.ndarray  # (3, T)
    neuropil_fields: np.ndarray  # (3, ch, cw)
    neuropil_time: np.ndarray  # (3, T)
    vessel_mask: np.ndarray  # (ch, cw) transmission, ones when off
    warp_field: np.ndarray  # (2, h, w) unit-peak displacement (empty when off)
    warp_amp: np.ndarray  # (T,) px
    hemo_field: np.ndarray  # (ch, cw)
    hemo_time: np.ndarray  # (T,)
    frame_map: np.ndarray  # (T,) source frame per recorded frame, -1 = black
    aperture: np.ndarray  # (h, w) GRIN transmission on the sensor (empty when off)
    rim_blur: np.ndarray  # (h, w) weight of the blurred frame toward the lens rim
    late: np.ndarray  # (n,) bool, the late-active cells
    oof_ys: np.ndarray  # (m,) out-of-focus patch top-left in canvas coordinates
    oof_xs: np.ndarray
    oof_sigma: np.ndarray  # (m,) px
    oof_signal: np.ndarray  # (m, T) emitted fluorescence (bleached, no baseline)
    oof_patches: np.ndarray  # (m, ps, ps) footprints, peak 1


def _gauss_patch(size: int = PATCH_MIN, sigma: float = CELL_SIGMA) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma**2))
    return (g / g.max()).astype(np.float32)


def _patch_size(sigma: np.ndarray) -> int:
    """Odd patch side holding ±2.5 sigma of the largest cell, at least the legacy 15."""
    return max(PATCH_MIN, int(np.ceil(5 * sigma.max())) | 1)


def _cells(
    rng: np.random.Generator, child, o: SynthOptions, h: int, w: int, m: int, T: int
) -> tuple[np.ndarray, ...]:
    """Positions, spikes and amplitudes of all cells. Legacy draws (positions, spikes, spike
    amplitudes, motion steps) first, in the legacy order, then the effects on their streams.
    Returns ``(ys, xs, spikes, amp_u, steps, sigma, amp_f, group, late)``."""
    ch, cw = h + 2 * m, w + 2 * m
    n_base = max(1, int(ch * cw / 900))
    n = n_base + o.overlap
    r_size = child(_S_SIZE)
    sigma = o.cell_sigma * fx.lognormal_factors(r_size, o.size_cv, n)
    amp_f = fx.lognormal_factors(r_size, o.amp_cv, n)
    ps = _patch_size(sigma)

    ys = rng.integers(0, ch - ps, n_base)
    xs = rng.integers(0, cw - ps, n_base)
    rate = o.spike_rate * fx.lognormal_factors(child(_S_RATE), o.rate_cv, n_base)
    spikes = (rng.random((n_base, T)) < rate[:, None]).astype(np.float32)
    amp_u = rng.uniform(0.5, 1.5, (n_base, 1)).astype(np.float32)
    steps = rng.integers(-o.motion_step, o.motion_step + 1, (T, 2))

    if o.bursts:
        spikes = fx.markov_spikes(child(_S_BURST), n_base, T)
    late_mask = np.zeros(n, bool)
    if o.late_active > 0:
        # the last late_active cells whose centre is well inside the frame: silent, then
        # bursting at LATE_RATE from late_active_start on (the long-recording unit-loss case)
        cy, cx = ys + ps // 2 - m, xs + ps // 2 - m
        inside = (cy >= DETECT_BORDER) & (cy < h - DETECT_BORDER)
        inside &= (cx >= DETECT_BORDER) & (cx < w - DETECT_BORDER)
        late = np.flatnonzero(inside)[-o.late_active :]
        start = int(o.late_active_start * T)
        spikes[late] = 0.0
        spikes[late, start:] = child(_S_LATE).random((late.size, T - start)) < LATE_RATE
        late_mask[late] = True
    group = np.full(n, -1)
    if o.overlap > 0:
        r = child(_S_OVERLAP)
        partners = r.choice(n_base, o.overlap, replace=o.overlap > n_base)
        dist = r.uniform(0.4, 0.8, o.overlap) * 4 * o.cell_sigma
        ang = r.uniform(0, 2 * np.pi, o.overlap)
        oy = np.clip(np.rint(ys[partners] + dist * np.sin(ang)), 0, ch - ps).astype(ys.dtype)
        ox = np.clip(np.rint(xs[partners] + dist * np.cos(ang)), 0, cw - ps).astype(xs.dtype)
        own = (r.random((o.overlap, T)) < o.spike_rate).astype(np.float32)
        ospk = o.overlap_corr * spikes[partners] + (1 - o.overlap_corr) * own
        ys, xs = np.concatenate([ys, oy]), np.concatenate([xs, ox])
        spikes = np.concatenate([spikes, ospk.astype(np.float32)])
        amp_u = np.concatenate([amp_u, r.uniform(0.5, 1.5, (o.overlap, 1)).astype(np.float32)])
        group[partners] = np.arange(o.overlap)
        group[n_base:] = np.arange(o.overlap)
    return ys, xs, spikes, amp_u, steps, sigma, amp_f, group, late_mask


def _out_of_focus(
    r: np.random.Generator, o: SynthOptions, ch: int, cw: int, T: int, bleach: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Cells above or below the focal plane: wide (sigma ~3x), dim (15-35 % of an in-focus
    transient), with their own activity. Returns ``(ys, xs, sigma, signal, patches)``."""
    m = o.out_of_focus
    sigma = o.cell_sigma * OOF_SIGMA * fx.lognormal_factors(r, 0.2, m)
    ps = min(_patch_size(sigma), min(ch, cw))
    ys = r.integers(0, ch - ps + 1, m)
    xs = r.integers(0, cw - ps + 1, m)
    spikes = (
        fx.markov_spikes(r, m, T)
        if o.bursts
        else (r.random((m, T)) < o.spike_rate).astype(np.float32)
    )
    amp = r.uniform(*OOF_AMP, m).astype(np.float32) * np.float32(CELL_AMP * o.amp_scale)
    signal = fx.calcium(spikes, o.decay_tau, o.rise_tau) * amp[:, None]
    signal = (signal * bleach.astype(np.float32)[None]).astype(np.float32)
    patches = np.stack([_gauss_patch(ps, s) for s in sigma])
    return ys, xs, sigma, signal, patches


def _build(
    h: int, w: int, T: int, seed: int, options: SynthOptions | None
) -> tuple[SynthSetup, np.random.Generator]:
    """All draws before the frame loop; returns the truth and the RNG positioned for noise."""
    o = options or SynthOptions()
    rng = np.random.default_rng(seed)
    kids = np.random.SeedSequence(seed).spawn(12)  # child i is the same for any count > i

    def child(i: int) -> np.random.Generator:
        return np.random.default_rng(kids[i])

    m = o.motion_margin
    ch, cw = h + 2 * m, w + 2 * m
    ys, xs, spikes, amp_u, steps, sigma, amp_f, group, late = _cells(rng, child, o, h, w, m, T)
    n = ys.size

    # exactly 60.0 per cell when amp_cv == 0 and amp_scale == 1
    scale = (CELL_AMP * o.amp_scale * amp_f).astype(np.float32)
    amp = scale * amp_u[:, 0]
    transient = fx.calcium(spikes * amp_u, o.decay_tau, o.rise_tau)  # legacy: traces * amp_u
    bleach = fx.bleach_curve(T, o.bleach_tau) if o.bleach_tau else np.ones(T)
    signal = transient * scale[:, None]  # (n, T) above F0
    bleach32 = bleach.astype(np.float32)
    C = ((np.float32(o.cell_f0) + signal) * bleach32[None]).astype(np.float32)
    F0 = np.broadcast_to((np.float32(o.cell_f0) * bleach32)[None], (n, T))
    has = spikes > 0
    first_active = np.where(has.any(axis=1), has.argmax(axis=1), T)

    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    vign = 40.0 + 30.0 * np.exp(
        -((yy - ch / 2) ** 2 + (xx - cw / 2) ** 2) / (2 * (0.4 * max(ch, cw)) ** 2)
    )
    if o.background != 1.0:
        vign = vign * np.float32(o.background)
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * np.arange(T) / (30 * 60))
    shifts = np.clip(np.cumsum(steps, axis=0), -m, m)
    patches = np.stack([_gauss_patch(_patch_size(sigma), s) for s in sigma])

    empty_f = np.zeros((0, ch, cw), np.float32)
    empty_t = np.zeros((0, T), np.float32)
    bg_fields, bg_time = empty_f, empty_t
    if o.background_drift > 0:
        r = child(_S_BG)
        bg_fields = np.stack([fx.smooth_field(r, (ch, cw), max(ch, cw) / 4) for _ in range(3)])
        bg_time = np.stack([fx.ou_series(r, T, 300.0) for _ in range(3)])
    np_fields, np_time = empty_f, empty_t
    if o.neuropil > 0:
        r = child(_S_NP)
        np_sigma = 2 * float(np.median(sigma))
        np_fields = np.stack([fx.smooth_field(r, (ch, cw), np_sigma) for _ in range(3)])
        pop = fx.lowpass(signal.mean(axis=0, keepdims=True), 45.0)[0]
        pop = pop / pop.mean() if pop.mean() > 0 else np.ones(T, np.float32)
        np_time = np.stack([0.7 * pop + 0.3 * fx.ou_series(r, T, 300.0) for _ in range(3)])
    vessel_mask = (
        fx.vessel_mask(child(_S_VESSEL), (ch, cw), o.vessels, o.vessel_density)
        if o.vessels > 0
        else np.ones((ch, cw), np.float32)
    )
    warp_field, warp_amp = np.zeros((0, h, w), np.float32), np.zeros(T, np.float32)
    if o.nonrigid > 0:
        r = child(_S_WARP)
        warp_field = fx.warp_field(r, (h, w), h / 3)
        warp_amp = fx.walk_amplitude(r, T, o.nonrigid)
    hemo_field, hemo_time = np.zeros((ch, cw), np.float32), np.zeros(T, np.float32)
    if o.hemo > 0:
        hemo_field = fx.smooth_field(child(_S_HEMO), (ch, cw), max(ch, cw) / 3)
        hemo_time = (0.5 * (1 + np.sin(2 * np.pi * 0.1 / 30 * np.arange(T)))).astype(np.float32)
    frame_map = fx.frame_map(child(_S_FRAMES), T, o.dropped, o.duplicated)
    aperture, rim_blur = (
        fx.grin_aperture((h, w), o.aperture)
        if o.aperture > 0
        else (np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32))
    )
    if o.out_of_focus > 0:
        oof = _out_of_focus(child(_S_OOF), o, ch, cw, T, bleach)
    else:
        oof = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), empty_t, np.zeros((0, 1, 1)))

    setup = SynthSetup(
        ys=ys,
        xs=xs,
        C=C,
        vign=vign,
        drift=drift,
        shifts=shifts,
        patch=patches[0],
        margin=m,
        options=o,
        sigma=sigma,
        amp=amp,
        F0=F0,
        spikes=spikes,
        first_active=first_active,
        group=group,
        patches=patches,
        bleach=bleach,
        bg_fields=bg_fields,
        bg_time=bg_time,
        neuropil_fields=np_fields,
        neuropil_time=np_time,
        vessel_mask=vessel_mask,
        warp_field=warp_field,
        warp_amp=warp_amp,
        hemo_field=hemo_field,
        hemo_time=hemo_time,
        frame_map=frame_map,
        aperture=aperture,
        rim_blur=rim_blur,
        late=late,
        oof_ys=oof[0],
        oof_xs=oof[1],
        oof_sigma=oof[2],
        oof_signal=oof[3],
        oof_patches=oof[4].astype(np.float32),
    )
    return setup, rng


def synth_setup(
    h: int, w: int, T: int, seed: int = 0, options: SynthOptions | None = None
) -> SynthSetup:
    """Everything ``synth_chunks`` draws before its frame loop, in the same order."""
    return _build(h, w, T, seed, options)[0]


def true_centroids(setup: SynthSetup) -> np.ndarray:
    """(n_cells, 2) float (y, x) cell centres in the frame coordinates of an unshifted frame."""
    half = setup.patch.shape[0] // 2
    return np.stack(
        [setup.ys + half - setup.margin, setup.xs + half - setup.margin], axis=1
    ).astype(float)


def detectable(setup: SynthSetup, h: int, w: int) -> np.ndarray:
    """(n,) bool: cells a pipeline can be expected to find, i.e. centred at least
    ``DETECT_BORDER`` px inside the frame with at least ``DETECT_SPIKES`` spikes."""
    c = true_centroids(setup)
    inside = (c >= DETECT_BORDER).all(axis=1) & (c < np.array([h, w]) - DETECT_BORDER).all(axis=1)
    return inside & ((setup.spikes > 0).sum(axis=1) >= DETECT_SPIKES)


def oof_centroids(setup: SynthSetup) -> np.ndarray:
    """(m, 2) float (y, x) out-of-focus cell centres in unshifted frame coordinates."""
    half = setup.oof_patches.shape[-1] // 2
    return np.stack(
        [setup.oof_ys + half - setup.margin, setup.oof_xs + half - setup.margin], axis=1
    ).astype(float)


def categories(setup: SynthSetup, h: int, w: int) -> dict[str, np.ndarray]:
    """(n,) bool masks of the cells each effect makes hard to find, for a recall per category.

    ``overlap``: one of a close, correlated pair; ``near_vessel``: a vessel crosses the cell
    (transmission < 0.8 where its footprint is above 20 %); ``late``: silent until late in
    the session; ``small`` / ``dim``: sigma or transient amplitude well below the typical
    cell; ``edge``: centred within ``EDGE_PX`` of the frame edge; ``near_out_of_focus``: an
    out-of-focus cell's core (one sigma) covers the centre.
    """
    s, c = setup, true_centroids(setup)
    ps = s.patches.shape[-1]
    vessel = np.array(
        [
            s.vessel_mask[y : y + ps, x : x + ps][s.patches[k] > 0.2].min() < 0.8
            for k, (y, x) in enumerate(zip(s.ys, s.xs, strict=True))
        ]
    )
    edge = np.minimum(c, np.array([h, w]) - 1 - c).min(axis=1) < EDGE_PX
    near_oof = np.zeros(c.shape[0], bool)
    oc = oof_centroids(s)
    if oc.size:
        d = np.hypot(c[:, None, 0] - oc[None, :, 0], c[:, None, 1] - oc[None, :, 1])
        near_oof = (d < s.oof_sigma[None]).any(axis=1)
    return {
        "detectable": detectable(s, h, w),
        "overlap": s.group >= 0,
        "near_vessel": vessel,
        "late": s.late.copy(),
        "small": s.sigma < 0.8 * s.options.cell_sigma,
        "dim": s.amp < 0.6 * CELL_AMP * s.options.amp_scale,
        "edge": edge,
        "near_out_of_focus": near_oof,
    }


def footprints(setup: SynthSetup, h: int, w: int) -> sparse.csr_matrix:
    """(n, h*w) float32 cell footprints (peak 1, values >= 1e-3) in unshifted frame
    coordinates, row-major, cut at the frame edge."""
    s, ps = setup, setup.patches.shape[-1]
    rows, cols, vals = [], [], []
    for k in range(s.ys.size):
        yy, xx = np.mgrid[0:ps, 0:ps]
        y, x = yy + s.ys[k] - s.margin, xx + s.xs[k] - s.margin
        keep = (y >= 0) & (y < h) & (x >= 0) & (x < w) & (s.patches[k] >= 1e-3)
        rows.append(np.full(int(keep.sum()), k))
        cols.append((y * w + x)[keep])
        vals.append(s.patches[k][keep])
    n = s.ys.size
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, h * w),
        dtype=np.float32,
    )


def truth_arrays(setup: SynthSetup, h: int, w: int) -> dict[str, np.ndarray]:
    """What scoring needs, as plain arrays (the content of ``truth.npz`` minus footprints).

    ``traces`` are the cells' activity above F0 in recorded frame order (zeros on dropped
    frames, ``frame_ok`` False there); ``shifts`` are the source frames' crop offsets.
    """
    s = setup
    above = (s.C - s.F0).astype(np.float32)
    ok = s.frame_map >= 0
    traces = np.zeros_like(above)
    traces[:, ok] = above[:, s.frame_map[ok]]
    out = {
        "centroids": true_centroids(s),
        "traces": traces,
        "frame_ok": ok,
        "shifts": s.shifts,
        "spikes": s.spikes.astype(np.float32),
        "sigma": s.sigma,
        "amp": s.amp,
        "group": s.group,
        "first_active": s.first_active,
        "oof_centroids": oof_centroids(s),
        "oof_sigma": s.oof_sigma,
    }
    out.update({f"cat_{k}": v for k, v in categories(s, h, w).items()})
    return out


def write_truth(setup: SynthSetup, h: int, w: int, path: Path) -> None:
    """``truth.npz``: ``truth_arrays`` plus the footprints (``A_data``, ``A_indices``,
    ``A_indptr``, ``A_shape`` of a CSR matrix), so scoring does not regenerate the session."""
    A = footprints(setup, h, w)
    arrays = truth_arrays(setup, h, w)
    arrays.update(A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=A.shape)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Rendering (shared by the generator and the truth)
# ---------------------------------------------------------------------------


def _render(
    s: SynthSetup, t0: int, t1: int, noise: np.ndarray | None, want_cells: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Canvas (n, ch, cw) float32 of source frames ``t0:t1`` and, if wanted, the cells-only
    signal above F0. ``noise`` is added before the cells (the legacy order)."""
    o, n = s.options, t1 - t0
    ch, cw = s.vign.shape
    gain = (s.drift[t0:t1] * s.bleach[t0:t1])[:, None, None].astype(np.float32)
    canvas = s.vign[None] * gain
    if o.background_drift > 0:
        canvas += np.float32(o.background_drift / 3) * np.tensordot(
            s.bg_time[:, t0:t1].T, s.bg_fields, axes=1
        )
    if o.neuropil > 0:
        canvas += np.float32(o.neuropil / 3) * np.tensordot(
            s.neuropil_time[:, t0:t1].T, s.neuropil_fields, axes=1
        )
    if o.hemo > 0:
        canvas += np.float32(o.hemo) * s.hemo_time[t0:t1, None, None] * s.hemo_field[None]
    if noise is not None:
        canvas += noise  # canvas is a fresh array (vign * gain), so in place is safe
    cells = np.zeros((n, ch, cw), np.float32) if want_cells else None
    ps = s.patch.shape[0]
    for k in range(s.ys.size):
        sl = (slice(None), slice(s.ys[k], s.ys[k] + ps), slice(s.xs[k], s.xs[k] + ps))
        canvas[sl] += s.C[k, t0:t1, None, None] * s.patches[k][None]
        if cells is not None:
            cells[sl] += (s.C[k, t0:t1] - s.F0[k, t0:t1])[:, None, None] * s.patches[k][None]
    ops = s.oof_patches.shape[-1]
    for k in range(s.oof_ys.size):  # out-of-focus cells are background to CNMF: not in cells
        sl = (
            slice(None),
            slice(s.oof_ys[k], s.oof_ys[k] + ops),
            slice(s.oof_xs[k], s.oof_xs[k] + ops),
        )
        canvas[sl] += s.oof_signal[k, t0:t1, None, None] * s.oof_patches[k][None]
    if o.vessels > 0:
        canvas *= s.vessel_mask[None]
        if cells is not None:
            cells *= s.vessel_mask[None]
    return canvas, cells


def _crop_frames(s: SynthSetup, canvas: np.ndarray, t0: int, h: int, w: int) -> np.ndarray:
    """Recorded (n, h, w) float32 frames: rigid crop, non-rigid warp, GRIN aperture, edge
    band."""
    o, m = s.options, s.margin
    out = np.empty((canvas.shape[0], h, w), np.float32)
    for i in range(canvas.shape[0]):
        dy, dx = s.shifts[t0 + i]
        fr = canvas[i, m + dy : m + dy + h, m + dx : m + dx + w]
        out[i] = fx.warp(fr, s.warp_field, float(s.warp_amp[t0 + i])) if o.nonrigid > 0 else fr
    if o.aperture > 0:  # fixed on the sensor: blur toward the rim, then the lens transmission
        for i in range(out.shape[0]):
            blurred = cv2.GaussianBlur(out[i], (0, 0), 2.5, borderType=cv2.BORDER_REFLECT)
            out[i] += s.rim_blur * (blurred - out[i])
        out *= s.aperture[None]
    if o.edge_band > 0:
        b = o.edge_band
        out[:, :b] *= 0.2
        out[:, -b:] *= 0.2
        out[:, :, :b] *= 0.2
        out[:, :, -b:] *= 0.2
    return out


def _apply_frame_map(
    s: SynthSetup, frames: np.ndarray, t0: int, prev: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Recorded frames from source frames ``t0:``; returns them and the last one (carried
    into the next chunk for a duplicate at a chunk start)."""
    o = s.options
    if o.dropped <= 0 and o.duplicated <= 0:
        return frames, frames[-1]
    out = np.empty_like(frames)
    for i in range(frames.shape[0]):
        src = s.frame_map[t0 + i]
        if src < 0:
            out[i] = 0.0
        elif src >= t0:
            out[i] = frames[src - t0]
        else:
            out[i] = out[i - 1] if i > 0 else prev
    return out, out[-1]


# ---------------------------------------------------------------------------
# Synthetic movie
# ---------------------------------------------------------------------------


def synth_movie(
    h: int, w: int, T: int, seed: int = 0, options: SynthOptions | None = None
) -> np.ndarray:
    """(T, h, w) uint8 movie with cells, background, noise and motion."""
    return np.concatenate([c for _, c in synth_chunks(h, w, T, seed, options=options)])


def synth_chunks(
    h: int,
    w: int,
    T: int,
    seed: int = 0,
    chunk: int = GEN_CHUNK,
    options: SynthOptions | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(t0, (n, h, w) uint8)`` chunks of the synthetic movie, bounded memory."""
    s, rng = _build(h, w, T, seed, options)
    ch, cw = s.vign.shape
    prev = None
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        o = s.options
        noise = rng.standard_normal((t1 - t0, ch, cw), dtype=np.float32)
        if o.shot_noise > 0:  # on the sensor, after the optics: sd grows with intensity
            canvas, _ = _render(s, t0, t1, None, want_cells=False)
            frames = _crop_frames(s, canvas, t0, h, w)
            del canvas
            sd = np.sqrt(o.noise_sd**2 + o.shot_noise**2 * np.maximum(frames, 0))
            frames += noise[:, :h, :w] * sd
            del noise, sd
        else:
            noise *= o.noise_sd
            canvas, _ = _render(s, t0, t1, noise, want_cells=False)
            del noise  # bounds the peak at two canvases plus one frame array
            frames = _crop_frames(s, canvas, t0, h, w)
            del canvas
        frames, prev = _apply_frame_map(s, frames, t0, prev)
        yield t0, np.clip(frames, 0, 255, out=frames).astype(np.uint8)


def truth_chunks(
    setup: SynthSetup, h: int, w: int, chunk: int = 100
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(t0, clean, cells)`` per frame chunk.

    ``clean`` (n, h, w) uint8: the recorded frames without noise (same clip and uint8
    cast as the generator). ``cells`` (n, h, w) float32: only the cells' activity above
    their F0 baseline, i.e. the signal CNMF is after. Both in recorded (moving) frame
    coordinates and recorded frame order (``setup.frame_map``); ``setup.shifts`` gives
    each source frame's offset.
    """
    s = setup
    T = s.C.shape[1]
    prev_c = prev_s = None
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        canvas, cells = _render(s, t0, t1, None, want_cells=True)
        assert cells is not None
        clean, prev_c = _apply_frame_map(s, _crop_frames(s, canvas, t0, h, w), t0, prev_c)
        cell_out, prev_s = _apply_frame_map(s, _crop_frames(s, cells, t0, h, w), t0, prev_s)
        yield t0, np.clip(clean, 0, 255, out=clean).astype(np.uint8), cell_out


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
    options: SynthOptions | None = None,
) -> dict:
    """Write ``N.avi`` segments (uncompressed, fourcc GREY) to ``out_dir`` and ``meta.json``
    and ``truth.npz`` (``write_truth``) to ``out_dir.parent``; returns the metadata dict."""
    T = int(round(minutes * 60 * fps))
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"GREY")
    t0 = time.perf_counter()
    seg = -1
    for f0, chunk in synth_chunks(side, side, T, seed, chunk=segment, options=options):
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
        "options": (options or SynthOptions()).to_dict(),
        "bytes_raw": T * side * side,
        "generate_seconds": time.perf_counter() - t0,
    }
    write_truth(synth_setup(side, side, T, seed, options), side, side, out_dir.parent / "truth.npz")
    (out_dir.parent / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta
