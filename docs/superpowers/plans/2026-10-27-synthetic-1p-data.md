# Richer synthetic 1p test data: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `minime.testing.synth` with per-effect toggles (background drift, neuropil, overlapping cells, late-active cells, bleaching, vessels, large and non-rigid motion, dropped and duplicated frames, cell size and brightness variation, bursting spikes, a non-calcium blob, an edge band), each with exact ground truth, keeping the default session byte-identical, and add one pipeline test per effect.

**Architecture:** `SynthOptions` (frozen dataclass, all defaults "off") drives `_build` (all random draws: legacy ones on the main RNG in the current order, each effect on its own child stream), `_render` (noise-free canvas + cells-only signal, shared by generator and truth), `_crop_frames` (rigid crop, non-rigid warp, edge band) and `_apply_frame_map` (dropped and duplicated frames). Pure numpy building blocks (smooth fields, OU series, Markov spikes, calcium kernel, vessel mask, warp, frame map) live in a new `synth_effects.py`.

**Tech Stack:** numpy, OpenCV (`cv2.GaussianBlur`, `cv2.remap`), pytest, ruff. Python venv `.venv`.

**Spec:** `docs/superpowers/specs/2026-10-27-synthetic-1p-data-design.md`

## Global Constraints

- The default session must stay byte-identical: `sha256(synth_movie(64, 64, 300, 0).tobytes())[:16] == "ce39843f43462047"`, `(32, 32, 20, 1) → "de776615f5097caf"`, `(128, 128, 300, 0) → "b9139b8021c0ba0a"` (computed from the pre-refactor generator on 2026-10-27).
- Public API kept: `synth_movie(h, w, T, seed=0)`, `synth_chunks(h, w, T, seed=0, chunk=GEN_CHUNK)`, `synth_setup(h, w, T, seed=0)`, `truth_chunks(setup, h, w, chunk=100)`, `true_centroids(setup)`, `write_session(out_dir, minutes, fps=30, segment=1000, seed=0, side=600)`, `SynthSetup` fields `ys, xs, C, vign, drift, shifts, patch, margin`. New parameters are appended with defaults.
- Do not edit existing tests beyond additions to `tests/test_synth.py` (a parallel session prunes `tests/`; `tests/conftest.py` is changing, so the new test module does not import from it).
- Units: grey levels on uint8 frames, 30 fps assumed (decay 15 frames = 0.5 s), noise sd 4, baseline F0 = 20, cell sigma 3 px, amplitude 60.
- Run tests with `.venv/bin/python -m pytest -q`; finish with `.venv/bin/ruff format src tests bench && .venv/bin/ruff check --fix src tests bench`.
- Commits: one sentence, no Claude attribution.

## Review Focus

1. `overlap > n_cells` (tiny canvases): partners must be drawn with replacement, never raise. Pinned in Task 3 (`test_overlap_more_than_cells`).
2. `late_active > n_cells`: clamp to all cells, never a negative slice. Pinned in Task 3 (`test_late_active_clamped`).
3. Duplicated frame right after a dropped frame, and a duplicate as the first frame of a chunk: the recorded movie must equal `frame_map` semantics across chunk borders for both the generator (chunk 50) and `truth_chunks` (chunk 100). Pinned in Task 3 (`test_frame_map_matches_movie` uses `dropped=5, duplicated=5` on 300 frames with `chunk=7`).
4. `motion_step > motion_margin` or `nonrigid` large: shifts are clipped to the margin; the warp uses border replication so no NaN or black corners. Pinned in Task 2 (`test_warp_replicates_border`) and Task 3 (`test_large_motion_clipped`).
5. `meta.json` written by an older generator (no `options` key) must still score. Pinned in Task 5 (`test_score_meta_without_options`).

---

### Task 1: Pin the default session before touching the generator

**Files:**
- Modify: `tests/test_synth.py` (append)

**Interfaces:**
- Produces: the three checksums every later task must keep.

- [ ] **Step 1: Append the checksum test**

```python
import hashlib


def _digest(h, w, T, seed, **kw):
    return hashlib.sha256(synth.synth_movie(h, w, T, seed, **kw).tobytes()).hexdigest()[:16]


def test_default_session_is_byte_identical():
    # pinned on 2026-10-27 from the pre-options generator
    assert _digest(64, 64, 300, 0) == "ce39843f43462047"
    assert _digest(32, 32, 20, 1) == "de776615f5097caf"
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_synth.py -q`
Expected: PASS (5 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_synth.py
git commit -m "Pin the default synthetic session's bytes before adding effect options"
```

---

### Task 2: Building blocks in `synth_effects.py`

**Files:**
- Create: `src/minime/testing/synth_effects.py`
- Modify: `tests/test_synth.py` (append block tests)

**Interfaces:**
- Produces (all pure, RNG passed in):
  - `lognormal_factors(rng, cv: float, n: int) -> np.ndarray` (n,) float64, mean 1; ones when `cv <= 0`
  - `smooth_field(rng, shape: tuple[int, int], sigma: float) -> np.ndarray` float32 in [0, 1], peak 1
  - `ou_series(rng, T: int, tau: float) -> np.ndarray` (T,) float32, `exp(0.4·x − 0.08)` of a unit-variance OU process
  - `lowpass(x: np.ndarray, tau: float) -> np.ndarray` causal first-order low-pass along the last axis
  - `markov_spikes(rng, n: int, T: int) -> np.ndarray` (n, T) float32 of 0/1
  - `calcium(spikes: np.ndarray, decay_tau: float = 15.0, rise_tau: float = 0.0) -> np.ndarray` (n, T) float32
  - `bleach_curve(T: int, tau: float) -> np.ndarray` (T,) float64, `0.5 + 0.5·exp(−t/tau)`
  - `vessel_mask(rng, shape, n: int, opacity: float = 0.7) -> np.ndarray` float32 in (0, 1], ones when `n == 0`
  - `warp_field(rng, shape, sigma: float) -> np.ndarray` (2, h, w) float32, max displacement magnitude 1
  - `walk_amplitude(rng, T: int, target: float) -> np.ndarray` (T,) float32, 99th percentile of |amp| = target
  - `warp(frame: np.ndarray, field: np.ndarray, amp: float) -> np.ndarray` float32 (h, w)
  - `frame_map(rng, T: int, dropped: int, duplicated: int) -> np.ndarray` (T,) int64

- [ ] **Step 1: Write the failing block tests** (append to `tests/test_synth.py`)

```python
from minime.testing import synth_effects as fx


def test_lognormal_factors_mean_one():
    rng = np.random.default_rng(0)
    f = fx.lognormal_factors(rng, 0.3, 20000)
    assert abs(f.mean() - 1.0) < 0.02 and abs(f.std() - 0.3) < 0.02
    assert np.array_equal(fx.lognormal_factors(rng, 0.0, 5), np.ones(5))


def test_smooth_field_range_and_scale():
    f = fx.smooth_field(np.random.default_rng(0), (40, 50), 8.0)
    assert f.shape == (40, 50) and f.dtype == np.float32
    assert f.min() == 0.0 and f.max() == 1.0
    assert np.abs(np.diff(f, axis=1)).max() < 0.2  # smooth


def test_ou_series_is_positive_with_unit_mean():
    s = fx.ou_series(np.random.default_rng(0), 20000, 10.0)
    assert s.shape == (20000,) and (s > 0).all() and abs(s.mean() - 1.0) < 0.05


def test_lowpass_is_causal():
    x = np.zeros((1, 50), np.float32)
    x[0, 10] = 1.0
    y = fx.lowpass(x, 5.0)
    assert (y[0, :10] == 0).all() and y[0, 10] > 0 and y[0, 11] < y[0, 10] * 1.01 and y[0, 20] > 0


def test_markov_spikes_burst():
    s = fx.markov_spikes(np.random.default_rng(0), 50, 3000)
    assert s.shape == (50, 3000) and set(np.unique(s)) <= {0.0, 1.0}
    rate = s.mean()
    assert 0.002 < rate < 0.05
    # bursts: a spike is followed by another within 5 frames far more often than chance
    follow = (s[:, :-5].astype(bool) & (s[:, 1:-4] + s[:, 2:-3] + s[:, 3:-2] + s[:, 4:-1] + s[:, 5:] > 0)).sum()
    assert follow / max(s.sum(), 1) > 5 * rate * 5


def test_calcium_matches_legacy_recursion_without_rise():
    sp = np.zeros((2, 40), np.float32)
    sp[0, 3] = 1.0
    sp[1, 5] = 0.5
    c = fx.calcium(sp)
    decay = np.float32(np.exp(-1.0 / 15.0))
    assert c[0, 3] == 1.0 and c[0, 4] == decay and c[1, 5] == 0.5
    r = fx.calcium(sp, rise_tau=2.0)
    assert r[0, 3] < 0.5 and r[0].max() == 1.0 and r[0].argmax() > 3


def test_bleach_curve_endpoints():
    b = fx.bleach_curve(1000, 100.0)
    assert b[0] == 1.0 and 0.5 < b[-1] < 0.51


def test_vessel_mask_dark_curves():
    m = fx.vessel_mask(np.random.default_rng(0), (80, 80), 2)
    assert m.shape == (80, 80) and m.dtype == np.float32
    assert m.max() <= 1.0 and 0.25 < m.min() < 0.75
    assert 0.02 < (m < 0.9).mean() < 0.4
    assert np.array_equal(fx.vessel_mask(np.random.default_rng(0), (8, 8), 0), np.ones((8, 8), np.float32))


def test_warp_field_unit_peak_and_walk_amplitude():
    f = fx.warp_field(np.random.default_rng(0), (64, 64), 20.0)
    assert f.shape == (2, 64, 64) and abs(np.hypot(f[0], f[1]).max() - 1.0) < 1e-5
    a = fx.walk_amplitude(np.random.default_rng(0), 5000, 2.0)
    assert abs(np.quantile(np.abs(a), 0.99) - 2.0) < 1e-4


def test_warp_replicates_border():
    frame = np.full((32, 32), 7.0, np.float32)
    field = np.ones((2, 32, 32), np.float32)
    out = fx.warp(frame, field, 50.0)  # far beyond the frame
    assert out.shape == (32, 32) and np.isfinite(out).all() and (out == 7.0).all()
    assert np.array_equal(fx.warp(frame, field, 0.0), frame)


def test_frame_map_semantics():
    fm = fx.frame_map(np.random.default_rng(0), 100, dropped=5, duplicated=5)
    assert fm.shape == (100,) and fm[0] == 0 and (fm == -1).sum() == 5
    dup = [t for t in range(1, 100) if fm[t] != t and fm[t] != -1]
    assert len(dup) == 5 and all(fm[t] == fm[t - 1] for t in dup)
    assert np.array_equal(fx.frame_map(np.random.default_rng(0), 10, 0, 0), np.arange(10))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_synth.py -q`
Expected: ImportError on `synth_effects`.

- [ ] **Step 3: Write `src/minime/testing/synth_effects.py`**

```python
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
    b = np.exp(-1.0 / tau)
    y = np.empty_like(x, dtype=np.float32)
    acc = np.zeros(x.shape[:-1], np.float32)
    for t in range(x.shape[-1]):
        acc = acc * b + (1.0 - b) * x[..., t]
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


def vessel_mask(rng: Generator, shape: tuple[int, int], n: int, opacity: float = 0.7) -> np.ndarray:
    """Multiplicative transmission of ``n`` dark tortuous vessels (ones when ``n == 0``).

    Each vessel starts on a random edge heading inwards, walks in 2 px steps with heading
    noise (sd 0.15 rad) until it leaves the canvas, is 2 to 4 px wide and has a Beer-Lambert
    profile ``1 - opacity (1 - exp(-chord / width))`` blurred with sigma 1 px.
    """
    ch, cw = shape
    mask = np.ones(shape, np.float32)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    for _ in range(n):
        edge = rng.integers(4)
        u = rng.uniform(0.1, 0.9)
        y, x, theta = {
            0: (0.0, u * cw, np.pi / 2),  # top edge, heading down
            1: (ch - 1.0, u * cw, -np.pi / 2),
            2: (u * ch, 0.0, 0.0),  # left edge, heading right
            3: (u * ch, cw - 1.0, np.pi),
        }[int(edge)]
        theta += rng.uniform(-0.5, 0.5)
        width = rng.uniform(2.0, 4.0)
        r = width / 2
        chord = np.zeros(shape, np.float32)
        for _ in range(2 * (ch + cw)):
            if not (0 <= y < ch and 0 <= x < cw):
                break
            d2 = (yy - y) ** 2 + (xx - x) ** 2
            np.maximum(chord, 2 * np.sqrt(np.maximum(r * r - d2, 0)), out=chord)
            theta += rng.normal(0, 0.15)
            y += 2 * np.sin(theta)
            x += 2 * np.cos(theta)
        m = 1 - opacity * (1 - np.exp(-chord / width))
        mask *= cv2.GaussianBlur(m, (0, 0), 1.0)
    return mask


def warp_field(rng: Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """(2, h, w) float32 smooth (dy, dx) displacement field with peak magnitude 1."""
    f = np.stack([smooth_field(rng, shape, sigma) for _ in range(2)])
    f -= f.mean(axis=(1, 2), keepdims=True)
    mag = np.hypot(f[0], f[1]).max()
    return (f / mag).astype(np.float32) if mag > 0 else f


def walk_amplitude(rng: Generator, T: int, target: float) -> np.ndarray:
    """(T,) float32 random walk (step sd 0.1) scaled so the 99th percentile of |amp| is ``target``."""
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
    k = min(dropped + duplicated, T - 1)
    picked = rng.choice(np.arange(1, T), k, replace=False)
    drops, dups = picked[:dropped], picked[dropped:]
    fm[drops] = -1
    for t in np.sort(dups):
        fm[t] = fm[t - 1]
    return fm
```

- [ ] **Step 4: Run the block tests**

Run: `.venv/bin/python -m pytest tests/test_synth.py -q`
Expected: PASS. If `test_markov_spikes_burst` or `test_vessel_mask_dark_curves` fail on their statistical bounds, print the measured value and widen the bound to what the model gives, not the model to the bound.

- [ ] **Step 5: Commit**

```bash
git add src/minime/testing/synth_effects.py tests/test_synth.py
git commit -m "Add the pure building blocks for the synthetic 1p effects"
```

---

### Task 3: Options, extended truth and the shared renderer in `synth.py`

**Files:**
- Modify: `src/minime/testing/synth.py` (replace the "Synthetic movie" and "Ground truth" sections; keep `write_session` for Task 4)
- Modify: `tests/test_synth.py` (append)

**Interfaces:**
- Consumes: everything in `synth_effects.py` (Task 2).
- Produces:
  - `SynthOptions` frozen dataclass with the spec's fields and `to_dict()` / `from_dict(d)`
  - `SCENARIOS: dict[str, SynthOptions]`
  - `SynthSetup` with the new fields `options, sigma, amp, F0, spikes, first_active, group, patches, bleach, bg_fields, bg_time, neuropil_fields, neuropil_time, vessel_mask, warp_field, warp_amp, hemo_field, hemo_time, frame_map`
  - `synth_setup(h, w, T, seed=0, options=None)`, `synth_chunks(h, w, T, seed=0, chunk=GEN_CHUNK, options=None)`, `synth_movie(h, w, T, seed=0, options=None)`, `truth_chunks(setup, h, w, chunk=100)` (reads `setup.options`)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_synth.py`)

```python
ALL_ON = synth.SynthOptions(
    background_drift=15.0,
    neuropil=10.0,
    overlap=2,
    late_active=1,
    bleach_tau=100.0,
    vessels=2,
    motion_step=2,
    motion_margin=12,
    nonrigid=1.5,
    dropped=3,
    duplicated=3,
    size_cv=0.3,
    amp_cv=0.3,
    noise_sd=6.0,
    bursts=True,
    rise_tau=2.0,
    hemo=20.0,
    edge_band=3,
)


def test_options_round_trip_and_scenarios():
    d = ALL_ON.to_dict()
    assert synth.SynthOptions.from_dict(d) == ALL_ON
    assert synth.SynthOptions.from_dict(None) == synth.SynthOptions()
    assert synth.SCENARIOS["clean"] == synth.SynthOptions()
    assert set(synth.SCENARIOS) >= {"realistic", "overlap", "late_active", "vessels"}


def test_truth_matches_generator_with_every_effect():
    h, w, T = 48, 48, 90
    setup = synth.synth_setup(h, w, T, seed=3, options=ALL_ON)
    movie = synth.synth_movie(h, w, T, seed=3, options=ALL_ON)
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, h, w, chunk=7)])
    diff = movie.astype(float) - clean.astype(float)
    assert abs(diff.mean()) < 0.5 and diff.std() < 7.0
    assert setup.C.shape[0] == setup.ys.size == setup.spikes.shape[0]
    assert setup.F0.shape == setup.C.shape and setup.patches.shape[0] == setup.ys.size


@pytest.mark.parametrize(
    "opts",
    [
        dict(background_drift=15.0),
        dict(neuropil=10.0),
        dict(overlap=2),
        dict(late_active=1),
        dict(bleach_tau=100.0),
        dict(vessels=2),
        dict(nonrigid=1.5),
        dict(dropped=3, duplicated=3),
        dict(amp_cv=0.3),
        dict(bursts=True, rise_tau=2.0),
        dict(hemo=20.0),
        dict(edge_band=3),
    ],
)
def test_single_effect_keeps_cells_and_motion(opts):
    base = synth.synth_setup(64, 64, 60, seed=0)
    on = synth.synth_setup(64, 64, 60, seed=0, options=synth.SynthOptions(**opts))
    n = base.ys.size
    assert np.array_equal(synth.true_centroids(on)[:n], synth.true_centroids(base))
    assert np.array_equal(on.shifts, base.shifts)


def test_size_cv_keeps_positions_but_grows_patch():
    base = synth.synth_setup(64, 64, 30, seed=0)
    on = synth.synth_setup(64, 64, 30, seed=0, options=synth.SynthOptions(size_cv=0.3))
    assert on.sigma.size == base.ys.size and on.sigma.std() > 0
    assert on.patch.shape[0] >= 15 and on.patch.shape[0] % 2 == 1


def test_overlap_more_than_cells():
    s = synth.synth_setup(16, 16, 20, seed=0, options=synth.SynthOptions(overlap=5))
    assert s.ys.size == max(1, int(32 * 32 / 900)) + 5
    assert (s.group[-5:] >= 0).all()


def test_overlap_pairs_are_close_and_correlated():
    s = synth.synth_setup(64, 64, 600, seed=0, options=synth.SynthOptions(overlap=2))
    n = s.ys.size - 2
    for k in range(2):
        j = int(np.flatnonzero(s.group[:n] == k)[0])
        d = np.hypot(s.ys[n + k] - s.ys[j], s.xs[n + k] - s.xs[j])
        assert 4.0 <= d <= 10.0
        assert np.corrcoef(s.C[j], s.C[n + k])[0, 1] > 0.5


def test_late_active_clamped_and_first_active():
    s = synth.synth_setup(64, 64, 300, seed=0, options=synth.SynthOptions(late_active=99))
    assert (s.spikes[:, :210] == 0).all() and (s.first_active >= 210).all()
    base = synth.synth_setup(64, 64, 300, seed=0)
    has = base.spikes.any(axis=1)
    assert (base.first_active[has] < 300).all() and (base.first_active[~has] == 300).all()


def test_bleach_scales_baseline_and_cells():
    s = synth.synth_setup(64, 64, 200, seed=0, options=synth.SynthOptions(bleach_tau=50.0))
    assert np.allclose(s.F0[:, 0], 20.0) and np.allclose(s.F0[:, -1], 20.0 * s.bleach[-1])
    assert s.bleach[-1] < 0.52


def test_frame_map_matches_movie():
    h, w, T = 32, 32, 120
    o = synth.SynthOptions(dropped=5, duplicated=5)
    movie = np.concatenate([c for _, c in synth.synth_chunks(h, w, T, seed=1, chunk=7, options=o)])
    s = synth.synth_setup(h, w, T, seed=1, options=o)
    for t in range(T):
        src = s.frame_map[t]
        if src == -1:
            assert not movie[t].any()
        elif src != t:
            assert np.array_equal(movie[t], movie[t - 1])
    assert (s.frame_map == -1).sum() == 5
    cells = np.concatenate([c for _, _, c in synth.truth_chunks(s, h, w, chunk=100)])
    assert not cells[s.frame_map == -1].any()


def test_large_motion_clipped():
    s = synth.synth_setup(32, 32, 500, seed=0, options=synth.SynthOptions(motion_step=5, motion_margin=6))
    assert np.abs(s.shifts).max() <= 6 and s.margin == 6
    m = synth.synth_movie(32, 32, 20, seed=0, options=synth.SynthOptions(motion_step=5, motion_margin=6))
    assert m.shape == (20, 32, 32)


def test_vessels_darken_movie():
    base = synth.synth_movie(64, 64, 20, seed=0)
    on = synth.synth_movie(64, 64, 20, seed=0, options=synth.SynthOptions(vessels=2))
    assert on.mean() < base.mean() - 0.5


def test_edge_band_darkens_border():
    m = synth.synth_movie(64, 64, 5, seed=0, options=synth.SynthOptions(edge_band=4))
    assert m[:, :4].mean() < 0.4 * m[:, 10:-10, 10:-10].mean()
```

Also add `import pytest` at the top of `tests/test_synth.py`.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_synth.py -q`
Expected: AttributeError `SynthOptions`.

- [ ] **Step 3: Rewrite the generator and truth sections of `synth.py`**

Replace everything from the module docstring to just before `# Session writer` with:

```python
"""Synthetic miniscope session with ground truth.

Cells are Gaussian patches with spiking exp-decay traces on a smooth vignette background
with slow drift, Gaussian noise and integer random-walk rigid motion, uint8. Optional 1p
effects (``SynthOptions``: drifting background, neuropil, overlapping cells, late-active
cells, bleaching, vessels, large or non-rigid motion, dropped frames, cell variation,
bursting spikes, a non-calcium blob, an edge band) are off by default. With the defaults
the RNG call order (positions, traces, motion steps, then per-chunk noise) is the legacy
one, so a given ``(h, w, T, seed)`` reproduces the same movie as before the options
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

from minime.testing import synth_effects as fx

MOTION_MARGIN = 8  # px of canvas around the FOV for synthetic motion
GEN_CHUNK = 50  # frames generated at once (bounds memory)
BASELINE_F0 = 20.0  # grey levels; keeps cells visible
CELL_SIGMA = 3.0  # px
CELL_AMP = 60.0  # grey levels per unit transient
DECAY_TAU = 15.0  # frames (0.5 s at 30 fps)
PATCH_MIN = 15

# child RNG streams, one per effect
_S_SIZE, _S_BURST, _S_OVERLAP, _S_BG, _S_NP, _S_VESSEL, _S_WARP, _S_FRAMES, _S_HEMO = range(9)


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
    late_active: int = 0  # cells with no spikes before late_active_start * T
    late_active_start: float = 0.7
    bleach_tau: float | None = None  # frames; global envelope 0.5 + 0.5 exp(-t / tau)
    vessels: int = 0  # dark tortuous curves moving with the tissue
    motion_step: int = 1  # max |step| of the rigid random walk, px
    motion_margin: int = 8  # canvas margin, px; the walk is clipped to it
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> SynthOptions:
        return cls(**(d or {}))


SCENARIOS: dict[str, SynthOptions] = {
    "clean": SynthOptions(),
    "background": SynthOptions(background_drift=15.0),
    "neuropil": SynthOptions(neuropil=10.0),
    "overlap": SynthOptions(overlap=3),
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


def _gauss_patch(size: int = PATCH_MIN, sigma: float = CELL_SIGMA) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma**2))
    return (g / g.max()).astype(np.float32)


def _patch_size(sigma: np.ndarray) -> int:
    """Odd patch side holding ±2.5 sigma of the largest cell, at least the legacy 15."""
    return max(PATCH_MIN, int(np.ceil(5 * sigma.max())) | 1)


def _build(
    h: int, w: int, T: int, seed: int, options: SynthOptions | None
) -> tuple[SynthSetup, np.random.Generator]:
    """All draws before the frame loop; returns the truth and the RNG positioned for noise."""
    o = options or SynthOptions()
    rng = np.random.default_rng(seed)
    kids = np.random.SeedSequence(seed).spawn(9)

    def child(i: int) -> np.random.Generator:
        return np.random.default_rng(kids[i])

    m = o.motion_margin
    ch, cw = h + 2 * m, w + 2 * m
    n_base = max(1, int(ch * cw / 900))
    n = n_base + o.overlap
    r_size = child(_S_SIZE)
    sigma = CELL_SIGMA * fx.lognormal_factors(r_size, o.size_cv, n)
    amp_f = fx.lognormal_factors(r_size, o.amp_cv, n)
    ps = _patch_size(sigma)

    # legacy draws, in the legacy order: positions, spikes, spike amplitudes, motion steps
    ys = rng.integers(0, ch - ps, n_base)
    xs = rng.integers(0, cw - ps, n_base)
    spikes = (rng.random((n_base, T)) < 0.01).astype(np.float32)
    amp_u = rng.uniform(0.5, 1.5, (n_base, 1)).astype(np.float32)
    steps = rng.integers(-o.motion_step, o.motion_step + 1, (T, 2))

    if o.bursts:
        spikes = fx.markov_spikes(child(_S_BURST), n_base, T)
    if o.late_active > 0:
        k = min(o.late_active, n_base)
        spikes[n_base - k :, : int(o.late_active_start * T)] = 0.0
    group = np.full(n, -1)
    if o.overlap > 0:
        r = child(_S_OVERLAP)
        partners = r.choice(n_base, o.overlap, replace=o.overlap > n_base)
        dist = r.uniform(0.4, 0.8, o.overlap) * 4 * CELL_SIGMA
        ang = r.uniform(0, 2 * np.pi, o.overlap)
        oy = np.clip(np.rint(ys[partners] + dist * np.sin(ang)), 0, ch - ps).astype(ys.dtype)
        ox = np.clip(np.rint(xs[partners] + dist * np.cos(ang)), 0, cw - ps).astype(xs.dtype)
        own = (r.random((o.overlap, T)) < 0.01).astype(np.float32)
        ospk = o.overlap_corr * spikes[partners] + (1 - o.overlap_corr) * own
        ys, xs = np.concatenate([ys, oy]), np.concatenate([xs, ox])
        spikes = np.concatenate([spikes, ospk.astype(np.float32)])
        amp_u = np.concatenate([amp_u, r.uniform(0.5, 1.5, (o.overlap, 1)).astype(np.float32)])
        group[partners] = np.arange(o.overlap)
        group[n_base:] = np.arange(o.overlap)

    scale = (CELL_AMP * amp_f).astype(np.float32)  # exactly 60.0 per cell when amp_cv == 0
    amp = scale * amp_u[:, 0]
    transient = fx.calcium(spikes * amp_u, DECAY_TAU, o.rise_tau)  # legacy: traces * amp_u
    bleach = fx.bleach_curve(T, o.bleach_tau) if o.bleach_tau else np.ones(T)
    signal = transient * scale[:, None]  # (n, T) above F0
    C = ((np.float32(BASELINE_F0) + signal) * bleach.astype(np.float32)[None]).astype(np.float32)
    F0 = np.broadcast_to((BASELINE_F0 * bleach).astype(np.float32)[None], (n, T))
    has = spikes > 0
    first_active = np.where(has.any(axis=1), has.argmax(axis=1), T)

    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    vign = 40.0 + 30.0 * np.exp(
        -((yy - ch / 2) ** 2 + (xx - cw / 2) ** 2) / (2 * (0.4 * max(ch, cw)) ** 2)
    )
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * np.arange(T) / (30 * 60))
    shifts = np.clip(np.cumsum(steps, axis=0), -m, m)
    patches = np.stack([_gauss_patch(ps, s) for s in sigma])

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
        np_fields = np.stack(
            [fx.smooth_field(r, (ch, cw), 2 * float(np.median(sigma))) for _ in range(3)]
        )
        pop = fx.lowpass(signal.mean(axis=0, keepdims=True), 45.0)[0]
        pop = pop / pop.mean() if pop.mean() > 0 else np.ones(T, np.float32)
        np_time = np.stack([0.7 * pop + 0.3 * fx.ou_series(r, T, 300.0) for _ in range(3)])
    vessel_mask = (
        fx.vessel_mask(child(_S_VESSEL), (ch, cw), o.vessels)
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

    setup = SynthSetup(
        ys, xs, C, vign, drift, shifts, patches[0], m, o, sigma, amp, F0, spikes,
        first_active, group, patches, bleach, bg_fields, bg_time, np_fields, np_time,
        vessel_mask, warp_field, warp_amp, hemo_field, hemo_time, frame_map,
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
    canvas = s.vign[None] * (s.drift[t0:t1] * s.bleach[t0:t1])[:, None, None].astype(np.float32)
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
        canvas = canvas + noise
    cells = np.zeros((n, ch, cw), np.float32) if want_cells else None
    ps = s.patch.shape[0]
    for k in range(s.ys.size):
        sl = (slice(None), slice(s.ys[k], s.ys[k] + ps), slice(s.xs[k], s.xs[k] + ps))
        canvas[sl] += s.C[k, t0:t1, None, None] * s.patches[k][None]
        if cells is not None:
            cells[sl] += (s.C[k, t0:t1] - s.F0[k, t0:t1])[:, None, None] * s.patches[k][None]
    if o.vessels > 0:
        canvas *= s.vessel_mask[None]
        if cells is not None:
            cells *= s.vessel_mask[None]
    return canvas, cells


def _crop_frames(s: SynthSetup, canvas: np.ndarray, t0: int, h: int, w: int) -> np.ndarray:
    """Recorded (n, h, w) float32 frames: rigid crop, non-rigid warp, edge band."""
    o, m = s.options, s.margin
    out = np.empty((canvas.shape[0], h, w), np.float32)
    for i in range(canvas.shape[0]):
        dy, dx = s.shifts[t0 + i]
        fr = canvas[i, m + dy : m + dy + h, m + dx : m + dx + w]
        out[i] = fx.warp(fr, s.warp_field, float(s.warp_amp[t0 + i])) if o.nonrigid > 0 else fr
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
        noise = rng.standard_normal((t1 - t0, ch, cw), dtype=np.float32) * s.options.noise_sd
        canvas, _ = _render(s, t0, t1, noise, want_cells=False)
        frames, prev = _apply_frame_map(s, _crop_frames(s, canvas, t0, h, w), t0, prev)
        yield t0, np.clip(frames, 0, 255).astype(np.uint8)


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
        yield t0, np.clip(clean, 0, 255).astype(np.uint8), cell_out
```

Construct `SynthSetup` with keyword arguments in the real code (27 fields). Bit-identity of the default rests on: the legacy draws staying in their order on the main RNG, `scale` being exactly `60.0` when `amp_cv == 0`, no float64 promotion (`drift * bleach` is float64 ones, cast to float32 afterwards as before), and the noise being added before the cells.

- [ ] **Step 4: Run all synth tests, including the checksum**

Run: `.venv/bin/python -m pytest tests/test_synth.py -q`
Expected: PASS. If `test_default_session_is_byte_identical` fails, diff `synth_movie(32, 32, 20, 1)` against a copy of the old function (git show HEAD~2:src/minime/testing/synth.py) frame by frame to find the first differing op; the usual suspects are float64 promotion (`drift * bleach`, `amp` scaling) and adding noise after the cells.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q -x`
Expected: PASS (the fixture session is unchanged, so nothing else moves).

- [ ] **Step 6: Commit**

```bash
git add src/minime/testing/synth.py tests/test_synth.py
git commit -m "Add SynthOptions with per-effect 1p artefacts and extended ground truth to the synthetic session"
```

---

### Task 4: `write_session` options, `meta.json` round trip, CLI scenarios

**Files:**
- Modify: `src/minime/testing/synth.py` (`write_session`)
- Modify: `src/minime/cli.py:121-127` (`synth` command)
- Modify: `tests/test_synth.py` (append)

**Interfaces:**
- Produces: `write_session(out_dir, minutes, fps=30, segment=1000, seed=0, side=600, options=None) -> dict` with `meta["options"]`; `minime synth OUT --scenario NAME --side N`.

- [ ] **Step 1: Failing test**

```python
def test_write_session_stores_options(tmp_path):
    o = synth.SynthOptions(vessels=1, dropped=2)
    meta = synth.write_session(tmp_path / "raw", minutes=0.05, segment=30, seed=0, side=32, options=o)
    assert meta["options"] == o.to_dict()
    import json

    saved = json.loads((tmp_path / "meta.json").read_text())
    assert synth.SynthOptions.from_dict(saved["options"]) == o
    assert synth.SynthOptions.from_dict(synth.write_session(tmp_path / "raw2", 0.05, segment=30, side=32)["options"]) == synth.SynthOptions()
```

- [ ] **Step 2: Run, expect TypeError on `options`**

- [ ] **Step 3: Implement**

In `write_session` add `options: SynthOptions | None = None` after `side`, pass `options=options` to `synth_chunks`, and add `"options": (options or SynthOptions()).to_dict()` to `meta` after `"seed"`. In `cli.py`:

```python
@app.command()
def synth(
    out: Path,
    minutes: float = 10.0,
    fps: int = 30,
    segment: int = 1000,
    seed: int = 0,
    side: int = 600,
    scenario: str = "clean",
) -> None:
    """Write a synthetic Miniscope session (N.avi, GREY) and meta.json.

    SCENARIO is one of minime.testing.synth.SCENARIOS (clean, realistic, one per effect).
    """
    from minime.testing.synth import SCENARIOS, write_session

    if scenario not in SCENARIOS:
        raise typer.BadParameter(f"unknown scenario {scenario!r}; one of {sorted(SCENARIOS)}")
    typer.echo(write_session(out, minutes, fps, segment, seed, side, SCENARIOS[scenario]))
```

- [ ] **Step 4: Run `tests/test_synth.py` and `tests/test_cli*.py` if present; expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/minime/testing/synth.py src/minime/cli.py tests/test_synth.py
git commit -m "Store the synthetic options in meta.json and add --scenario and --side to minime synth"
```

---

### Task 5: Scorer reads the options, F0 and the frame map

**Files:**
- Modify: `bench/score.py` (`score_run`)
- Create: `tests/test_synth_effects.py` (first tests; the helper is reused in Task 6)

**Interfaces:**
- Consumes: `synth.SynthOptions.from_dict`, `setup.F0`, `setup.frame_map`.
- Produces: `score_run` unchanged signature; `_truth_traces(setup) -> tuple[np.ndarray, np.ndarray]` (recorded-order truth traces `(n, T)` and a `(T,)` bool mask of recorded frames that show a source frame).

- [ ] **Step 1: Failing tests** (`tests/test_synth_effects.py`)

```python
"""What the pipeline does with each optional 1p effect of the synthetic session.

One tiny pipeline run (300 frames, 64 x 64, ~9 s) per effect, scored against the truth.
``xfail`` marks document a current weakness; the reason names it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from minime import pipeline
from minime.params import Params
from minime.testing import synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import score  # noqa: E402

T, SIDE = 300, 64
# the tiny-session parameters of tests/conftest.py (copied: conftest is being reworked)
PARAMS = dict(
    tile=32,
    seeds_wnd=100,
    seeds_stp=50,
    seeds_max_wnd=8,
    template_frames=50,
    inita_wnd=8,
    dl_wnd=5,
    size_thres=(9, None),
    svd_k0=16,
    motion_before_tophat=False,
)


def _session(root: Path, options: synth.SynthOptions) -> Path:
    raw = root / "raw"
    synth.write_session(raw, minutes=T / 30 / 60, fps=30, segment=100, seed=0, side=SIDE, options=options)
    return raw


def _run(root: Path, options: synth.SynthOptions) -> tuple[dict, Path]:
    raw = _session(root, options)
    run = pipeline.run(raw, Params(**PARAMS), run_name="t", workers=1)
    return score.score_pipeline_dir(run), run


def _unit_centroids(run: Path) -> np.ndarray:
    meta = json.loads((run / "factors" / "meta.json").read_text())
    A = sparse.load_npz(run / "cnmf" / "A.npz")
    return score.footprint_centroids(A, int(meta["H"]), int(meta["W"]))


def test_score_truth_against_itself_with_dropped_frames(tmp_path):
    o = synth.SynthOptions(dropped=5, duplicated=5)
    raw = _session(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    truth, ok = score._truth_traces(s)
    assert truth.shape == (s.ys.size, T) and ok.sum() == T - 5
    cents = np.rint(synth.true_centroids(s)).astype(int)
    keep = np.flatnonzero((cents >= 0).all(axis=1) & (cents < SIDE).all(axis=1))  # inside the frame
    A = sparse.lil_matrix((keep.size, SIDE * SIDE), dtype=np.float32)
    for i, k in enumerate(keep):
        A[i, cents[k, 0] * SIDE + cents[k, 1]] = 1.0
    out = score.score_run(sparse.csr_matrix(A), truth[keep], SIDE, SIDE, raw.parent / "meta.json", s.shifts)
    assert out["n_matched"] == keep.size and out["trace_corr_median"] > 0.999


def test_score_meta_without_options(tmp_path):
    raw = _session(tmp_path, synth.SynthOptions())
    meta_path = raw.parent / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["options"]
    meta_path.write_text(json.dumps(meta))
    s = synth.synth_setup(SIDE, SIDE, T, 0)
    A = sparse.csr_matrix(np.zeros((1, SIDE * SIDE), np.float32))
    out = score.score_run(A, np.zeros((1, T), np.float32), SIDE, SIDE, meta_path, s.shifts)
    assert out["n_true"] == s.ys.size
```

- [ ] **Step 2: Run, expect AttributeError `_truth_traces`**

- [ ] **Step 3: Implement in `bench/score.py`**

Replace `BASELINE_F0 = 20.0` with a helper and use it in `score_run`:

```python
def _truth_traces(setup: synth.SynthSetup) -> tuple[np.ndarray, np.ndarray]:
    """Truth traces above F0 in recorded frame order, and which recorded frames show a source
    frame (dropped frames are black and excluded from the correlation)."""
    above = (setup.C - setup.F0).astype(np.float32)
    fm = setup.frame_map
    ok = fm >= 0
    out = np.zeros_like(above)
    out[:, ok] = above[:, fm[ok]]
    return out, ok
```

In `score_run`:

```python
    meta = json.loads(Path(meta_path).read_text())
    options = synth.SynthOptions.from_dict(meta.get("options"))
    setup = synth.synth_setup(meta["height"], meta["width"], meta["frames"], meta["seed"], options)
    ...
    truth, ok = _truth_traces(setup)
    corr = _trace_corr(traces[ui][:, ok], truth[ti][:, ok]) if len(ui) else np.array([np.nan])
```

Keep `BASELINE_F0` defined (other modules may import it) but unused by `score_run`.

- [ ] **Step 4: Run `tests/test_synth_effects.py tests/test_score.py`; expect PASS**

- [ ] **Step 5: Commit**

```bash
git add bench/score.py tests/test_synth_effects.py
git commit -m "Score against the stored synthetic options, the per-frame baseline and the frame map"
```

---

### Task 6: One pipeline test per effect

**Files:**
- Modify: `tests/test_synth_effects.py` (append)

**Interfaces:**
- Consumes: `_run`, `_unit_centroids`, `PARAMS` from Task 5; `synth.SCENARIOS`.

- [ ] **Step 1: Add the clean baseline fixture and the effect tests**

```python
@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("clean"), synth.SynthOptions())[0]


def test_clean_baseline(clean):
    assert clean["recall"] == 1.0 and clean["alignment_ok"]
    assert clean["trace_corr_median"] > 0.7


def test_background_drift(tmp_path, clean):
    out, _ = _run(tmp_path, synth.SCENARIOS["background"])
    assert out["recall"] == 1.0
    assert out["n_units"] <= clean["n_units"] + 2


def test_neuropil(tmp_path, clean):
    out, _ = _run(tmp_path, synth.SCENARIOS["neuropil"])
    assert out["recall"] == 1.0
    assert out["trace_corr_median"] > clean["trace_corr_median"] - 0.15


def test_overlapping_cells_are_split(tmp_path):
    o = synth.SCENARIOS["overlap"]
    out, run = _run(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    n = s.ys.size - o.overlap
    true_c = synth.true_centroids(s) + np.array(out["align_offset_px"])
    ui, ti = score.match_units(_unit_centroids(run), true_c)
    matched = set(ti.tolist())
    pairs_split = sum(
        int(j in matched and n + k in matched) for k in range(o.overlap) for j in np.flatnonzero(s.group[:n] == k)
    )
    assert pairs_split >= o.overlap - 1


def test_late_active_cell_is_found(tmp_path):
    o = synth.SCENARIOS["late_active"]
    out, run = _run(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    late = np.flatnonzero(s.first_active >= int(o.late_active_start * T))
    true_c = synth.true_centroids(s) + np.array(out["align_offset_px"])
    _, ti = score.match_units(_unit_centroids(run), true_c)
    assert set(late.tolist()) <= set(ti.tolist())


def test_bleaching(tmp_path, clean):
    out, _ = _run(tmp_path, synth.SynthOptions(bleach_tau=150.0))
    assert out["recall"] == 1.0
    assert out["trace_corr_median"] > clean["trace_corr_median"] - 0.15


def test_vessels(tmp_path, clean):
    o = synth.SCENARIOS["vessels"]
    out, run = _run(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    assert out["recall"] == 1.0
    m = s.margin
    inner = s.vessel_mask[m:-m, m:-m]
    uc = _unit_centroids(run)
    uc = uc[np.isfinite(uc).all(axis=1)]
    c = np.rint(uc - np.array(out["align_offset_px"])).astype(int)
    c = c[(c >= 0).all(axis=1) & (c < SIDE).all(axis=1)]
    assert (inner[c[:, 0], c[:, 1]] < 0.5).sum() == 0


def test_large_rigid_motion(tmp_path):
    o = synth.SCENARIOS["large_motion"]
    out, run = _run(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    est = np.load(run / "motion" / "arrays.npz")["shifts"]
    _, rmse, _ = score.motion_fit(np.asarray(est, float), s.shifts.astype(float))
    assert rmse < 0.5 and out["recall"] >= 0.9


def test_nonrigid_motion(tmp_path):
    out, _ = _run(tmp_path, synth.SCENARIOS["nonrigid"])
    assert out["recall"] >= 0.8


def test_dropped_and_duplicated_frames(tmp_path, clean):
    out, _ = _run(tmp_path, synth.SCENARIOS["dropped"])
    assert out["recall"] == 1.0
    assert out["n_units"] <= clean["n_units"] + 2
    assert out["trace_corr_median"] > clean["trace_corr_median"] - 0.15


def test_cell_size_and_brightness_variation(tmp_path):
    o = synth.SCENARIOS["cell_variation"]
    out, run = _run(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    true_c = synth.true_centroids(s) + np.array(out["align_offset_px"])
    _, ti = score.match_units(_unit_centroids(run), true_c)
    big_and_bright = np.flatnonzero((s.sigma >= 2.5) & (s.amp >= 30))
    assert set(big_and_bright.tolist()) <= set(ti.tolist())


def test_low_snr(tmp_path):
    out, _ = _run(tmp_path, synth.SCENARIOS["low_snr"])
    assert out["recall"] >= 0.8


def test_bursting_spikes_with_rise(tmp_path, clean):
    out, _ = _run(tmp_path, synth.SCENARIOS["bursts"])
    assert out["recall"] == 1.0
    assert out["trace_corr_median"] > clean["trace_corr_median"] - 0.15


def test_hemodynamic_blob_is_not_a_unit(tmp_path, clean):
    o = synth.SCENARIOS["hemo"]
    out, run = _run(tmp_path, o)
    s = synth.synth_setup(SIDE, SIDE, T, 0, o)
    assert out["recall"] == 1.0
    assert out["n_units"] <= clean["n_units"] + 2


def test_edge_band(tmp_path):
    out, _ = _run(tmp_path, synth.SCENARIOS["edge"])
    assert out["recall"] == 1.0


def test_realistic_scenario_floor(tmp_path):
    out, _ = _run(tmp_path, synth.SCENARIOS["realistic"])
    assert out["recall"] >= 0.7 and out["precision"] >= 0.3
    assert out["trace_corr_median"] > 0.5
```

- [ ] **Step 2: Run the module once in the background and read the failures**

Run: `.venv/bin/python -m pytest tests/test_synth_effects.py -q -p no:cacheprovider 2>&1 | tail -40` (about 3 minutes)

- [ ] **Step 3: For each failing test decide: bug or weakness**

A bug in the generator or the test (wrong index, wrong coordinate offset, a threshold that no sensible pipeline meets, for example the recall in `test_realistic_scenario_floor` set above what the clean run reaches) is fixed. A genuine weakness is marked, with the measured number in the reason, e.g.:

```python
@pytest.mark.xfail(reason="late-active cells are missed: the seed windows before frame 210 never see them (recall of the late cell 0/2 on 2026-10-27)", strict=True)
```

Use `strict=True` so the mark is removed when the pipeline improves. Record each xfail in the plan's commit message.

- [ ] **Step 4: Run the module again, expect PASS / XFAIL only**

- [ ] **Step 5: Commit**

```bash
git add tests/test_synth_effects.py
git commit -m "Add one pipeline test per synthetic 1p effect, xfail where the pipeline is weak"
```

---

### Task 7: Docs, lint, full suite

**Files:**
- Modify: `README.md` (new section before "## Out of scope")
- Modify: `TODO.md` (one line per xfail under "## Bugs" or "## Ideas")

- [ ] **Step 1: README section**

```markdown
## Synthetic data

`minime synth OUT --minutes 5 --side 600 --scenario realistic` writes a session with known
truth (`meta.json`); `minime score RUN` scores a run against it. Scenarios
(`minime.testing.synth.SCENARIOS`): `clean`, one per effect (`background`, `neuropil`,
`overlap`, `late_active`, `bleach`, `vessels`, `large_motion`, `nonrigid`, `dropped`,
`cell_variation`, `low_snr`, `bursts`, `hemo`, `edge`) and `realistic` (all tier-1 effects).
In Python, `SynthOptions` sets any combination. The truth (`synth_setup`) has per-cell size,
amplitude, spikes, first active frame and overlap group, the background and neuropil
components, the vessel mask, the warp, the bleaching envelope and the frame map.
`tests/test_synth_effects.py` shows what the pipeline does with each effect; `xfail` marks
there are known weaknesses.
```

- [ ] **Step 2: TODO.md** - add one bullet per `xfail` test under `## Bugs`, each naming the test and the measured number.

- [ ] **Step 3: Lint and format**

Run: `.venv/bin/ruff format src tests bench && .venv/bin/ruff check --fix src tests bench`
Expected: no remaining errors.

- [ ] **Step 4: Full suite in the background**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -5`
Expected: all pass, the new xfails listed as xfailed, total runtime up by about 3 minutes.

- [ ] **Step 5: Commit**

```bash
git add README.md TODO.md src tests bench
git commit -m "Document the synthetic scenarios and list the weaknesses the effect tests expose"
```
