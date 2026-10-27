# Richer synthetic one-photon data for the tests

Date: 2026-10-27 · Branch `main` · Generator `src/minime/testing/synth.py`

## Goal

The test suite runs on one synthetic session: Gaussian cells, exponential transients, rigid
random-walk motion, Gaussian noise. That is clean enough that the pipeline's real failure
modes (drifting out-of-focus background, neuropil, overlapping cells, cells silent for most
of a long recording, bleaching, vessels, large or non-rigid motion, dropped frames, dim or
small cells) are never exercised, and labelled one-photon data to test against is scarce.

This spec extends the generator with one toggle per effect, each with exact ground truth,
so a test can switch on a single effect and show what the pipeline does with it. The ideas
and formulas come from minisim (github.com/miniscope/minisim, GPL-3), a physical forward
model of 1p miniscope data; the code does not (minisim is a 17 k-line package with xarray,
zarr and pydantic, and cannot reproduce the current default session).

Decided with the user:

- Priorities. Tier 1: drifting background, neuropil, overlapping correlated cells,
  late-active cells, photobleaching, vessels. Tier 2: large and non-rigid motion, dropped and
  duplicated frames, per-cell size and brightness variation. Tier 3, only because they are a
  few lines each: a non-calcium (hemodynamic-like) signal and a dark edge band.
- No WL27-specific artefacts (LED/gain steps, the striped oval): the WL27 recording is no
  longer a target.
- Approach: an options layer on the current generator (not minisim as a dependency, not a
  rewrite as a step pipeline). The default session stays byte-identical, so the ~100 tests
  on the 300-frame fixture are untouched.

## 1. API

```python
@dataclass(frozen=True)
class SynthOptions:            # every default means "off"
    background_drift: float = 0.0   # grey levels
    neuropil: float = 0.0           # grey levels
    overlap: int = 0                # extra cells placed on top of existing ones
    overlap_corr: float = 0.8       # share of the partner's spikes they inherit
    late_active: int = 0            # cells with no spikes before late_active_start * T
    late_active_start: float = 0.7
    bleach_tau: float | None = None # frames
    vessels: int = 0
    motion_step: int = 1            # max |step| of the rigid random walk, px
    motion_margin: int = 8          # canvas margin, px; the walk is clipped to it
    nonrigid: float = 0.0           # px, 99th percentile of the warp amplitude
    dropped: int = 0                # frames replaced by a black frame
    duplicated: int = 0             # frames replaced by the previous frame
    size_cv: float = 0.0            # lognormal CV of the cell sigma
    amp_cv: float = 0.0             # lognormal CV of the transient amplitude
    noise_sd: float = 4.0           # grey levels
    bursts: bool = False            # Markov-gated bursting spikes
    rise_tau: float = 0.0           # frames; 0 = instantaneous rise
    hemo: float = 0.0               # grey levels, one broad blob at 0.1 Hz
    edge_band: int = 0              # px of darkened border
```

`synth_movie`, `synth_chunks`, `synth_setup`, `truth_chunks` and `write_session` take
`options: SynthOptions | None = None`. `SCENARIOS: dict[str, SynthOptions]` holds "clean"
(defaults), "realistic" (all tier 1 at moderate strength plus size and amplitude CV and
bursts) and one entry per effect; `minime synth` gets `--scenario` and `--side`.

Reproducibility: the legacy draws (positions, traces, motion steps, per-chunk noise) keep
their order on the main `default_rng(seed)`. Every effect draws from its own child stream
(`np.random.SeedSequence(seed).spawn(n)`, one fixed index per effect), so enabling an effect
does not move the cells or change the noise of the others, and "same seed, effect on vs off"
is a valid comparison.

## 2. Ground truth

`SynthSetup` gains fields with defaults (existing positional construction keeps working):

| Field | Shape | Meaning |
|---|---|---|
| `sigma`, `amp` | (n,) | per-cell Gaussian sigma (px) and transient amplitude (grey) |
| `F0` | (n, T) | baseline fluorescence per cell and frame (20 × bleach envelope) |
| `spikes` | (n, T) | spike amplitudes before the calcium kernel |
| `first_active` | (n,) | first frame with a spike, T if never |
| `group` | (n,) | overlap group id, -1 for cells without partner |
| `bleach` | (T,) | global envelope B(t), ones when off |
| `bg_fields`, `bg_time` | (3, ch, cw), (3, T) | drifting background components |
| `neuropil_fields`, `neuropil_time` | (3, ch, cw), (3, T) | neuropil components |
| `vessel_mask` | (ch, cw) | multiplicative transmission, ones when off |
| `warp_field`, `warp_amp` | (2, h, w), (T,) | non-rigid displacement (unit peak) and its per-frame amplitude in px |
| `hemo_field`, `hemo_time` | (ch, cw), (T,) | the non-calcium blob |
| `frame_map` | (T,) | source frame shown by each recorded frame, -1 for a dropped (black) frame |
| `options` | | the `SynthOptions` used |

`C` keeps its meaning: the emitted fluorescence of each cell per source frame, baseline
included, bleaching applied, so `truth_chunks` stays exact. Trace scoring uses `C - F0`
instead of `C - 20`. `true_centroids` is unchanged (centres in unshifted frame coordinates).

`meta.json` stores `options` (as a dict) next to `frames`, `height`, `width`, `seed`;
`synth_setup(h, w, T, seed, options)` rebuilds the identical truth from it.

One function `_render(setup, t0, t1) -> (canvas, cells)` produces the noise-free canvas
and the cells-only signal for both the generator and `truth_chunks`, so the two cannot
drift apart. The generator adds noise to the canvas; both then apply the same frame
transform (rigid crop, non-rigid warp, edge band, frame map, uint8 clip).

## 3. Effect models

Units: grey levels on the uint8 frame (cells peak at ~60 above a baseline of 20, noise
sd 4, vignette 40 to 70), frames at 30 fps. "Smooth random field" = white noise filtered
with a Gaussian of the given sigma, shifted to be non-negative, peak-normalised. "OU" =
Ornstein-Uhlenbeck process with correlation time tau, unit variance, exponentiated as
`exp(0.4·x − 0.08)` (mean 1) as in minisim.

Canvas composition per source frame t, before noise and before the frame transform:

```
canvas = vign·drift(t)·B(t)
       + background_drift · Σ_k bg_fields[k]·bg_time[k, t] / 3
       + neuropil         · Σ_k neuropil_fields[k]·neuropil_time[k, t] / 3
       + hemo             · hemo_field·hemo_time[t]
       + Σ_i C[i, t]·patch_i                       (C includes F0 and B(t))
canvas *= vessel_mask
```

- **Background drift**: 3 fields with sigma = max(ch, cw)/4, each with an OU time course,
  tau = 10 s (300 frames). Models out-of-focus tissue moving through the focal volume.
- **Neuropil**: 3 fields with sigma = 2 × the median cell sigma; time course
  `0.7·P(t) + 0.3·OU(t)`, where `P` is the population activity (mean of `C − F0` over
  cells) low-passed causally with tau = 1.5 s and mean-normalised; OU tau = 10 s.
- **Overlap**: `overlap` extra cells, each 0.4 to 0.8 cell diameters (random direction) from
  a distinct existing cell, inside the canvas. Spike train = partner's spikes × `overlap_corr`
  plus its own spikes × (1 − `overlap_corr`). Both members get the same `group` id.
- **Late-active**: the last `late_active` cells of the (non-overlap) population have their
  spikes zeroed before `late_active_start · T`. `first_active` is computed for every cell.
- **Bleaching**: `B(t) = 0.5 + 0.5·exp(−t / bleach_tau)`, applied to every cell's `C`
  (baseline and transient) and to the vignette. Global, not per cell (minisim's per-cell
  activity-driven ODE is more than the tests need).
- **Vessels**: `vessels` dark curves. Each starts on a random canvas edge, walks in steps
  of 2 px with heading noise sd 0.15 rad until it leaves the canvas; width 2 to 4 px;
  chord-length Beer-Lambert mask `M = 1 − 0.7·(1 − exp(−L / w))`, blurred with sigma 1 px.
  Masks multiply. Applied before motion so vessels move with the tissue, like the cells.
- **Rigid motion**: steps `integers(−motion_step, motion_step + 1)` clipped to
  `±motion_margin`, as today for the defaults.
- **Non-rigid motion**: one displacement field of two smooth random fields (sigma = h/3),
  normalised to unit peak magnitude, times a per-frame amplitude that is a random walk
  (step sd 0.1 px) scaled so its 99th percentile equals `nonrigid` px. Applied with
  `cv2.remap` (bilinear, border replicate) after the rigid crop. The truth frames are warped
  the same way.
- **Dropped and duplicated frames**: `dropped` random frames (never frame 0) are written as
  black; `duplicated` random frames (never frame 0) repeat the previous recorded frame.
  `frame_map[t]` is the source frame, −1 for black. Cells' `C` stays indexed by source frame.
- **Size and brightness variation**: per-cell sigma and amplitude are lognormal with mean
  equal to the default (3 px, 60 grey) and the given CV. The patch size follows the largest
  sigma (`ceil(5·sigma_max)`, odd). `F0` stays 20 for every cell.
- **Noise**: Gaussian with sd `noise_sd` (4 today), the same draw order as now.
- **Spikes**: default Bernoulli(0.01) per frame with amplitude U(0.5, 1.5), as today. With
  `bursts`, a two-state Markov gate per cell (quiescent to active 0.005, active to
  quiescent 0.3 per frame; rate 0.3 per frame when active, 0.003 when quiescent). Kernel
  `exp(−t/15)` as today; with `rise_tau > 0` the double exponential
  `exp(−t/15) − exp(−t/rise_tau)`, peak-normalised.
- **Hemodynamic-like signal**: one smooth field (sigma = max(ch, cw)/3) times
  `0.5·(1 + sin(2π·0.1 Hz·t))`.
- **Edge band**: the outer `edge_band` px of each recorded frame multiplied by 0.2 (after the
  warp, so it is fixed on the sensor like a lens edge).

## 4. Scorer and CLI

`bench/score.py`: build the setup from `meta["options"]`; truth traces are
`setup.C − setup.F0` evaluated through `frame_map` (dropped frames excluded from the
correlation); everything else unchanged. `minime synth OUT --scenario realistic --side 600`.

## 5. Tests

Generator (`tests/test_synth.py`, additions only):

- the default 300-frame 64 × 64 session equals the current output (array checksum fixed
  when the refactor lands, computed from the pre-refactor generator);
- with every effect on, `truth_chunks` matches the noise-free movie to within the noise;
- enabling any single effect leaves `true_centroids` of the original cells unchanged;
- `write_session` with options round-trips them through `meta.json`;
- `frame_map` matches the movie (black frames, repeated frames) and `first_active` matches
  `spikes`.

Pipeline (`tests/test_synth_effects.py`, new): a helper writes a 300-frame 64 × 64 session
with the given options, runs the tiny pipeline (`TINY_PARAMS` from `conftest.py`, ~9 s) and
scores it with `bench/score.py`. One test per effect asserts a concrete behaviour; which
tests pass and which are `xfail` is decided by running the pipeline, and an `xfail` reason
names the weakness. Intended assertions:

| Effect | Assertion |
|---|---|
| background drift | recall 1.0, no units in cell-free areas |
| neuropil | recall 1.0, trace correlation not worse than clean |
| overlap | both cells of a pair matched to distinct units |
| late-active | the late cell is matched |
| bleaching | trace correlation of matched cells ≥ clean |
| vessels | recall unchanged, no unit centred on the vessel |
| large rigid motion | shift RMSE < 0.5 px |
| non-rigid motion | recall ≥ 0.8 |
| dropped / duplicated | no units beyond the clean count, trace correlation over the frame map |
| size and amplitude CV | recall of cells above the size threshold 1.0; small or dim cells reported |
| bursts and rise | trace correlation ≥ clean |
| hemo | no unit at the blob centre |
| edge band | recall unchanged |
| realistic | f1 and trace correlation reported as a regression floor |

Not in this spec: changing pipeline behaviour for any weakness the tests expose, minisim
as a bench dependency, timestamps files or non-AVI formats.
