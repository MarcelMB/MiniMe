<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MarcelMB/MiniMe/main/docs/assets/minime-logo-dark-1200.png">
    <img alt="MiniMe" src="https://raw.githubusercontent.com/MarcelMB/MiniMe/main/docs/assets/minime-logo-1200.png" width="420">
  </picture>
</p>

# MiniMe (Miniscope Matrix-factorization Engine)

Formerly CaFast / CaAnFast; runs written under the old name still load.

Factor-first CNMF pipeline for one-photon miniscope calcium imaging. The movie is read
twice (pass 1: preprocess + motion correction; pass 2: per-tile low-rank compression); every
CNMF stage after that runs on the stored factors and never rereads the movie.

## Install
Released versions: `pip install minime-cnmf` (the import and the command are `minime`).


    uv venv -p 3.11 && uv pip install -e ".[dev,tune]"
    .venv/bin/pytest

## Run

    .venv/bin/minime run RAW_DIR --workers 6 --name NAME [--params p.json] [--keep-store] [--overwrite]
    .venv/bin/minime score RAW_DIR/../processed/NAME     # synthetic sessions only

**Input:** `RAW_DIR` = a folder of `N.avi` grayscale uint8 segments (Miniscope DAQ layout,
sorted by integer stem). Read-only. A single long AVI works if it is named `0.avi` (a
symlink is enough). Pass-1 workers stream each segment in 1000-frame batches, so worker
memory does not grow with the file (about 3 x 1000 frames, ~1.1 GB at 600 x 600). A single
file is still processed by one pass-1 worker, and the last segment (here: the whole file) is
decoded once more at startup to count its frames.

**Output:** `RAW_DIR/../processed/NAME/`

| path | content |
|---|---|
| `params.json` | every parameter used, raw content hash, package version |
| `motion/` | shifts (T, 2), template (the prepared half-resolution image, about H/2 x W/2), per-frame quality; reused on a rerun with the same raw data and motion params |
| `factors/` | per-tile factors `W, V, mean, sigma, energy`, max projection, block maxima |
| `cnmf/A.npz` | footprints, scipy CSR (K, H*W) float32; row maximum 1, except units merged after the last spatial update (mean of the merged rows, maximum can be below 1) |
| `cnmf/C.npy`, `cnmf/YrA.npy` | (K, T) float32: trace fed to the last spatial update; final residual trace |
| `cnmf/b.npy`, `cnmf/f.npy` | background footprint (H, W) and trace (T,) |
| `cnmf/units.csv` | unit_id, centroid_h, centroid_w, size_px, peak |
| `report.json` | wall seconds per stage, seed/unit counts per stage, bytes, peak RSS |

The temporary tile store (`tiles/`, same size as the raw data) is deleted after pass 2.

## Tuning window 2

Seeds, initialisation and CNMF run on the saved factors, so their parameters can be changed
without rereading the movie (WL25, 200 x 200 px, 39,822 frames: 16 s instead of 49 s).

    .venv/bin/minime cnmf RUN_DIR [--params p.json] [--name NAME]
    .venv/bin/minime tune RUN_DIR [--minian minian_units.npz] [--port 8765]

**`minime cnmf`**: `RUN_DIR` = a finished run. `--params` may hold only the keys to change;
the rest come from `RUN_DIR/params.json`. Output: `RUN_DIR/reruns/NAME/` with `params.json`,
`report.json` and `cnmf/` as in a full run, `stages/` (seed tables after init / PNR / merge,
footprints after initA, init merge and every spatial update and merge) and symlinks
`factors`, `motion` to the source run, so the tools that read a run work on it.
Parameters that shaped the factors (`median_ksize`, `tophat_wnd`, `motion_*`,
`template_frames`, `tile`, `svd_*`, `rank_*`, `noise_freq`) are refused; `seeds_wnd` / `seeds_stp` only
if their windows fall on the block edges pass 1 stored. Same parameters give the same result
as the full run (tested).

**`minime tune`**: a page on `http://127.0.0.1:8765/` (standard library server, matplotlib
PNGs; needs the `tune` extra). On start it reruns the run's own parameters as `base`. The
form holds the window-2 parameters; *Rerun window 2* writes `reruns/tNNN/` and draws its
figures: seeds after init / PNR / merge with the PNR histogram, initA and init merge
footprints, footprints after each spatial update and merge (red: more than one piece above
10 % of peak), size and fragmentation histograms, example traces, and with `--minian` the
final overlay (Minian npz with `A`, `C`, `max_proj`; aligned by phase correlation and
matched by centroid). Shared figures: max projection, motion, per-pixel noise and mean.
The run list compares every rerun (changed parameters, counts, median size, units in pieces,
Minian matches); *view* / *cmp* stack two reruns' figures. Click a units-table row for that
unit's footprint and trace. *Save params* writes the full set to `RUN_DIR/tune/FILE.json` for
`minime run --params`. *Pass-1 preview* shows raw / median / registration image / tophat on
the first 300 frames for `median_ksize`, `tophat_wnd`, `motion_bandpass` (these need a full
`minime run`).

### Auto-tune

`minime autotune RUN_DIR --diameter D` (or "Propose parameters" on `minime tune`) sweeps the
window-2 parameters stage by stage (~5 min on a 30-min session), scores each setting with
label-free unit checks (trace PNR, one piece, cell size from D, no duplicates; score = good -
flagged) and writes the proposal to `RUN_DIR/reruns/auto/` with `sweep.csv` (the map),
`unit_flags.csv` and `autotune.json`. D is the cell diameter in px of the factor frame.
The grids are built around the run's own values (thresholds x 0.5 ... 2, correlations
+-0.1, `pnr_thres` also "auto") and extended outward while the best value sits at an edge,
so any run's values are inside its grids. The run's own parameters are the first setting
tried; the proposal is never worse than them (the page and the CLI say when the search found
nothing better). Size parameters (`seeds_max_wnd`, `inita_wnd`, `dl_wnd`, `merge_dist`,
`size_thres`) are derived from D only where the run still uses the built-in default. The
init-level scores only prune within a step: the best init settings of every distinct
`merge_corr` go through CNMF before the choice. `bench/autotune_synth.py` checks the search
on synthetic sessions with ground truth (proposal vs. a default and a detuned start).

## Browser viewer

    minime serve [FOLDER]

starts a local server on 127.0.0.1 and opens the browser. It lists the runs under FOLDER
(default: your home folder), shows every stage of a run, and starts new runs. Runs are only
read; results it has to compute for older runs go to `~/.cache/minime/` (or
`$MINIME_CACHE`). On a remote machine, start it with `--no-browser --port 8765` and open
an SSH tunnel: `ssh -L 8765:127.0.0.1:8765 host`, then the printed URL (a different local
port works too, e.g. `ssh -L 9000:127.0.0.1:8765 host` and the URL with `:9000`). Stopping
the server (Ctrl+C) also stops a run it started; only one server runs per cache folder.

### Building the front end

The browser app lives in `web/` (React + TypeScript, Vite). `minime serve` serves the
build in `src/minime/serve/static/`, which is not in git:

    cd web && npm install && npm run build

During development run the back end and Vite's dev server side by side:

    minime serve FOLDER --no-browser --port 8765 --token dev
    cd web && npm run dev      # then open http://localhost:5173/?token=dev

Checks: `npm run typecheck`, `npm test` (unit tests), `npm run e2e` (smoke test in Chromium
against a tiny synthetic run; run `npx playwright install chromium` once first; set `MINIME_PYTHON` if the venv is not `../.venv`). The e2e run has 300 frames, so it exercises the full-resolution trace path, not the overview bands.

## Parameters

`src/minime/params.py` holds every parameter with its default (Minian notebook values).
To change some, write a JSON file with only those keys and pass `--params`:

    {"pnr_thres": 2.5, "gmm_refine": true, "workers": 6}

A run's own `params.json` can be passed back to `--params` to reproduce it (its `raw_dir`,
`raw_hash` and `minime_version` entries are ignored; `--workers` still overrides).

Defaults that differ from the Minian notebook: `gmm_refine` is off (Minian's reference run also skipped
it), and `motion_before_tophat` is on: shifts are estimated on a band-passed copy of the
denoised frame, then the frame is warped and tophat-filtered (on a real session the tophat
removed the vessels registration needs). Other defaults were checked on synthetic data only.

Minian's `sparse_penal` is `sparse_snr` here, with `sparse_snr = sparse_penal * sqrt(T)` for
the session length `T` it was tuned on (0.01 on a 10,000-frame session is `sparse_snr: 1`,
the default); a params file with `sparse_penal` is rejected with that hint. Unlike Minian's,
the penalty does not grow with the length of the recording (see "Known differences").

`median_ksize: 1` / `tophat_wnd: 0` skip that step, for input preprocessed upstream.

## Results (synthetic, 600 x 600 px, 421 true cells, M1 Max)

| session | pipeline | workers | wall | units | recall | precision | F1 | trace corr (median) |
|---|---|---|---|---|---|---|---|---|
| 5 min (9000 fr) | Minian 2.0.2 | 4 x 2 thr | 471 s | 508 | 0.914 | 0.758 | 0.829 | 0.964 |
| 5 min | minime | 6 | 43.4 s | 404 | 0.933 | 0.973 | **0.953** | **0.995** |
| 5 min | minime, `fix/rank-vs-T` (sparse_snr, tree seed merge) | 6 | 46.1 s | 407 | 0.948 | 0.980 | **0.964** | 0.994 |
| 10 min (18000 fr) | minime* | 6 | 68.5 s | 400 | 0.924 | 0.973 | 0.948 | 0.995 |
| 10 min | minime* | 1 | 147.3 s | 400 | 0.924 | 0.973 | 0.948 | 0.995 |

\* 10-min rows: commit 16f6ac8, before the background footprint `b` was recomputed on the
last CNMF iteration (that adds one background pass, +3.8 s on 5 min); not rerun.

Gate (spec section 7): F1 >= 0.809 and trace corr >= 0.944 on the 5-min session: passed.
Peak disk = raw size (tile store) + factors (52 MB / 93 MB).

## Synthetic data

`minime synth OUT --minutes 5 --side 600 --scenario realistic` writes a session with known
truth (`meta.json`); `minime score RUN` scores a run against it. Scenarios
(`minime.testing.synth.SCENARIOS`): `clean`, one per effect (`background`, `neuropil`,
`overlap`, `late_active`, `bleach`, `vessels`, `large_motion`, `nonrigid`, `dropped`,
`cell_variation`, `low_snr`, `bursts`, `hemo`, `edge`) and `realistic` (all tier-1 effects).
In Python, `SynthOptions` sets any combination; the defaults reproduce the legacy session
byte for byte. The truth (`synth_setup`) has per-cell size, amplitude, spikes, first active
frame and overlap group, the background and neuropil components, the vessel mask, the warp,
the bleaching envelope and the frame map; `detectable` marks the cells a pipeline can be
expected to find. `tests/test_synth_effects.py` shows what the pipeline does with each
effect; its `xfail` marks are known weaknesses. The effect models follow
[minisim](https://github.com/miniscope/minisim).

## Out of scope

- Temporal update (OASIS deconvolution). `C` is the clipped residual trace (`YrA`) instead.
- `ks_refine` (dropped), non-rigid motion, browser UI, PyPI packaging.
- Real-data parameter tuning (one real session run so far).

## Known differences from Minian

- Seeds, correlations and the spatial update use the denoised factor reconstruction, not
  the raw movie.
- Noise `sigma` (spatial penalty and PNR): Minian's `noise_fft(..., "logmexp")` over
  (noise_freq, 0.5), computed in pass 2 on the uint8 tile, so it is 0.75 x the white-noise sd.
  It replaced 1.4826 x MAD of first differences / sqrt 2, which on uint8 tophat frames was
  quantized (WL25: 0 / 1.05 / 2.10, 10 % of pixels 0 and so unpenalized). PNR divides by
  `6.18 sigma`, so it reads 1.33x the white-noise-calibrated value (WL25: 1459 -> 1639 seeds).
- Background: rank-1 fit of the unclipped residual `Y - A C` on the factors
  (`b = clip(mean_t (Y - A C), 0)`), not Minian's mean of `clip(Y - A C, 0)`. `b` differs by
  up to 6 % of its maximum; scores and traces do not change (worst unit trace corr 0.998) and
  `cnmf` is ~2x faster. `background_clip: true` restores Minian's definition.
- Footprints carry a long tail of tiny weights: median 1029 px > 0 vs Minian 509 px, but
  329 px > 1 % of the maximum in both, and 5.7 % (ours) vs 5.5 % (Minian) of the footprint
  mass lies outside the true 15 x 15 cell patch.
- Spatial penalty: `lam = sqrt(T) * sparse_snr * sn` per pixel (on unit-norm traces), so a
  pixel keeps a unit where its std along the unit's trace exceeds `sparse_snr * sn`, whatever
  `T`. Minian's `LassoLars(alpha = sparse_penal * sn)` amounts to `lam = T * sparse_penal * sn`,
  a threshold that grows as `sqrt(T)`: on a 2 h session it dropped 149 of 632 units and
  shrank footprints to 30 px. Minian's value converts as `sparse_snr = sparse_penal * sqrt(T)`.
- `seeds_merge` joins each seed to its brightest correlated neighbour (correlation above
  `merge_corr` within `merge_dist`, brighter than itself) and keeps the roots, instead of
  Minian's connected components of the same links. With 280 rolling windows a 2 h session
  seeds almost every pixel of a cell halo (48k seeds); the components then run through the
  dim pixels between cells and merged 32k seeds into 166. The tree rule stops at the valley
  between two cells: 490 units on that session, F1 0.959 -> 0.964 on the 5-min synthetic
  session.
- Per-tile rank cap 300 (was 100): a 64 x 64 tile dense with cells passed 176 components
  of the autocorrelation cut at 140k frames (124 at 10k), and the cap at 100 left 76 of
  them (capture < 0.5) out of the factors.

## Releasing

Bump `version` in `pyproject.toml`, commit, then tag and push:

    git tag v0.0.1 && git push origin v0.0.1

`.github/workflows/release.yml` builds the web front end and the wheel, publishes to
TestPyPI, and then to PyPI once the `pypi` environment is approved on GitHub. Both use
Trusted Publishing, so no tokens are stored. `test.yml` runs ruff and the tests on every push.

## License

GPL-3.0-or-later. Parts of the algorithms are ported from
[Minian](https://github.com/denisecailab/minian) (GPL-3.0).
