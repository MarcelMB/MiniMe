# cafast

Factor-first CNMF pipeline for one-photon miniscope calcium imaging. The movie is read
twice (pass 1: preprocess + motion correction; pass 2: per-tile low-rank compression); every
CNMF stage after that runs on the stored factors and never rereads the movie.

## Install

    uv venv -p 3.11 && uv pip install -e ".[dev]"    # add ,bench for cafast tune
    .venv/bin/pytest

## Run

    .venv/bin/cafast run RAW_DIR --workers 6 --name NAME [--params p.json] [--keep-store]
    .venv/bin/cafast score RAW_DIR/../processed/NAME     # synthetic sessions only

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

    .venv/bin/cafast cnmf RUN_DIR [--params p.json] [--name NAME]
    .venv/bin/cafast tune RUN_DIR [--minian minian_units.npz] [--port 8765]

**`cafast cnmf`**: `RUN_DIR` = a finished run. `--params` may hold only the keys to change;
the rest come from `RUN_DIR/params.json`. Output: `RUN_DIR/reruns/NAME/` with `params.json`,
`report.json` and `cnmf/` as in a full run, `stages/` (seed tables after init / PNR / merge,
footprints after initA, init merge and every spatial update and merge) and symlinks
`factors`, `motion` to the source run, so `bench/compare_minian.py` reads it like a run.
Parameters that shaped the factors (`median_ksize`, `tophat_wnd`, `motion_*`,
`template_frames`, `tile`, `svd_*`, `rank_*`) are refused; `seeds_wnd` / `seeds_stp` only
if their windows fall on the block edges pass 1 stored. Same parameters give the same result
as the full run (tested).

**`cafast tune`**: a page on `http://127.0.0.1:8765/` (standard library server, matplotlib
PNGs; needs the `bench` extra). On start it reruns the run's own parameters as `base`. The
form holds the window-2 parameters; *Rerun window 2* writes `reruns/tNNN/` and draws its
figures: seeds after init / PNR / merge with the PNR histogram, initA and init merge
footprints, footprints after each spatial update and merge (red: more than one piece above
10 % of peak), size and fragmentation histograms, example traces, and with `--minian` the
final overlay (Minian npz with `A`, `C`, `max_proj`; aligned and matched as in
`bench/compare_minian.py`). Shared figures: max projection, motion, per-pixel noise and mean.
The run list compares every rerun (changed parameters, counts, median size, units in pieces,
Minian matches); *view* / *cmp* stack two reruns' figures. Click a units-table row for that
unit's footprint and trace. *Save params* writes the full set to `RUN_DIR/tune/FILE.json` for
`cafast run --params`. *Pass-1 preview* shows raw / median / registration image / tophat on
the first 300 frames for `median_ksize`, `tophat_wnd`, `motion_bandpass` (these need a full
`cafast run`).

## Parameters

`src/cafast/params.py` holds every parameter with its default (Minian notebook values).
To change some, write a JSON file with only those keys and pass `--params`:

    {"pnr_thres": 2.5, "gmm_refine": true, "workers": 6}

A run's own `params.json` can be passed back to `--params` to reproduce it (its `raw_dir`,
`raw_hash` and `cafast_version` entries are ignored; `--workers` still overrides).

Defaults that differ from the Minian notebook, with the measurement behind each, are in
`bench/README.md`. Changed so far: `gmm_refine` is off (Minian's reference run also skipped
it), and `motion_before_tophat` is on: shifts are estimated on a band-passed copy of the
denoised frame, then the frame is warped and tophat-filtered (on a real session the tophat
removed the vessels registration needs). Other defaults were checked on synthetic data only.

`median_ksize: 1` / `tophat_wnd: 0` skip that step, for input preprocessed upstream.

## Results (synthetic, 600 x 600 px, 421 true cells, M1 Max)

| session | pipeline | workers | wall | units | recall | precision | F1 | trace corr (median) |
|---|---|---|---|---|---|---|---|---|
| 5 min (9000 fr) | Minian 2.0.2 | 4 x 2 thr | 471 s | 508 | 0.914 | 0.758 | 0.829 | 0.964 |
| 5 min | cafast | 6 | 43.4 s | 404 | 0.933 | 0.973 | **0.953** | **0.995** |
| 10 min (18000 fr) | cafast* | 6 | 68.5 s | 400 | 0.924 | 0.973 | 0.948 | 0.995 |
| 10 min | cafast* | 1 | 147.3 s | 400 | 0.924 | 0.973 | 0.948 | 0.995 |

\* 10-min rows: commit 16f6ac8, before the background footprint `b` was recomputed on the
last CNMF iteration (that adds one background pass, +3.8 s on 5 min); not rerun.

Gate (spec section 7): F1 >= 0.809 and trace corr >= 0.944 on the 5-min session: passed.
Peak disk = raw size (tile store) + factors (52 MB / 93 MB). Details and per-stage times:
`bench/README.md`.

## Out of scope

- Temporal update (OASIS deconvolution). `C` is the clipped residual trace (`YrA`) instead.
- `ks_refine` (dropped), non-rigid motion, browser UI, PyPI packaging.
- Real-data parameter tuning. One real session run so far (WL25, `bench/README.md`).

## Known differences from Minian

- Seeds, correlations and the spatial update use the denoised factor reconstruction, not
  the raw movie.
- Noise `sigma` (spatial penalty and PNR): Minian's `noise_fft(..., "logmexp")` over
  (noise_freq, 0.5), computed in pass 2 on the uint8 tile, so it is 0.75 x the white-noise sd.
  It replaced 1.4826 x MAD of first differences / sqrt 2, which on uint8 tophat frames was
  quantized (WL25: 0 / 1.05 / 2.10, 10 % of pixels 0 and so unpenalized). PNR divides by
  `6.18 sigma`, so it reads 1.33x the white-noise-calibrated value (WL25: 1459 -> 1639 seeds).
- Footprints carry a long tail of tiny weights: median 1029 px > 0 vs Minian 509 px, but
  329 px > 1 % of the maximum in both, and 5.7 % (ours) vs 5.5 % (Minian) of the footprint
  mass lies outside the true 15 x 15 cell patch.
