# cafast

Factor-first CNMF pipeline for one-photon miniscope calcium imaging. The movie is read
twice (pass 1: preprocess + motion correction; pass 2: per-tile low-rank compression); every
CNMF stage after that runs on the stored factors and never rereads the movie.

## Install

    uv venv -p 3.11 && uv pip install -e ".[dev]"
    .venv/bin/pytest

## Run

    .venv/bin/cafast run RAW_DIR --workers 6 --name NAME [--params p.json] [--keep-store]
    .venv/bin/cafast score RAW_DIR/../processed/NAME     # synthetic sessions only

**Input:** `RAW_DIR` = a folder of `N.avi` grayscale uint8 segments (Miniscope DAQ layout,
sorted by integer stem). Read-only. A single long AVI works if it is named `0.avi` (a
symlink is enough), but it is then processed by one pass-1 worker.

**Output:** `RAW_DIR/../processed/NAME/`

| path | content |
|---|---|
| `params.json` | every parameter used, raw content hash, package version |
| `motion/` | shifts (T, 2), template, per-frame quality; reused on a rerun with the same raw data and motion params |
| `factors/` | per-tile factors `W, V, mean, sigma, energy`, max projection, block maxima |
| `cnmf/A.npz` | footprints, scipy CSR (K, H*W) float32, row maximum 1 |
| `cnmf/C.npy`, `cnmf/YrA.npy` | (K, T) float32: trace fed to the last spatial update; final residual trace |
| `cnmf/b.npy`, `cnmf/f.npy` | background footprint (H, W) and trace (T,) |
| `cnmf/units.csv` | unit_id, centroid_h, centroid_w, size_px, peak |
| `report.json` | wall seconds per stage, seed/unit counts per stage, bytes, peak RSS |

The temporary tile store (`tiles/`, same size as the raw data) is deleted after pass 2.

## Parameters

`src/cafast/params.py` holds every parameter with its default (Minian notebook values).
To change some, write a JSON file with only those keys and pass `--params`:

    {"pnr_thres": 2.5, "gmm_refine": true, "workers": 6}

Defaults that differ from the Minian notebook, with the measurement behind each, are in
`bench/README.md`. The only one changed so far: `gmm_refine` is off (Minian's reference run
also skipped it). All defaults were checked on synthetic data only.

## Results (synthetic, 600 x 600 px, 421 true cells, M1 Max)

| session | pipeline | workers | wall | units | recall | precision | F1 | trace corr (median) |
|---|---|---|---|---|---|---|---|---|
| 5 min (9000 fr) | Minian 2.0.2 | 4 x 2 thr | 471 s | 508 | 0.914 | 0.758 | 0.829 | 0.964 |
| 5 min | cafast | 6 | 40.1 s | 404 | 0.933 | 0.973 | **0.953** | **0.995** |
| 10 min (18000 fr) | cafast | 6 | 68.5 s | 400 | 0.924 | 0.973 | 0.948 | 0.995 |
| 10 min | cafast | 1 | 147.3 s | 400 | 0.924 | 0.973 | 0.948 | 0.995 |

Gate (spec section 7): F1 >= 0.809 and trace corr >= 0.944 on the 5-min session: passed.
Peak disk = raw size (tile store) + factors (52 MB / 93 MB). Details and per-stage times:
`bench/README.md`.

## Out of scope

- Temporal update (OASIS deconvolution). `C` is the clipped residual trace (`YrA`) instead.
- `ks_refine` (dropped), non-rigid motion, browser UI, PyPI packaging.
- Real-data parameter tuning. No real session has been run yet (scheduled separately).

## Known differences from Minian

- Seeds, correlations and the spatial update use the denoised factor reconstruction, not
  the raw movie.
- Spatial penalty noise: `sn := sigma` (1.4826 x MAD of first differences / sqrt 2). Minian's
  default `noise_fft(..., "logmexp")` gives 0.75 x sigma for white noise, so our penalty is
  1.33x Minian's. On the 5-min session scaling `sn` by 0.75 changed no score (F1 0.9527 both).
- Footprints carry a long tail of tiny weights: median 1029 px > 0 vs Minian 509 px, but
  329 px > 1 % of the maximum in both, and 5.7 % (ours) vs 5.5 % (Minian) of the footprint
  mass lies outside the true 15 x 15 cell patch.
