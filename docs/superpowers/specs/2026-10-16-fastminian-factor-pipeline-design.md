# caanfast: end-to-end factor-first pipeline

Status: approved design, 2026-10-16. Working package name `caanfast` (final name
undecided; one search-and-replace renames it). Sibling repo of `minian_v2.0`; that repo's
`scripts/bench_*` directories hold every measured prototype this design builds on, and
its Obsidian note "Minian spatial update speedup" holds the measurements.

## 1. Goal

One local engine that turns a folder of Miniscope AVI segments into cell footprints and
traces, in two windows:

- **Window 1, heavy pass.** Preprocess, motion-correct and compress the movie once into
  per-tile low-rank factors plus a motion sidecar. Measured prototype: 45-min session
  in 8:07 on 1 core, 2:28 on 6 cores, factors 245 MB (`bench_browser`, `pre_rect_down2`).
- **Window 2, CNMF on factors.** Seeds, PNR, GMM, seed merge, initA, initC, unit merge,
  background, spatial update, traces, merge, two iterations. Never rereads the movie.

Done means: `caanfast run <raw_dir>` finishes unattended on a 10-min synthetic session
and on the 5-min session, writes `A, C, b, f` and a report, and its cells score at least
as well as original Minian against the synthetic ground truth. Out of scope: temporal
update (OASIS, later), `ks_refine` (dropped), browser UI, PyPI shipping, non-rigid motion,
real-data parameter tuning.

## 2. Repository

```
caanfast/
  pyproject.toml            # hatchling; deps: numpy scipy numba opencv-python-headless
                            # scikit-learn pandas typer; dev: pytest ruff
  src/caanfast/
    io.py                   # AVI segment reader, raw-folder listing, content hash
    tiles.py                # tile grid, pack/unpack, tile store (part files)
    preprocess.py           # median denoise, 31x31 square tophat
    motion.py               # Hann-windowed template registration at 1/2 res, warp, sidecar
    pass1.py                # frame-major pass: pre -> motion -> warp -> block maxima -> tile store
    compress.py             # per-tile streamed adaptive rSVD, rho rank cut
    factors.py              # Factors container: load/save, per-tile algebra helpers
    pass2.py                # tile-major pass: store -> factors (+ parallel driver)
    seeds.py                # seeds from block maxima, PNR + GMM on factors
    corr.py                 # low-passed factor Grams; seed merge, initA, unit merge
    init.py                 # initA scatter, initC (Cholesky), unit_merge
    cnmf.py                 # background b,f; spatial update (numba NNQP); traces; loop
    spatial_kernel.py       # numba Gram-shared non-negative coordinate descent
    pipeline.py             # run(raw_dir, params) -> processed/<name>/...
    params.py               # dataclass of all parameters, JSON load/save, defaults
    cli.py                  # typer: run, score
    testing/synth.py        # synthetic session generator + ground truth (from heavy_pass)
  tests/                    # pytest, tiny synthetic data, each file < 10 s
  bench/
    score.py                # units vs truth: recall, precision, F1, trace corr
    run_minian_5min.md      # how the Minian reference numbers were produced
  docs/superpowers/{specs,plans}/
  README.md                 # inputs -> outputs, one command
```

Own `uv` venv (`.venv`), Python 3.11. Minian is not a dependency. Every module gets type
hints, `ruff format` and `ruff check`. Raw data is read-only; everything derived goes to
`processed/<run_name>/` next to the raw folder (`<raw_dir>/../processed/`).

## 3. Data contracts

**Input.** A folder of `N.avi` grayscale uint8 segments (Miniscope DAQ layout, sorted by
integer stem). Optional `meta.json` next to the folder with `frames, height, width, fps,
seed` as `gen_dataset.py` writes it; only `bench/score.py` needs it.

**Output `processed/<run_name>/`:**

| path | content |
|---|---|
| `params.json` | every parameter used, package version, raw content hash |
| `motion/meta.json`, `motion/arrays.npz` | sidecar: `shifts (T,2) float32` (dy, dx, correction convention of the prototype), `template (h,w) float32`, `quality (T,) float32` (Pearson to template after correction), params, raw hash. Format of `bench_motion/motion_sidecar.py`, version 1. |
| `factors/meta.json` | `H, W, T, tile, n_tiles, rank per tile, format_version=1, timings` |
| `factors/tile_NNN.npz` | `W (P_b, r) float32` = U·s, `V (T, r) float32`, `mean (P_b,) float32`, `sigma (P_b,) float32` noise sd per pixel, `energy (P_b,) float64` = ||y − mean||² per pixel. Tile order and geometry as `tiles.tile_grid`, pixels row-major inside the tile. |
| `factors/max_proj.npy` | `(H, W) uint8` global max projection of the corrected movie |
| `factors/block_max.npy`, `factors/block_edges.npy` | `(n_blocks, H, W) uint8` block maxima at the seed-window edges and the `(n_blocks+1,)` frame edges |
| `cnmf/A.npz` | scipy CSR `(K, H*W) float32` footprints |
| `cnmf/C.npy`, `cnmf/YrA.npy` | `(K, T) float32`; `C` is the trace fed to the last spatial update, `YrA` the final residual trace |
| `cnmf/b.npy`, `cnmf/f.npy` | `(H, W) float32`, `(T,) float32` |
| `cnmf/units.csv` | unit_id, centroid_h, centroid_w, size_px, peak |
| `report.json` | wall seconds per stage, counts (seeds at each filter, units at each stage), bytes written, peak RSS |

The tile store (`<run>/tiles/tile_NNN.partPP`, uint8) is temporary and deleted after pass 2
unless `--keep-store`.

## 4. Window 1

Everything here is the measured `pre_rect_down2` candidate plus the motion fix, moved
out of the bench scripts with the same numerics.

**Template.** First 200 frames of segment 0: denoise, tophat, mean; align those frames to
it once and average again (`candidates/motion.py:make_template`, `down=2`).

**Pass 1** (`pass1.py`), per AVI segment, in sub-batches of 100 frames:

1. `cv2.medianBlur` k=7;
2. tophat with a 31x31 square kernel (`cv2.morphologyEx`, `np.ones`);
3. rigid shift vs template: frames and template mean-removed, Hann-windowed, area
   downsampled 2x, cross-correlation by rfft2, peak search limited to ±20 px, parabolic
   sub-pixel, shifts scaled back by 2 (`candidates/motion.py:estimate`);
4. warp `cv2.warpAffine` INTER_LINEAR with the negated shift (prototype convention);
5. running global max projection, and block maxima: the seed windows (`wnd_size`,
   `stp_size`) define frame edges as in `bench_passes/candidates/seeds_blockmax.py`; the
   pass keeps one `(H, W)` max per block, updated per sub-batch;
6. pack the corrected frames into 64x64 tiles and append each tile to its part file.

Parallel: N worker processes, each single-threaded (BLAS, OpenCV pinned), each owning a
contiguous range of segments and writing part files `tile_NNN.partPP`; the driver merges
shifts, max projections and block maxima. Quality per frame = Pearson correlation of the
warped frame to the template (computed on the 2x-downsampled frame to keep it cheap). The
sidecar is written at the end of pass 1. Reruns with the same raw hash and motion params
load the sidecar and skip estimation (apply only).

**Pass 2** (`pass2.py`, `compress.py`), per tile, all frames, N workers over tiles:

1. read the tile's part files into one `(T, P_b)` uint8 array;
2. per-pixel mean (float64 accumulation) and noise `sigma` = Minian's `noise_fft`
   (logmexp of the periodogram over (noise_freq, 0.5), all frames, 128-pixel chunks);
3. streamed randomized SVD of the mean-removed tile (Halko, oversampling 10, 1 power
   iteration, float32 chunks of 4096 frames, BLAS `sgemm`), start rank k=40;
4. keep component i if lag-1 autocorrelation of V_i > max(4/√T, 0.2); if any of the last
   10 components passed, double k (cap: min(100, P_b−1, T−1)) and repeat;
5. write `W = U[:, keep]·s[keep]`, `V[:, keep]`, mean, sigma.

Tiles do not overlap (no halo), as in the measured prototype.

## 5. Window 2: every stage on factors

Notation per tile b: `Y_b ≈ W_b V_bᵀ + mean_b` (P_b x T), `r_b` ≤ ~20. Reconstructing a tile
chunk `W_b V_b[t0:t1]ᵀ` costs 2·P_b·r_b·Tc flops; for the whole 10-min movie that is
~1e11 flops, about one second, so any stage may stream over reconstructed chunks when it
needs a nonlinearity (clip, max, percentile). Nothing is ever written back as a movie.

**Seeds** (`seeds.py`). Rolling windows (`wnd_size` 1000, `stp_size` 500, Minian notebook
defaults) are maxima over the stored block maxima; local maxima per window with the
cv2 dilate/erode union over `ksize` 2..`max_wnd`−1, `diff_thres`, connected components,
per-component coordinate median (vectorised, from `seeds_blockmax.py`). Seeds = pixels
that are a local maximum in at least one window, with their count.

**PNR on factors.** Seed trace `y_p = w_p V_bᵀ + mean_p` (denoised). Minian's PNR divides
the raw peak-to-peak by the high-passed peak-to-peak; on factors the high-pass energy is
gone, so define
`pnr_p = ptp_q(y_p) / (ptp_q of white noise with sd sigma_p) = ptp_q(y_p) / (6.18·sigma_p)`
for q = (0.1, 99.9). Threshold: 2-component GMM on log pnr, keep the upper component
(`thres="auto"`), fixed threshold as an option. **GMM refine** on ptp_q(y_p) exactly as
Minian's `gmm_refine` (2 components, keep upper, mean mask). Both filters are in; `ks_refine`
is not.

**Correlations** (`corr.py`). Low-pass each tile's V once: brick-wall rfft mask, DC and bins
≥ `int(noise_freq·T)` zeroed (`noise_freq` 0.06). For tiles a, b: `G_ab = V_lp,aᵀ V_lp,b`
(r_a x r_b). `corr(i, j) = (w_i G_ab w_jᵀ) / sqrt(n_i n_j)` with
`n_i = w_i G_aa w_iᵀ + f_lp·resid_i`, `resid_i = ||y_i − mean_i||² − ||w_i||²` the energy
the compression discarded (assumed white), `f_lp = 2(nr−1)/T`. This is
`compressed_svd.py` with `NORM_MODE="resid"`; it brings a noise-only pixel back to ~0
correlation. `||y_i − mean_i||²` is the `energy` array pass 2 stores per tile.

- **seeds_merge:** seed pairs within `thres_dist` 10 px and corr > `thres_corr` 0.8 form
  connected groups; keep the seed with the largest `max_proj` value per group.
- **initA:** for each seed, corr to every pixel within `wnd` 10 px (crossing tile borders
  through `G_ab`), zero below `thres_corr` 0.8, that image is the footprint. Empty
  footprints are dropped.
- **unit_merge:** units whose footprints overlap (`A Aᵀ > 0`) and whose low-passed `C`
  correlate > 0.9 merge transitively; merged A and C are means. Plain K x K algebra.

**initC** (`init.py`). `C = (AᵀA)⁻¹ AᵀY`, `AᵀY = Σ_b (A_bᵀ W_b) V_bᵀ + (A_bᵀ mean_b) 1ᵀ`,
Cholesky with `pinv` fallback above cond 1e12 (`initc_normal.py`). A as CSR.

**Noise for the spatial penalty.** Minian uses `sn` from the PSD in (noise_freq, 0.5).
Here `sn := sigma` from pass 2, the same estimate (0.75 x the sd for white noise). A
median of integer frame differences was quantized on uint8 frames and left pixels at 0.

**Background** (`cnmf.py`). Exact Minian semantics with the clip:
`b = mean_t clip(Y − A C, 0)`, `f_t = clip(Y_t − A C_t, 0)·b / (b·b)`. Two streamed passes
over reconstructed tile chunks (`bg_streamed.py`), the first skipped when `b` from the
previous iteration is reused. Nothing written.

**Spatial update** (`cnmf.py`, `spatial_kernel.py`). `sub` = each footprint dilated by
`dl_wnd` 10 px (disk). `Cc` = C minus its row means, `G = Cc Ccᵀ` (K x K),
`B = Y Ccᵀ = Σ_b W_b (V_bᵀ Ccᵀ)` per tile (the mean term vanishes because Cc rows sum to
zero), `lam = T · sparse_penal · sn_p`. Per pixel a non-negative quadratic program over its
allowed units solved by cyclic coordinate descent in numba with a duality-gap stop
(`bench_spatial/candidates/numba_cd.py`, float32 B). Then Minian's post-processing:
normalise each footprint by its maximum (Minian `norm_fac = A.max`), scale C by the same factor, drop units smaller than
`size_thres` (25, None) px. Background footprint is not updated (`update_background=False`,
notebook default).

**Traces.** `YrA = C + A_norm((Y − b fᵀ)ᵀA − Cᵀ AᵀA)` clipped at 0, with `YᵀA` from the
factors as in initC and `A_norm = diag(1/||a_k||²)` (`compute_trace` semantics,
`trace_shared.py`).

**Loop.** init → `update_background` → for 2 iterations: `update_spatial` →
`update_background` → traces → `C := YrA` → `unit_merge`. Without a temporal update
the clipped residual trace stands in for the deconvolved `C`; this is the documented
substitute until OASIS lands.

## 6. Parameters (`params.py`)

All in one dataclass with the notebook defaults used for the Minian 5-min reference:
`median_ksize 7, tophat_wnd 15, motion_max_shift 20, motion_down 2, tile 64,
svd_k0 40, svd_power_iter 1, rank_rho_min 0.2, rank_z 4, seeds_wnd 1000, seeds_stp 500,
seeds_max_wnd 15, seeds_diff_thres 3, noise_freq 0.06, pnr_thres "auto", merge_dist 10,
merge_corr 0.8, inita_corr 0.8, inita_wnd 10, unit_merge_corr 0.8 (init) / 0.9 (loop),
dl_wnd 10, sparse_penal 0.01, size_thres (25, None), n_iter 2, workers 1`.
`caanfast run --params p.json` overrides; the resolved set is written to `params.json`.

## 7. Correctness gates

Unit tests per module on 64x64 x 500-frame synthetic data (generator in
`testing/synth.py`, a port of `heavy_pass.synth_chunks` + `synth_truth.py`):

| module | test |
|---|---|
| io, tiles | round trip AVI → frames; pack/unpack tiles bit-identical |
| preprocess | tophat square kernel equals `cv2` reference; shapes, dtype uint8 |
| motion | RMSE vs true shifts < 0.15 px on a 300-frame clip; sidecar round trip and hash miss |
| pass1 + compress | factors reconstruct the noise-free movie with rel err ≤ 0.30 and cell capture ≥ 0.97 (the 5-min prototype got 0.145 / 0.991); block maxima equal `np.max` over the same frames |
| seeds | seeds from block maxima equal seeds from the full projections (same windows) |
| corr | factor correlation vs raw Pearson of low-passed noise-free traces: median abs diff < 0.05 on true-cell pixels |
| init | initC equals `np.linalg.lstsq` on the reconstructed movie to 1e-5 |
| spatial_kernel | objective within 1e-6 relative of a scipy `nnls` per pixel; zeros outside `sub` |
| cnmf background, traces | equal to a dense numpy reference on the reconstructed movie to 1e-5 |
| pipeline | end to end on the tiny session: finishes, finds ≥ 80 % of the planted cells |

End-to-end gate (`bench/score.py`, same script for both pipelines): true cells from
`synth_setup` (centroid = position + 7, shifted into template coordinates by the fitted
motion offset); a unit matches a true cell when its footprint centroid is within 5 px;
one-to-one greedy matching by distance. Report recall, precision, F1, median Pearson of
`YrA` (ours) or `C` (Minian) against the true cell trace, wall time, peak disk written.
**Pass:** F1 ≥ Minian's F1 on the 5-min session and median trace correlation ≥ Minian's,
both minus 0.02 tolerance; runs unattended on the 10-min session. Minian's own A, C come
from `run_minian_baseline.py --keep` (re-run started 2026-10-16; `A_new`, `C_new` in its
intermediate zarr). Targets, not gates: 5-min total ≤ 160 s on 6 workers (Minian: 471 s
to the same point); 10-min ≤ 6 min on 6 workers; peak disk ≤ raw size + 10 %.

## 8. Build process

Governing session dispatches one implementer agent per module with: this spec, the
module's prototype files, the contract, and the test it must pass (TDD). A reviewer
agent checks each against the prototype numerics before merge. Model choice: Sonnet for
ports of gated kernels (io, tiles, preprocess, motion, compress, spatial_kernel, seeds,
init), Opus for the stages that were only ever timed as algebra (corr, cnmf, pipeline
loop, score), Haiku for scaffolding, lint and README. Integration loop: run the 5-min
session, triage, dispatch fixes, repeat until section 7 passes; then the 10-min session;
then one smoke run on the real `WL25_demo` session (no truth; must finish and yield a
plausible unit count). Commits go to branch `feat/factor-pipeline`, never pushed.

## 9. Risks and their handling

- **Thresholds calibrated for raw Pearson** (`inita_corr`, `merge_corr`). The "resid"
  normalisation approximates raw Pearson; if F1 misses the gate, sweep `inita_corr` in
  {0.6, 0.7, 0.8, 0.9} on the 5-min set, pick by F1, and record that the default was
  fitted on synthetic data.
- **PNR scale differs from Minian.** GMM auto-threshold is the default for that reason.
- **Dim cells below the rank cut are gone by construction.** Cell capture 0.991 on the
  prototype says this is a 1 % effect; the end-to-end recall gate catches it if not.
- **Tile borders.** No halo; footprints crossing a border are assembled from two tiles'
  factors through `G_ab`. If border cells show lower recall, add a 10-px fit halo in pass 2
  (keep only core rows of U) as a follow-up.
- **numba inside worker processes.** Window 2 runs in one process with numba `prange`;
  pass 1 and 2 workers use no numba.
