# bench: scoring against truth and the Minian reference

Inputs -> outputs: a pipeline's units (footprints `A`, traces, motion `shifts`) -> `score.json`
(recall, precision, F1, trace correlation) against the ground truth of the synthetic session
(`cafast.testing.synth`, dataset meta in `<raw>/../meta.json`).

## Scoring method (`bench/score.py`)
- Footprint centroids are intensity-weighted. Units are matched one-to-one to true centroids,
  greedy by increasing distance, max 5 px. `recall = matched / n_true`,
  `precision = matched / n_units`.
- Units live in template coordinates, true centroids in unshifted-frame coordinates. The offset
  is fitted from `motion_fit(pipeline_shifts, true_shifts)` (`est = c + sign * true`) and added
  to the true centroids as `sign * c`; if the median matched distance is >= 1.5 px, the opposite
  sign is tried and the better one kept. `offset_sign_used` is the multiplier on `c` that was
  applied (`align_offset_px` = the offset actually added).
- `trace_corr_*` = Pearson(`traces[k]`, `C_true[j] - 20`) over matched pairs.

## Minian reference (synth_v4_5min, 9000 frames, 600 x 600)
Run: `scripts/bench_browser/run_minian_baseline.py --keep` in `minian_v2.0` (original Minian
v2.0.2, `pipeline.ipynb` defaults, ks_refine and temporal update skipped; 4 workers x 2 threads,
4 GB each). The final `update_background_2` stage was killed after the `A_new` / `C_new` save.
Full earlier run: **471 s, 24.9 GB peak disk** (`results/BASELINE_5min.md`).

Caveat (A_new recompute): `run_minian_baseline.py` runs
`A_new = save_minian(A_new.rename("A_new"), intpath, overwrite=True)` on the normalised
`A_new` that `update_spatial` returns. That array lazily reads `A_new.zarr`, which the save
overwrites, so the stored `A_new` is all NaN (`C_new` is fine). The reference numbers below
come from a recompute with `bench/minian_redo_spatial.py`: `update_spatial(Y_hw_chk, A, C,
sn_spatial, dl_wnd=10, sparse_penal=0.01, size_thres=(25, None))` with `Y_hw_chk`, `A`, `C`,
`sn_spatial` read from the kept intermediates (105 s, 508 of 508 units kept, unit ids identical
to `C_new`). The result is un-normalised, which does not change centroids. The kept
`minian_intermediate/` was deleted afterwards, so reproducing needs a full Minian re-run
(about 8 min, 25 GB disk) followed by the recompute. Open these zarr stores with
`consolidated=False` (stale consolidated metadata after rechunker).

Minian's accuracy (508 units, 421 true cells):

| recall | precision | F1 | trace corr median | trace corr q25 | median match dist |
|---|---|---|---|---|---|
| 0.914 | 0.758 | **0.829** | **0.964** | 0.942 | 0.14 px |

Offset between frame and template coordinates: (-7.47, -7.88) px, sign +1. Traces are Minian's
`C_new` (trace after the spatial update).

## Commands
Order matters: baseline `--keep` -> redo spatial -> export -> score.
```bash
MINIAN_DIR=/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/minian_v2.0
MINIAN_PY=$MINIAN_DIR/.venv/bin/python
OUT=/Users/mbrosch/minian_testing_local/synth_v4_5min/processed/minian_baseline_keep
RAW=/Users/mbrosch/minian_testing_local/synth_v4_5min/raw
SCRATCH=/tmp/minian_respatial        # receives the recomputed A_new.zarr
# 1. Minian baseline keeping intermediates (its stored A_new is NaN, see above)
$MINIAN_PY $MINIAN_DIR/scripts/bench_browser/run_minian_baseline.py --keep
# 2. recompute A_new from the kept intermediates (read-only on them)
$MINIAN_PY bench/minian_redo_spatial.py $OUT --scratch $SCRATCH
# 3. export (A_new from SCRATCH, C_new and motion from the run) -> $OUT/minian_units.npz
$MINIAN_PY bench/minian_export.py $OUT A_new C_new $SCRATCH
# 4. score under our venv -> $OUT/score.json
.venv/bin/cafast score --minian $OUT/minian_units.npz --raw $RAW
# score a cafast run (--raw only needed if RUN_DIR/params.json is missing)
.venv/bin/cafast score RUN_DIR [--raw $RAW]                         # -> RUN_DIR/score.json
```
`bench/minian_export.py` refuses to export if `A_new` contains NaN. Score dicts carry
`alignment_ok`: False (plus a stderr warning) when neither offset sign gives a median matched
distance below 1.5 px, i.e. the motion offset could not be verified.

## cafast runs (Task 12, commit 16f6ac8, M1 Max, 10 cores)

Fresh runs (no motion sidecar reused), default params, scored with `cafast score`.

| session | workers | wall | pass 1 | pass 2 | window 2 (cnmf) | units | recall | precision | F1 | trace corr median / q25 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 min, Minian 2.0.2 | 4 x 2 thr | 471 s | | | | 508 | 0.914 | 0.758 | 0.829 | 0.964 / 0.942 |
| 5 min | 6 | 40.1 s | 11.1 s | 6.4 s | 21.5 s (20.1 s) | 404 | 0.933 | 0.973 | **0.953** | **0.995** / 0.991 |
| 10 min | 6 | 68.5 s | 17.2 s | 11.1 s | 38.6 s (35.5 s) | 400 | 0.924 | 0.973 | 0.948 | 0.995 / 0.989 |
| 10 min | 1 | 147.3 s | 65.4 s | 42.2 s | 38.4 s (35.6 s) | 400 | 0.924 | 0.973 | 0.948 | 0.995 / 0.989 |

Rerun after the final-review fixes (background `b` recomputed on the last CNMF iteration,
streamed pass 1, decoded frame counts, NaN guards), fresh, 6 workers:

| session | workers | wall | pass 1 | pass 2 | window 2 (cnmf) | units | recall | precision | F1 | trace corr median / q25 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 min | 6 | 43.4 s | 11.0 s | 5.9 s | 25.4 s (23.9 s) | 404 | 0.933 | 0.973 | **0.953** | **0.995** / 0.991 |

Same units and detection scores as before; trace corr median 0.99505 -> 0.99515, q25
0.99101 -> 0.99119. The extra background pass costs 3.8 s. Peak RSS: main 3.8 GB, largest
worker 2.2 GB. The 10-min rows above were not rerun.

- Gate: F1 >= 0.829 - 0.02 = 0.809 and trace corr >= 0.964 - 0.02 = 0.944: passed. Targets
  5-min <= 160 s and 10-min <= 6 min on 6 workers: met.
- Disk: tile store = raw size (3.24 / 6.48 GB, deleted after pass 2), factors 52 / 93 MB,
  cnmf 33 / 61 MB. Peak RSS: main 3.8-4.0 GB, largest worker 2.2-2.3 GB.
- Seed and unit counts, 5 min: seeds_init 2273 -> pnr 1258 -> merge 407 -> initA 407 ->
  final 404. 10 min: 3926 -> 1903 -> 402 -> 400 -> 400. Motion: median frame quality
  0.835 / 0.844, all alignment self-checks ok (median match distance 0.16 / 0.17 px).
- The 1- and 6-worker 10-min runs give identical scores and unit counts.
- Window 2 runs in one process; its time is mostly `update_background` (two full passes over
  the reconstructed movie per call, single-threaded numpy, ~4.7 s per pass on 5 min). The
  NNQP spatial solve takes 0.1-0.2 s.

## Default changed and why

**`gmm_refine` off by default** (`Params.gmm_refine = False`, opt-in). Diagnostic run with it
on (old default, 5 min): 199 units, recall 0.468, precision 0.990, F1 0.635. Seeds matched to
truth (a true cell is covered when a seed lies within 5 px):

| stage | seeds | true cells covered (of 421) |
|---|---|---|
| seeds_init | 2273 | 421 |
| pnr_refine (auto) | 1258 | 421 |
| gmm_refine | 595 | **216** |
| seeds_merge after gmm | 199 | 211 |
| seeds_merge without gmm | 407 | 412 |

All synthetic cells share one amplitude distribution, so the 2-GMM on peak-to-peak splits
real cells into a bright and a dim half and drops the dim half. Minian's reference run did
not use `gmm_refine` either. With it off the 5-min F1 is 0.953. No other default changed.

Checked and left unchanged (measured on the 5-min factors):
- `pnr_thres="auto"`: cuts at pnr 2.5-2.7. PNR of seeds on true cells: min 2.9, 5th
  percentile 6.2; non-cell seeds: 95th percentile 0.34. Fixed thresholds 1, 1.5, 2, 2.5 keep
  the same 421 cells; auto drops none.
- `merge_corr=0.8`: of 2059 seed pairs within 10 px, 1444 are on the same cell (median corr
  0.999) and 603 on different cells (median 0.28, 9.6 % above 0.8). 20 of 407 merged groups
  span more than one cell (max 3). Of the 25 in-FOV cells we miss, most have another true
  cell within 2-7 px; Minian finds 21 of those 25 but misses 29 in-FOV cells overall.
- `inita_corr=0.8`: no sweep needed (gate passed by 0.14 F1).
- `sparse_penal` / `sn`: see README "Known differences"; scaling `sn` by Minian's logmexp
  factor 0.749 changed no score.
- Spatial NNQP sweeps (5 min): median 1, p99 11-12; 46 of 246 820 pixels hit the 100 000
  sweep cap in iteration 0 (none in iteration 1). The whole solve takes 0.16 s, so no cap change.

## Fixes found by the runs

- `update_background` used all K units in every tile's `A^T C`; now only the units touching
  the tile (results bit-identical, background pass 10.6 s -> 8.8 s on 5 min).
- Accelerate float32 transposed gemv (`R.T @ b`) returned wrong values for output lengths of
  about 513-575 right after a numba parallel region (macOS 26, M1 Max; reproduced in isolation,
  never without a prior numba parallel call, never in float64). A session whose frame count
  leaves a 541-frame tail chunk got NaN in `f` and then all units dropped (3 of 6 runs of a
  39541-frame real session). `update_background` now uses `np.einsum` for this product
  (same speed). The synthetic sessions (T multiple of 1000) never hit it. The final review
  could not reproduce the bug (0 of 150 trials with the old code), so the cause is not
  established: `run_cnmf` now raises on NaN/inf after every stage, and `update_background`
  recomputes one tile chunk in float64 and warns above 1e-3 relative error. The unit test
  for it is a canary that also passes on the old code.
- `pnr_refine` / `gmm_refine` held all seed traces at once (5.4 GB for 11k seeds x 39541
  frames); now 1024 seeds at a time (0.5 GB).
- `read_avi` kept every decoded 3-channel frame alive until a final stack (4x the segment
  size for codecs decoded as BGR); it now fills one preallocated array, and the template
  reads only its 200 frames.

Real-data smoke run (WL25): not part of this result set; scheduled separately. Pass 1 used
to hold a whole segment ~3x in memory per worker (6.5 GB worker RSS for a 1.6 GB single-file
AVI); workers now stream 1000-frame batches (not yet measured on WL25).

## Real session WL25_20251204 (2026-10-19): motion before tophat

200 x 200 px, 39,822 frames, 20 Hz. Inputs (read-only): `WL25_20251204_destriped.avi` (Fiji
FFT destripe, pal8 rawvideo) and the existing Minian input `..._fftonly_proc.avi` (destripe +
`minizero_preprocess.py`: min-projection subtraction, gaussian 0.6, tophat r 8, gain 2.5).
Run folder: `~/minian_testing_local/WL25_20251204/` (see its README).

**Motion after the tophat fails on real data.** The tophat (and min-projection subtraction)
removes the static structure (vessels) that registration locks onto; what is left is sparse
flickering cells and a fixed template goes stale over 33 min (quality 0.49 -> 0.21).

| registration image, session-start template | jumps >= 3 px | frames at +-19 px |
|---|---|---|
| `fftonly_proc` (tophat before motion, old cafast order) | 4305 | 2364 |
| `minizero_preprocess` without tophat (min-proj subtracted) | 2114 | 1423 |
| destriped, band-pass 1-6 px (`motion_before_tophat`, new default) | 5 | 0 |

Default changed: `motion_before_tophat = True` (median -> shifts on a band-passed copy ->
warp -> tophat). Synthetic 5 min: F1 0.954, trace corr 0.995 (was 0.953 / 0.995); wall
43.4 -> 49.6 s (band-pass 10 CPU-s, extra warp of the registration image for the quality
score). The tiny 64 x 64 session cannot be registered in either order (3.8 / 5.8 px); its
end-to-end test keeps the old order.

**vs the existing Minian run** (`bench/compare_minian.py`, params `params_wl25_destriped.json`:
Minian's WL25 set, `median_ksize 3`, `tophat_wnd 8`, `seeds_diff_thres 8` for gain 1, no
upper size limit): CaFast 132 units / Minian 158, 115 matched (73 % of Minian, 87 % of
CaFast), trace corr median 0.80 (CaFast `YrA` vs Minian `C`, neither deconvolved), 49 s.
Open: CaFast footprints are larger (median 188 vs 54 px above 10 % of peak, 601 vs 67 above
1 %), so Minian's `size_thres` upper limit 250 drops every CaFast unit. Minian's run is not
a clean reference: extra unit merge, `drop_correlated_pairs`, post-hoc filters, and its
`motion.zarr` is all NaN.

**Minian v2.0 timing on WL25** (`bench/minian_real.py`, Minian venv, 4 workers x 2 threads):
378 s for the 39,822 frames (+ about 25 s of `minizero_preprocess` for its input), 165 units,
12.4 GB peak disk; cafast on the destriped video: 49 s. Minian skips its own preprocessing
but runs motion (the input is not motion-corrected; the original WL25 run also entered at
`estimate_motion`). Its last stages overlapped a cafast run, so 378 s is slightly high. Two
fixes were needed: the merge results are saved under new names (`save_minian(overwrite=True)`
deleted `A_init` / `C_init` while they were still being read), and Minian's `LassoLars` gets
the unit-norm column scaling that scikit-learn < 1.2 applied by default (a local change in
`minian_v2.0/minian/cnmf.py`, uncommitted there). After the same fix in cafast (`61a63ce`):
footprints 67 vs 51 px above 10 % of peak, 110 of 165 Minian units matched, trace corr
median 0.83.

## Second CNMF iteration: n_iter 1 vs 2 (2026-10-19, commit d616477)

`bench/compare_iters.py RUN_A RUN_B [--out FILE.json]` matches the units of two runs by
centroid (`score.match_units`, 5 px) and reports footprint IoU (pixels above 10 % of peak),
Pearson correlation of matched `C` and `YrA`, cnmf seconds, unit counts and, on synthetic
sessions, both scores. It splits pairs by whether RUN_A's unit came out of RUN_A's last
merge (row maximum < 1), and the others by whether they overlap such a unit. Runs:
`processed/cafast_iter{1,2,3}` of each session (WL27 params of `cafast_v1`, synth defaults;
n_iter 3 on WL27 only), motion sidecar reused, 6 workers. Factors and all pre-CNMF
counts are identical between the two runs.

| session | cnmf s (1 / 2) | units (1 / 2) | matched | footprint IoU median (q05) | `YrA` corr median (q05) | `C` corr median (q05) |
|---|---|---|---|---|---|---|
| synth 5 min | 18.8 / 24.0 | 403 / 403 | 402 | 1.000 (0.967) | 1.000 (1.000) | 1.000 (0.998) |
| WL27, all | 9.8 / 11.6 | 91 / 90 | 90 | 0.876 (0.695) | 0.990 (0.966) | 0.985 (0.890) |
| WL27, merged in iter 0 | | | 12 | 0.826 (0.633) | 0.985 (0.956) | 0.984 (0.947) |
| WL27, not merged, touching a merged unit | | | 12 | 0.839 (0.690) | 0.985 (0.964) | 0.980 (0.880) |
| WL27, not merged, isolated | | | 66 | 0.907 (0.723) | 0.992 (0.970) | 0.985 (0.883) |
| WL27, n_iter 2 vs 3 | 11.6 / 14.3 | 90 / 90 | 90 | 0.941 (0.814) | 0.999 (0.989) | 0.998 (0.972) |

- Synth score is unchanged: n_iter 1 and 2 both give recall 0.938, precision 0.980, F1 0.959,
  trace corr median 0.994. No unit merges in either iteration.
- WL27 already differs after the iteration-0 merge (91 vs 90): with n_iter 1 that iteration
  is the last, so it recomputes the background footprint `b` before the merge. The one extra
  n_iter-1 unit is a merged unit.
- On WL27 the second pass changes unmerged footprints too, not only the merged ones: units
  that overlap no merged unit also move (IoU median 0.91, q05 0.72). Its spatial update uses the residual traces
  `YrA` instead of the init traces. Median footprint size above 10 % of peak: 69 px
  (n_iter 1), 75 (2), 75 (3). Traces change little.
- A third pass changes footprints less again (IoU 0.94) and traces hardly at all.

**Conclusion:** n_iter 1 is safe on synth (identical score, -5 s) but not on WL27, where the
second pass refines footprints of all units, not only merged ones, for 1.8 s. Refitting only
merged units would not reproduce it, so keep n_iter 2 as the default.
The cnmf seconds above predate the background speed-up below, which removes most of them.

```bash
P=/Users/mbrosch/minian_testing_local/WL27_20251205/processed
.venv/bin/python bench/compare_iters.py $P/cafast_iter1 $P/cafast_iter2 --out $P/cafast_iter2/compare_iters.json
```

## Long recordings (2026-10-20): unit loss on 2 h sessions

Runs: `V4_2h/processed/cafast_2h` (UCLA V4, 600 x 600, 140,000 frames at 20 fps, default
params) and `WL27_2h/processed/cafast_2h` (200 x 200, 144,000 frames, tuned params). V4
kept **17 units** against 362 on a 10,000-frame cut of the same recording; WL27 2 h kept 73.
Scripts for the measurements below were run from the saved factors (window 2 only; slices
of the 2 h factors restrict `V` to a frame range and re-orthonormalise it).

**Not the compression.** Window 2 on the 2 h factors restricted to frames 100k-110k gives
463 units (the 10k-frame run of those files: 362), so the factor subspace holds the cells.
On one 64 x 64 tile the streamed rSVD matches an exact SVD (Gram matrix, 300 components)
at 10k, 40k and 140k frames: ranks 41/42/43 vs 42/42/44, every exact component that passes
the autocorrelation cut captured > 0.9 except the last few. Only the densest tile hits the
rank cap: 124 components pass the cut at 10k frames, 176 at 140k, and the cap at 100 drops
76 of them (capture < 0.5). `MAX_RANK` is now 300 (that tile: rank 175, 22 s instead of 19).

**Cause 1: the spatial penalty grows as sqrt(T).** Units initialised on the 10k slice,
spatial update at full T: `sparse_penal 0.01` keeps 149 of 632 (median 63 px); scaling it by
`sqrt(10k / 140k)` keeps 525 (133 px), the slice itself 492 (134 px). WL27 2 h with its tuned
0.0035: 72 units of 30 px; with the equivalent of its 38k-frame tuning (`sparse_snr 0.69`):
131 units of 92 px. Fix: `sparse_snr`, `lam = sqrt(T) * sparse_snr * sn` (README).

**Cause 2: the seed merge percolates.** V4 2 h: 48,405 seeds (280 rolling windows; 7,864
on 10k frames) -> 32,314 after PNR -> **166** merged. Subsampling the same seeds to 5,000
gives 449 merged, to 1,500: 584; the slice (4,980 seeds) 646. Of the 1.8 M seed pairs within
10 px, 42 % correlate > 0.8, and of those 95 % still correlate > 0.7 after a high-pass at
0.002 cycles/frame, so the links are mixture pixels between cells, not drift. Minian's
connected components run through them once the halo is fully seeded. Variants on the full
2 h (penalty fixed; "dup" = unit pairs with centroids within 6 px and `YrA` corr > 0.7):

| seed merge | V4 2 h units (dup pairs) | V4 slice 10k (dup) | WL27 38k | WL27 2 h | synth F1 |
|---|---|---|---|---|---|
| connected components (Minian) | 121 (2) | 463 (26) | 90 | 131 | 0.9587 |
| brightest-first suppression (NMS) | 1770 (1249) | 763 (143) | 90 | 130 | 0.9688 |
| high-pass 0.002 cyc/frame + components | 360 (60) | 724 (227) | 94 | 124 | 0.9587 |
| 21 windows, length ~ T | 406 (4) | - | - | 103 | - |
| **brightest parent (tree), now default** | **490 (13)** | 572 (43) | 95 | 131 | **0.9638** |

NMS keeps every dim halo seed that correlates < 0.8 with the kept centre as a unit; the
high-pass dilutes same-cell correlations (duplicates); fewer, longer windows fix the density
but lose coverage on the sparse WL27 session. The tree rule (each seed joins its brightest
correlated neighbour, roots are kept) absorbs the halo downhill and cannot cross the
brightness valley between two cells. Synth: recall 0.938 -> 0.948, precision 0.980 both.

Final code (`fix/rank-vs-T`), V4 2 h from the kept tile store, pass 2 + window 2 with
default params: **496 units** (17 before), 10 duplicate pairs, median footprint 172 px; pass 2 313 s (264 s with the cap at 100; rank sum 2885 vs 2710, 4 tiles above 100, max 175), window 2 104 s (`pnr_refine` 64 s on 48k seeds). Units sit on cells (`V4_2h/cafast_2h_fix_units.png`). Synthetic 5 min, full pipeline (`processed/cafast_rank_vs_T`): 407 units, recall 0.948, precision 0.980, F1 0.9638, trace corr median 0.994, 46.1 s wall (was 403 / 0.938 / 0.980 / 0.9587 / 0.994).

**3 h runs with the fix** (2026-10-20, 6 workers, M1 Max, run alone): V4 files 0-215 of the 24 h
recording (`/Volumes/Extreme SSD/cafast_v4_3h`, 216,000 frames, 600 x 600) and WL27 hours 1-3
(`WL27_2h/raw_3h`, destriped, 216,000 frames, 200 x 200, tuned params with `sparse_snr 0.69`).

| stage | V4 3 h | WL27 3 h |
|---|---|---|
| pass 1 (preprocess + motion) | 450 s | 63 s |
| pass 2 (compression, rank cap 300) | 503 s | 171 s |
| seeds (init + PNR + merge) | 140 s | 27 s |
| init (initA, initC, unit_merge) | 14 s | 86 s |
| CNMF | 28 s | 32 s |
| wall | 1185 s | 389 s |
| peak RSS main / worker | 13.9 / 4.7 GB | 3.7 / 2.2 GB |
| raw = tile store | 78 GB | 8.6 GB |
| seeds -> PNR -> merged | 59,504 -> 39,769 -> 582 | 4,042 -> 2,512 -> 846 |
| units | **503** | 112 |

V4: 503 units at 3 h vs 496 at 2 h, rank sum 2950 (max 173). WL27: `unit_merge` 64 s and
`initC` 20 s on 846 initial units (the hour-2 brightness step inflates its seeds); motion
quality 0.47. Pass 2 is now the longest stage on both; it grew with the rank cap (WL27 2 h:
51 s before).

Tile store of the 2 h session kept at `/Volumes/Extreme SSD/cafast_v4_2h/processed/store/`
(50 GB) for further pass-2 experiments.

## Background speed-up (2026-10-19)

`update_background` was ~80 % of the `cnmf` stage (three calls, five reconstructions of the
movie). Two variants, measured with `bench/background_variants.py` (window-2 rerun from saved
factors; baseline = the code before this change on the same factors). Factors: synth
`synth_v4_5min/processed/cafast_iter2`, WL27 `WL27_20251205/processed/cafast_iter2`, WL25
`WL25_20251204/processed/cafast_sn_fft`. Outputs in `<session>/processed/bg_bench/`.
M1 Max, 10 cores; another session kept the load at 5-7 during all runs.

1. **Threads over tiles** (now always on, `threads` = all cores). Each tile's `R^T b` is
   summed in tile order, so the output is bit-identical to the serial loop for any thread
   count (`b`, `f`, `A`, `C`, `YrA` all `array_equal`); the float64 spot check is unchanged.
2. **Unclipped, on the factors** (`Params.background_clip = False`, opt-in):
   `b = clip(mean_t Y - A^T mean_t C, 0)`, `f_t = (Y_t - A^T C_t) . b / (b . b)` with
   `Y_t . b` from `V W^T b + mean . b`. No movie reconstruction.

Seconds (background = sum of the three `update_background` calls):

| session | frames | cnmf base | cnmf threads | cnmf unclipped | background base | threads | unclipped |
|---|---|---|---|---|---|---|---|
| synth 5 min | 9,000 | 24.0 | 9.0 | 4.2 | 19.7 | 4.7 | 0.05 |
| WL27 | 38,763 | 11.8 | 4.5 | 2.0 | 10.0 | 2.6 | 0.07 |
| WL25 | 39,822 | 12.7 | 5.2 | 2.7 | 10.1 | 2.6 | 0.09 |

Thread scaling, synth background: 1 thread 20.0 s, 2: 10.4, 4: 6.7, 6: 5.4, 10: 4.7
(memory-bound: the elementwise clip and row sum dominate; chunk size 128-2000 changes nothing).

Unclipped vs base (units matched by centroid; every unit matched, same unit counts):

| session | b max rel diff | f max rel diff | C corr min / median | YrA corr min / median |
|---|---|---|---|---|
| synth | 0.034 | 0.0048 | 0.99995 / 1.0000 | 0.99983 / 1.0000 |
| WL27 | 0.062 | 0.011 | 0.9997 / 0.99996 | 0.9994 / 0.99996 |
| WL25 | 0.058 | 0.011 | 0.9995 / 0.99995 | 0.9984 / 0.99993 |

Synth score (`cafast score`): base, threads and unclipped all F1 0.9587 (403 units, recall
0.938, precision 0.980), trace corr median 0.9941, q25 0.9893 / 0.9893 / 0.9894. WL25 vs
Minian (`compare_minian.py`): 124 of 158 matched for base and unclipped, trace corr median
0.813 / 0.812.

**Default changed: `background_clip = False`.** It takes the background to ~0 s (`cnmf`
another ~2x faster than threads) and changes no score here (worst unit trace corr 0.998);
Minian parity of `b` itself is not needed. `background_clip = True` keeps Minian's definition,
now threaded (bit-identical to the old loop, `cnmf` 2.4-2.7x faster than before).
