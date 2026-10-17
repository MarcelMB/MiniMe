# bench: scoring against truth and the Minian reference

Inputs -> outputs: a pipeline's units (footprints `A`, traces, motion `shifts`) -> `score.json`
(recall, precision, F1, trace correlation) against the ground truth of the synthetic session
(`fastminian.testing.synth`, dataset meta in `<raw>/../meta.json`).

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
.venv/bin/fastminian score --minian $OUT/minian_units.npz --raw $RAW
# score a fastminian run (--raw only needed if RUN_DIR/params.json is missing)
.venv/bin/fastminian score RUN_DIR [--raw $RAW]                         # -> RUN_DIR/score.json
```
`bench/minian_export.py` refuses to export if `A_new` contains NaN. Score dicts carry
`alignment_ok`: False (plus a stderr warning) when neither offset sign gives a median matched
distance below 1.5 px, i.e. the motion offset could not be verified.

## fastminian runs (Task 12, commit 16f6ac8, M1 Max, 10 cores)

Fresh runs (no motion sidecar reused), default params, scored with `fastminian score`.

| session | workers | wall | pass 1 | pass 2 | window 2 (cnmf) | units | recall | precision | F1 | trace corr median / q25 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 min, Minian 2.0.2 | 4 x 2 thr | 471 s | | | | 508 | 0.914 | 0.758 | 0.829 | 0.964 / 0.942 |
| 5 min | 6 | 40.1 s | 11.1 s | 6.4 s | 21.5 s (20.1 s) | 404 | 0.933 | 0.973 | **0.953** | **0.995** / 0.991 |
| 10 min | 6 | 68.5 s | 17.2 s | 11.1 s | 38.6 s (35.5 s) | 400 | 0.924 | 0.973 | 0.948 | 0.995 / 0.989 |
| 10 min | 1 | 147.3 s | 65.4 s | 42.2 s | 38.4 s (35.6 s) | 400 | 0.924 | 0.973 | 0.948 | 0.995 / 0.989 |

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
  (same speed). The synthetic sessions (T multiple of 1000) never hit it.
- `pnr_refine` / `gmm_refine` held all seed traces at once (5.4 GB for 11k seeds x 39541
  frames); now 1024 seeds at a time (0.5 GB).
- `read_avi` kept every decoded 3-channel frame alive until a final stack (4x the segment
  size for codecs decoded as BGR); it now fills one preallocated array, and the template
  reads only its 200 frames.

Real-data smoke run (WL25): not part of this result set; scheduled separately. Open issue
seen in exploratory runs: pass 1 holds a whole segment ~3x in memory per worker, which only
matters for single-file sessions (6.5 GB worker RSS for a 1.6 GB AVI).
