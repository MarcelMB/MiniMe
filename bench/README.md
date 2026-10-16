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
