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

Caveat: the script saved the normalised `A_new` over the stored `A_new.zarr` while reading it, so
the stored `A_new` was all NaN (`C_new` was fine). `update_spatial` was re-run on the kept
`Y_hw_chk`, `A`, `C`, `sn_spatial` (same parameters, 105 s, 508 of 508 units kept, same unit ids
as `C_new`) and its un-normalised `A_new` exported; normalisation does not change centroids.
`consolidated=False` is needed when opening these stores (stale consolidated metadata).

Minian's accuracy (508 units, 421 true cells):

| recall | precision | F1 | trace corr median | trace corr q25 | median match dist |
|---|---|---|---|---|---|
| 0.914 | 0.758 | **0.829** | **0.964** | 0.942 | 0.14 px |

Offset between frame and template coordinates: (-7.47, -7.88) px, sign +1. Traces are Minian's
`C_new` (trace after the spatial update).

## Commands
```bash
MINIAN_PY=/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/minian_v2.0/.venv/bin/python
OUT=/Users/mbrosch/minian_testing_local/synth_v4_5min/processed/minian_baseline_keep
RAW=/Users/mbrosch/minian_testing_local/synth_v4_5min/raw
# re-run Minian keeping intermediates (in minian_v2.0)
$MINIAN_PY scripts/bench_browser/run_minian_baseline.py --keep
# export under the Minian venv (writes $OUT/minian_units.npz), then score under ours
$MINIAN_PY bench/minian_export.py $OUT
.venv/bin/fastminian score --minian $OUT/minian_units.npz --raw $RAW   # -> $OUT/score.json
# score a fastminian run (--raw only needed if RUN_DIR/params.json is missing)
.venv/bin/fastminian score RUN_DIR [--raw $RAW]                         # -> RUN_DIR/score.json
```
`bench/minian_export.py` prints an error instead of exporting if `A_new` contains NaN; the
clobbered store then needs the `update_spatial` re-run described above.
