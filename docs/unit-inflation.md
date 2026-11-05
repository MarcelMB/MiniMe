# Browser unit inflation (2026-11-05)

**Question.** On real V4 data the browser build finds more units the longer the session (15 /
30 / 60 min: 766 / 768 / 958) while Python stays flat (635 / 630 / 592). Are the extra units real?

**Answer.** Mostly yes on synthetic data, but part of the growth was a stitch defect.

**Cause.** Each window's CNMF merges units closer than `merge_min_dist`, and so does Python
over the whole session. The browser's stitch never did that across windows. When one window
estimated a cell's footprint worse (cosine ≤ 0.7 with the cell's other windows), that cell
kept a second unit. Each window adds a few such units, so the count grows linearly with the
session length.

**Fix.** The stitch now joins groups whose mean footprints have centroids closer than
`merge_min_dist` (each group joins its brightest close group, the stitch's own chain guard).
Rust `stitch.rs` (`StitchParams.min_dist`) and the Python reference `stream.stitch(min_dist=)`
give identical footprints. `session.ts` passes `merge_min_dist`: it is 0 (off) with default
params, and auto-tune sets it to 0.4 × the cell diameter.

## Synthetic sessions (truth; grin_v4, tuned `grin.json` params, 8 workers)

The 60 and 120 min sessions have 60 cells that stay silent for the first half
(`synth_v4_grin{60,120}_late`).

**Browser before and after the fix:**

| length (windows) | units, before → after | real | duplicate | out-of-focus | noise | piece | F1, before → after |
|---|---|---|---|---|---|---|---|
| 30 min (12) | 378 → 353 | 318 → 312 | 13 → 4 | 40 → 32 | 5 → 3 | 2 → 2 | 0.758 → 0.767 |
| 60 min (24) | 404 → 384 | 337 → 331 | 6 → 1 | 49 → 41 | 7 → 4 | 5 → 7 | 0.779 → 0.783 |
| 120 min (48) | 455 → 401 | 341 → 335 | 18 → 3 | 75 → 47 | 16 → 11 | 5 → 5 | 0.745 → 0.777 |

**Python, same params:**

| length | units | real | out-of-focus | F1 |
|---|---|---|---|---|
| 30 min | 269 | 249 | 14 | 0.682 |
| 60 min | 240 | 229 | 4 | 0.653 |
| 120 min | 229 | 218 | 5 | 0.632 |

**After the fix, end to end:** precision 0.84 / 0.86 / 0.84 at 30 / 60 / 120 min (Python
0.93–0.95).

**Classes:**
- **real:** matched one-to-one to a true cell (≤ 5 px).
- **duplicate:** best true cell by footprint cosine ≥ 0.5, and that cell is already matched.
- **piece:** best true cell by cosine ≥ 0.5, but that cell is not matched.
- **out-of-focus:** within 1.5 σ of an out-of-focus cell (real signal, but not in the truth).
- **noise:** none of the above.

**Real extra units.** The browser finds 75 / 112 / 126 true cells that Python misses (mostly
dim, small and overlapping cells), and Python finds 6 / 4 / 9 the browser misses.

**Late cells.** Silent-then-active cells are found by both: recall 0.85 / 0.85 / 0.82 in the
browser vs 0.85 / 0.87 / 0.77 in Python. The fix does not change late recall (60 min:
0.85 → 0.85). The visible jump of real units at mid-session (60 min: 285 → 334 between
windows 12 and 16) is the late cells switching on.

**Duplicates by window count.** With the old stitch they grow with the number of windows
(120 min: 9 / 17 / 19 / 18 after 8 / 16 / 32 / 48 windows); with the fix they stay at 2–5.
Of the 13 duplicates at 30 min, 11 were found in one window only, never shared a window with
their cell, and had a best cross-window cosine of 0.63 (median) to it: the cell itself,
estimated worse, below the 0.7 link.

**Remaining false class: out-of-focus cells** (47 at 120 min, Python 5). These are dim
(median SNR 16 vs 40 for real cells) and mostly seen in one window. Per-window discovery picks
them up for the same reason it finds the extra dim real cells. An SNR floor only trades
between the two (60 min: best F1 0.787 vs 0.783), so it is not applied.

## Real V4 (no truth; browser vs Python matched by centroid ≤ 5 px)

| length | Python units | browser, before → after | browser units matching Python, before → after | single-window units, before → after |
|---|---|---|---|---|
| 15 min | 635 | 766 → 676 | 577 → 556 | 217 → 148 |
| 30 min | 630 | 768 → 673 | 563 → 548 | 197 → 125 |
| 60 min | 592 | 958 → 755 | 520 → 503 | 329 → 163 |

Before the fix, units seen in a single window arrive at ~13 per window (5–24) and only 18 %
of them match a Python unit. The fix cuts growth from 15 to 60 min from +25 % to +12 %.

## Rejected

- **Connected components on the close pairs** (Python's own `unit_merge` rule): on real V4 at
  60 min it chained up to 15 groups, and 26 components spanned 2 or more Python units
  (matches lost: 37 vs 17 with brightest-parent).
- **Attaching a one-window unit to the most similar recurring cell** (cosine ≥ 0.4–0.5): best
  on synthetic data (30 min F1 0.770), but on real V4 it joined units ~18 px apart, because
  large footprints give high cosines.
- **The streaming pass's trace-correlation merge** (`Amerged`, unused): on 30 min it merged 11
  pairs of distinct overlapping true cells and only 2 duplicates. The traces of a duplicate and
  its cell compete in the least-squares solve and do not correlate.

## Open

- Real-data visual check of the remaining single-window units (163 at 60 min). The stitch
  writes `n_windows.npy`, but the run folder's `cnmf/units.csv` has no window count yet.
  Adding it would let the viewer flag these units rather than drop them.
- The 2–24 h real runs with the fix, overnight only.

Scripts and inputs → outputs: `experiments/unit_inflation/README.md`; numbers in
`experiments/unit_inflation/out/`.
