# CaFast auto-tune (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cafast autotune RUN_DIR --diameter D` (and a "Propose" button on `cafast tune`) searches the window-2 parameters stage by stage, scores each setting with label-free unit checks, and writes the proposal as a normal rerun `RUN_DIR/reruns/auto/` that the tune page shows with its sweep curves and flagged units.

**Architecture:** A new package `cafast.autotune` with `score.py` (pure unit checks on A and traces) and `sweep.py` (an in-process stage runner on a `Window2Session`: factors and `FactorCorr` loaded once, seeds cached after `pnr_refine`, the chosen init reused for the CNMF grid), plus `run.py` (writes the rerun, `sweep.csv`, `autotune.json`, `unit_flags.csv`). The tune page gains a propose form, a proposal panel (table + one small curve per searched parameter, from `sweep.csv`), good/flagged columns in the run list, a flagged-units gallery, and a diameter measuring tool on the raw max projection.

**Tech Stack:** Python 3, numpy, scipy (`cKDTree`), pandas, OpenCV (`cv2`), matplotlib (`Figure` only, no pyplot), typer, stdlib `http.server`. No new dependencies.

**Spec:** Obsidian Science vault, `CaFast auto-tuning.md` (design agreed 2026-10-19). The parts this plan implements are restated in "Design" below so the plan is self-contained.

## Design (from the agreed note)

- **Input:** cell diameter `d` in px of the factors' frame, given once (CLI `--diameter`, or measured on the page).
- **Derived from d, not searched:** `seeds_max_wnd = inita_wnd = dl_wnd = merge_dist = round(d)`; `size_thres = (round(0.25·a), round(4·a))` with `a = π(d/2)²`.
- **Unit checks** (good = passes all four, else flagged with every failing reason):
  - `low_pnr`: trace PNR below `pnr_min` (see note below on the scale)
  - `in_pieces`: the largest 8-connected piece above 10 % of peak holds < 90 % of the footprint weight above 10 %
  - `bad_size`: pixels above 10 % of peak outside 0.3–3 × `a`
  - `duplicate`: another unit with centroid closer than `d` and trace corr ≥ 0.8 has a higher PNR (the weaker one is flagged)
  - No border check (decided).
- **Score** = good − flagged. Ties: fewer searched parameters changed from the run's own params, then first tried.
- **Search** (each setting is one row of the map):
  1. `seeds`: grid `seeds_diff_thres` {2, 2.5, 3, 4} × `pnr_thres` {auto, 1, 1.5, 2, 3}, scored on the init units
  2. `merge`: from the best seeds (cached after `pnr_refine`): grid `merge_corr` {0.7, 0.75, 0.8, 0.85, 0.9} × `init_merge_corr` {0.7, 0.8, 0.9}; then `inita`: `inita_corr` {0.6, 0.7, 0.8, 0.9} at the best merge values
  3. `confirm`: the top 3 distinct init-level settings through CNMF with the run's CNMF params; best final score wins
  4. `sparse`: `sparse_penal` {0.002, 0.003, 0.005, 0.007, 0.01, 0.02} on the chosen init; then `loop_merge`: `loop_merge_corr` {0.85, 0.9, 0.95} at the best penalty
  - A proposed value at either end of its numeric grid gets an "at edge" badge.
- **Outputs** in `RUN_DIR/reruns/auto/`: a normal rerun + `sweep.csv` + `autotune.json` + `unit_flags.csv`.
- **Page:** propose form (diameter + measure) → background run with progress → proposal panel (param, run value, proposed, curve, badge); override = view `auto`, edit the prefilled form, Rerun (opens compared with `auto`); flagged-units gallery by reason → per-unit view; show curves, not 50 runs.
- **Not in v1:** time-subset sweep for long recordings (phase 2), artifact guard, border check, per-iteration `dl_wnd`/`size_thres`.

**PNR scale (deviation to confirm with the user):** the note says "PNR ≥ 4". `figures.trace_pnr` (the `pnr` column the page already shows) is (99.9th − 0.1th percentile) / diff-noise std, so pure white noise scores ≈ 6.2 and 4 would pass noise. This plan keeps that one scale and sets the default `pnr_min = 12` (≈ 2 × pure noise). Task 8 calibrates it on WL25.

## Global Constraints

- Project-local env only: `uv venv` / `uv pip install`; run tools as `.venv/bin/python -m pytest`, `.venv/bin/ruff`. Never touch the base conda env.
- Before a task is done: `.venv/bin/ruff format src tests bench` and `.venv/bin/ruff check --fix src tests bench`, no remaining errors.
- Type hints on every function signature; small single-purpose functions; section headers (`# ----` blocks) as in the existing files.
- TDD: failing test first, then code.
- Commit messages: one short sentence, no body, **no Co-Authored-By or any Claude/Anthropic attribution**.
- Raw data is read-only; everything is written under `RUN_DIR/reruns/` or `RUN_DIR/tune/`.
- No new dependencies.
- Window-2 parameters only; factor parameters (`window2.FACTOR_PARAMS`) are never changed by the tuner.
- A rerun with the same params must stay bit-identical to a full run (`tests/test_window2.py::test_rerun_with_run_params_reproduces_full_run`).

## Review Focus

1. **Flat traces** (a unit whose trace is constant, e.g. all zeros) must be flagged `low_pnr`; today `trace_pnr` returns `inf` for them, which would pass. → test in Task 2.
2. **Settings that fail** (`pnr_refine` raises "no seeds survived", CNMF raises on non-finite values) must be recorded as failed rows and the sweep must go on; if every setting of a step fails, a clear `RuntimeError` names the step. → tests in Task 3.
3. **Empty or degenerate unit sets** (K = 0, or a footprint row with no pixels, whose centroid is NaN) must score 0/0 without crashing the KD-tree or the corr. → tests in Task 2.
4. **Bad diameters** (0, negative, text, fractional like 7.5, larger than the frame) → `ValueError`/HTTP 400 with a message for invalid ones, rounded ints ≥ 1 for valid ones. → tests in Tasks 3 and 5.
5. **The cache gives the same answer as a fresh rerun:** the proposal's score in `autotune.json` must equal the score of the written `reruns/auto` (catches in-place mutation of the reused init `A`, `C` or stale cached seeds). → test in Task 4.

---

## File structure

| File | Responsibility |
|---|---|
| `src/cafast/pipeline.py` (modify) | split `_seeds` into `seeds_unmerged` + `seeds_merged`; rename `_init_units` → `init_units` |
| `src/cafast/autotune/__init__.py` (create) | package docstring only |
| `src/cafast/autotune/score.py` (create) | footprint metrics (moved from `tune/figures.py`), `Checks`, `unit_flags`, `summarize` |
| `src/cafast/autotune/sweep.py` (create) | `Grids`, `derived_params`, `Trial`, `best`, `ranked`, `at_edge`, `Sweep` |
| `src/cafast/autotune/run.py` (create) | `autotune()` driver, `sweep_table`, output files |
| `src/cafast/cli.py` (modify) | `cafast autotune` command |
| `src/cafast/tune/figures.py` (modify) | import metrics from `score`; `summary`/`unit_table` take optional `Checks`; draw sweep curves for runs that have `sweep.csv` |
| `src/cafast/tune/autofig.py` (create) | sweep curves, unit thumbnails, raw max-projection PNG (keeps `figures.py` from growing) |
| `src/cafast/tune/server.py` (modify) | `checks()`, `start_autotune`, routes `/api/autotune`, `/thumb/RUN/K.png`, `/maxproj.png` |
| `src/cafast/tune/page.py` (modify) | propose form, proposal panel, good/flagged columns, flagged gallery |
| `src/cafast/tune/static/tune.js`, `tune.css` (modify) | propose submit, thumbnail clicks, measure tool |
| `bench/autotune_check.py` (create) | WL25 calibration + acceptance numbers against Minian |
| `tests/test_autotune_score.py`, `tests/test_autotune_sweep.py` (create), `tests/test_window2.py`, `tests/test_tune.py` (modify) | tests |
| `README.md` (modify) | one short "Auto-tune" section |

---

### Task 1: Split the seed stage so the tuner can cache it

**Files:**
- Modify: `src/cafast/pipeline.py:140-232` (`_seeds`, `_init_units`, `run_window2`)
- Test: `tests/test_window2.py`

**Interfaces:**
- Produces:
  - `pipeline.seeds_unmerged(fac: Factors, params: Params, timer: StageTimer, stages: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame` — seeds after init, PNR and GMM (opt-in), before the seed merge.
  - `pipeline.seeds_merged(fac: Factors, fc: corr.FactorCorr, s: pd.DataFrame, params: Params, timer: StageTimer, stages: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame` — the seed merge on `s`; does not modify `s`.
  - `pipeline.init_units(fac, fc, s, params, timer, footprints=None) -> tuple[sps.csr_matrix, np.ndarray]` (the old `_init_units`, same body).

- [ ] **Step 1: Write the failing test** (append to `tests/test_window2.py`)

```python
def test_split_seed_stages_match_run_window2(tiny_run):
    from cafast import corr, pipeline
    from cafast.report import StageTimer

    fac = window2.load_factors(tiny_run)
    prm = window2.run_params(tiny_run)
    fc = corr.FactorCorr(fac, prm.noise_freq)
    w2 = pipeline.run_window2(fac, prm, StageTimer(), fc)
    timer = StageTimer()
    s_pnr = pipeline.seeds_unmerged(fac, prm, timer)
    before = s_pnr.copy()
    s = pipeline.seeds_merged(fac, fc, s_pnr, prm, timer)
    pd.testing.assert_frame_equal(s, w2.seeds["seeds_merged"])
    pd.testing.assert_frame_equal(s_pnr, before)  # the cached input is not modified
    A, C = pipeline.init_units(fac, fc, s, prm, timer)
    assert np.array_equal(A.toarray(), w2.footprints["init_merge"].toarray())
    assert timer.counts["seeds_merged"] == len(s)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_window2.py::test_split_seed_stages_match_run_window2 -v`
Expected: FAIL with `AttributeError: module 'cafast.pipeline' has no attribute 'seeds_unmerged'`

- [ ] **Step 3: Implement**

In `src/cafast/pipeline.py` replace `_seeds` and `_init_units` with:

```python
def seeds_unmerged(
    fac: Factors,
    params: Params,
    timer: StageTimer,
    stages: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Seeds after init, PNR and GMM (opt-in), before the merge; ``stages`` collects each table.

    ``stages["seeds_pnr"]`` holds every initial seed with its ``pnr`` and ``mask_pnr``.
    """
    p, n = params, timer.counts
    st = {} if stages is None else stages
    with timer("seeds_init"):
        s = seeds.seeds_init(
            fac.block_max,
            fac.block_edges,
            fac.T,
            p.seeds_wnd,
            p.seeds_stp,
            p.seeds_max_wnd,
            p.seeds_diff_thres,
        )
    n["seeds_init"] = len(s)
    st["seeds_init"] = s
    with timer("pnr_refine"):
        st["seeds_pnr"] = seeds.pnr_refine(fac, s, thres=p.pnr_thres)
        s = _filter(st["seeds_pnr"], "mask_pnr")
    n["seeds_pnr"] = len(s)
    if p.gmm_refine:
        with timer("gmm_refine"):
            s = _filter(seeds.gmm_refine(fac, s), "mask_gmm")
        st["seeds_gmm"] = s
    n["seeds_gmm"] = len(s)  # equals seeds_pnr when gmm_refine is off
    return s


def seeds_merged(
    fac: Factors,
    fc: corr.FactorCorr,
    s: pd.DataFrame,
    params: Params,
    timer: StageTimer,
    stages: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """The seed merge on ``s`` (left unchanged, so a cached ``seeds_unmerged`` can be reused)."""
    p = params
    with timer("seeds_merge"):
        s = _filter(corr.seeds_merge(fc, s, fac.max_proj, p.merge_dist, p.merge_corr), "mask_mrg")
    timer.counts["seeds_merged"] = len(s)
    if stages is not None:
        stages["seeds_merged"] = s
    return s


def init_units(
    fac: Factors,
    fc: corr.FactorCorr,
    s: pd.DataFrame,
    params: Params,
    timer: StageTimer,
    footprints: dict[str, sps.csr_matrix] | None = None,
) -> tuple[sps.csr_matrix, np.ndarray]:
    """initA, initC and the initial unit merge on the merged seeds ``s``."""
    fp = {} if footprints is None else footprints
    with timer("initA"):
        A = init.initA(fc, s, params.inita_corr, params.inita_wnd)
    timer.counts["units_initA"] = A.shape[0]
    fp["initA"] = A
    with timer("initC"):
        C = init.initC(fac, A)
    with timer("unit_merge"):
        A, C, _ = init.unit_merge(A, C, fc.traces_corr, params.init_merge_corr)
    timer.counts["units_init_merge"] = A.shape[0]
    fp["init_merge"] = A
    return A, C
```

In `run_window2` replace the two calls with:

```python
    s = seeds_merged(fac, fc, seeds_unmerged(fac, params, timer, st), params, timer, st)
    A, C = init_units(fac, fc, s, params, timer, fp)
```

Check that `corr.seeds_merge` does not modify its input frame: `grep -n "seeds\[" src/cafast/corr.py`. If it assigns columns into `seeds`, call it on `s.copy()` inside `seeds_merged`. Then `grep -rn "_seeds(\|_init_units(" src tests bench` and update any remaining caller.

- [ ] **Step 4: Run the window-2 tests**

Run: `.venv/bin/python -m pytest tests/test_window2.py tests/test_pipeline.py -v`
Expected: all PASS (including `test_rerun_with_run_params_reproduces_full_run`).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/pipeline.py tests/test_window2.py
git commit -m "Split the window-2 seed stage at the merge so a tuner can reuse seeds after PNR"
```

---

### Task 2: Label-free unit checks (`cafast.autotune.score`)

**Files:**
- Create: `src/cafast/autotune/__init__.py`, `src/cafast/autotune/score.py`
- Modify: `src/cafast/tune/figures.py:44, 100-170` (move metrics out, import them back)
- Test: `tests/test_autotune_score.py`

**Interfaces:**
- Produces (all in `cafast.autotune.score`):
  - `SIZE_FRAC = 0.1`, `REASONS = ("low_pnr", "in_pieces", "bad_size", "duplicate")`
  - `size_above(A, frac=SIZE_FRAC) -> np.ndarray`, `n_parts(A, H, W, frac=SIZE_FRAC) -> np.ndarray`, `trace_pnr(C) -> np.ndarray`, `footprint_image(A, k, H, W) -> np.ndarray`, `peak_mask(img, frac) -> np.ndarray` (moved from `figures`, `figures` imports them under their old names)
  - `main_piece_frac(A, H, W, frac=SIZE_FRAC) -> np.ndarray`
  - `duplicates(cent: np.ndarray, traces: np.ndarray, pnr: np.ndarray, radius: float, thres: float) -> np.ndarray` (bool)
  - `@dataclass(frozen=True) class Checks(diameter: float, pnr_min: float = 12.0, piece_frac: float = 0.9, size_lo: float = 0.3, size_hi: float = 3.0, dup_corr: float = 0.8)` with property `cell_area`
  - `unit_flags(A, traces, H, W, checks) -> pd.DataFrame` with columns `unit_id, pnr, size10, main_piece, low_pnr, in_pieces, bad_size, duplicate, good`
  - `summarize(flags: pd.DataFrame) -> dict[str, int]` with keys `good, flagged, score, low_pnr, in_pieces, bad_size, duplicate`

- [ ] **Step 1: Write the failing tests** (`tests/test_autotune_score.py`)

```python
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from scipy.signal import lfilter

from cafast.autotune import score

H = W = 50


def _footprints(blobs: list[list[tuple[int, int, int]]]) -> sps.csr_matrix:
    """One row per unit: the union of square blobs ``(row, col, half-side)``, value 1."""
    rows = []
    for unit in blobs:
        img = np.zeros((H, W), np.float32)
        for r, c, s in unit:
            img[r - s : r + s + 1, c - s : c + s + 1] = 1.0
        rows.append(img.ravel())
    return sps.csr_matrix(np.array(rows))


def _transient(rng: np.random.Generator, T: int = 2000, amp: float = 20.0) -> np.ndarray:
    s = np.zeros(T)
    s[rng.choice(T, 10, replace=False)] = amp
    return lfilter([1.0], [1.0, -0.9], s)


def _good_traces(K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([_transient(rng) + rng.normal(0, 1, 2000) for _ in range(K)])


CH = score.Checks(diameter=6)  # cell area 28.3 px -> size 8.5-84.8 px


def test_pnr_flags_noise_and_flat_traces():
    rng = np.random.default_rng(1)
    A = _footprints([[(10, 10, 2)], [(10, 30, 2)], [(30, 10, 2)]])
    traces = np.vstack([_good_traces(1)[0], rng.normal(0, 1, 2000), np.zeros(2000)])
    f = score.unit_flags(A, traces, H, W, CH)
    assert f["low_pnr"].tolist() == [False, True, True]
    assert score.trace_pnr(np.zeros((1, 100)))[0] == 0.0  # flat: 0, not inf


def test_size_and_pieces():
    A = _footprints(
        [
            [(10, 10, 2)],  # 25 px: fine
            [(10, 30, 0)],  # 1 px: too small
            [(30, 25, 6)],  # 169 px: too large
            [(40, 5, 2), (40, 30, 2)],  # two equal pieces
            [(20, 40, 2), (20, 46, 0)],  # one piece + a 1-px speck (25 / 26 > 0.9)
        ]
    )
    f = score.unit_flags(A, _good_traces(5), H, W, CH)
    assert f["bad_size"].tolist() == [False, True, True, False, False]
    assert f["in_pieces"].tolist() == [False, False, False, True, False]
    assert f.loc[3, "main_piece"] == pytest.approx(0.5)


def test_duplicate_flags_the_weaker_correlated_neighbour():
    rng = np.random.default_rng(2)
    A = _footprints([[(10, 10, 2)], [(10, 10, 2)], [(10, 14, 2)], [(40, 40, 2)]])
    t0 = _transient(rng)
    traces = np.vstack(
        [
            t0 + rng.normal(0, 0.5, 2000),
            0.5 * t0 + rng.normal(0, 0.5, 2000),  # same cell, weaker
            _transient(rng) + rng.normal(0, 0.5, 2000),  # neighbour, other cell
            t0 + rng.normal(0, 0.5, 2000),  # same trace but far away
        ]
    )
    f = score.unit_flags(A, traces, H, W, CH)
    assert f["duplicate"].tolist() == [False, True, False, False]


def test_empty_and_degenerate_units():
    empty = score.unit_flags(sps.csr_matrix((0, H * W)), np.zeros((0, 100)), H, W, CH)
    assert score.summarize(empty) == {
        "good": 0, "flagged": 0, "score": 0,
        "low_pnr": 0, "in_pieces": 0, "bad_size": 0, "duplicate": 0,
    }
    A = sps.vstack([_footprints([[(10, 10, 2)]]), sps.csr_matrix((1, H * W))]).tocsr()
    f = score.unit_flags(A, _good_traces(2), H, W, CH)  # second row has no pixels
    assert f["good"].tolist() == [True, False]


def test_summarize_counts_reasons():
    f = pd.DataFrame(
        {
            "low_pnr": [False, True, True],
            "in_pieces": [False, False, True],
            "bad_size": [False, False, False],
            "duplicate": [False, False, False],
        }
    )
    f["good"] = ~f[list(score.REASONS)].any(axis=1)
    assert score.summarize(f) == {
        "good": 1, "flagged": 2, "score": -1,
        "low_pnr": 2, "in_pieces": 1, "bad_size": 0, "duplicate": 0,
    }


@pytest.mark.parametrize("d", [0, -3])
def test_checks_reject_bad_diameter(d):
    with pytest.raises(ValueError, match="diameter"):
        score.Checks(diameter=d)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_autotune_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cafast.autotune'`

- [ ] **Step 3: Implement `src/cafast/autotune/__init__.py`**

```python
"""Automatic window-2 parameter proposals: label-free unit checks and a stage-by-stage sweep."""
```

- [ ] **Step 4: Implement `src/cafast/autotune/score.py`**

Move `_row_of_entries`, `size_above`, `_image`, `_mask`, `n_parts`, `trace_pnr` and `SIZE_FRAC` from `src/cafast/tune/figures.py` into this file (rename `_image` → `footprint_image`, `_mask` → `peak_mask`), fix `trace_pnr` for flat traces, and add the checks:

```python
"""Label-free unit checks: each unit is good or flagged with its reasons; score = good - flagged.

Footprint metrics are shared with the ``cafast tune`` figures (``tune/figures.py`` imports them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.spatial import cKDTree

from cafast.init import centroids

SIZE_FRAC = 0.1  # footprint size = pixels above 10 % of peak
REASONS = ("low_pnr", "in_pieces", "bad_size", "duplicate")
_PAIR_CHUNK = 256  # pairs per batch in the duplicate check (bounds memory at long T)

# ---------------------------------------------------------------------------
# Footprint and trace metrics
# ---------------------------------------------------------------------------

# (paste _row_of_entries, size_above, footprint_image (was _image), peak_mask (was _mask)
#  and n_parts here unchanged from tune/figures.py)


def trace_pnr(C: np.ndarray) -> np.ndarray:
    """(K,) peak-to-noise of traces: (99.9th - 0.1th percentile) / noise std from diffs.

    A flat trace scores 0; a trace with range but no diff noise scores inf.
    """
    if C.shape[0] == 0:
        return np.zeros(0)
    lo, hi = np.percentile(C, [0.1, 99.9], axis=1)
    noise = np.median(np.abs(np.diff(C, axis=1)), axis=1) / 0.6745 / np.sqrt(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(noise > 0, (hi - lo) / noise, np.where(hi > lo, np.inf, 0.0))


def main_piece_frac(A: sps.csr_matrix, H: int, W: int, frac: float = SIZE_FRAC) -> np.ndarray:
    """(K,) share of the weight above ``frac`` of peak in the largest 8-connected piece (0 if empty)."""
    A = sps.csr_matrix(A)
    out = np.zeros(A.shape[0])
    for k in range(A.shape[0]):
        img = footprint_image(A, k, H, W)
        n, lab = cv2.connectedComponents(peak_mask(img, frac), connectivity=8)
        if n <= 1:
            continue
        w = np.bincount(lab.ravel(), weights=img.ravel(), minlength=n)[1:]
        out[k] = w.max() / w.sum()
    return out


def _pair_corr(Z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Pearson corr of rows ``i`` and ``j`` of the centred, unit-norm ``Z``, in chunks."""
    out = np.empty(len(i))
    for a in range(0, len(i), _PAIR_CHUNK):
        b = a + _PAIR_CHUNK
        out[a:b] = np.einsum("ij,ij->i", Z[i[a:b]], Z[j[a:b]])
    return out


def duplicates(
    cent: np.ndarray, traces: np.ndarray, pnr: np.ndarray, radius: float, thres: float
) -> np.ndarray:
    """(K,) True where a unit closer than ``radius`` with trace corr >= ``thres`` has a higher PNR.

    Equal PNR: the higher index is flagged. Units with a NaN centroid are never paired.
    """
    K = len(cent)
    dup = np.zeros(K, bool)
    ok = np.flatnonzero(np.isfinite(cent).all(axis=1))
    if len(ok) < 2:
        return dup
    pairs = cKDTree(cent[ok]).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return dup
    i, j = ok[pairs[:, 0]], ok[pairs[:, 1]]
    X = np.asarray(traces, np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Z = np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0)  # flat rows -> corr 0
    hit = _pair_corr(Z, i, j) >= thres
    i, j = i[hit], j[hit]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    dup[np.where(pnr[lo] >= pnr[hi], hi, lo)] = True
    return dup


# ---------------------------------------------------------------------------
# Checks and score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checks:
    """Thresholds of the four unit checks; sizes scale with the cell ``diameter`` (px)."""

    diameter: float
    pnr_min: float = 12.0  # trace_pnr scale: pure white noise scores ~6.2
    piece_frac: float = 0.9
    size_lo: float = 0.3  # x cell area
    size_hi: float = 3.0
    dup_corr: float = 0.8

    def __post_init__(self) -> None:
        if not (isinstance(self.diameter, int | float) and self.diameter > 0):
            raise ValueError(f"diameter must be a number of px > 0, got {self.diameter!r}")

    @property
    def cell_area(self) -> float:
        return math.pi * (self.diameter / 2) ** 2


def unit_flags(
    A: sps.csr_matrix, traces: np.ndarray, H: int, W: int, checks: Checks
) -> pd.DataFrame:
    """One row per unit: metrics, one bool column per reason in ``REASONS``, and ``good``."""
    A = sps.csr_matrix(A)
    K = A.shape[0]
    traces = np.asarray(traces)
    if traces.shape[0] != K:
        raise ValueError(f"{traces.shape[0]} traces for {K} footprints")
    pnr = trace_pnr(traces)
    size10 = size_above(A)
    with np.errstate(invalid="ignore", divide="ignore"):
        cent = centroids(A, H, W) if K else np.zeros((0, 2))
    a = checks.cell_area
    df = pd.DataFrame(
        {
            "unit_id": np.arange(K),
            "pnr": pnr,
            "size10": size10,
            "main_piece": main_piece_frac(A, H, W),
        }
    )
    df["low_pnr"] = pnr < checks.pnr_min
    df["in_pieces"] = df["main_piece"] < checks.piece_frac
    df["bad_size"] = (size10 < checks.size_lo * a) | (size10 > checks.size_hi * a)
    df["duplicate"] = duplicates(cent, traces, pnr, checks.diameter, checks.dup_corr)
    df["good"] = ~df[list(REASONS)].any(axis=1)
    return df


def summarize(flags: pd.DataFrame) -> dict[str, int]:
    """Counts of good and flagged units, the score (good - flagged) and each reason."""
    good = int(flags["good"].sum())
    flagged = len(flags) - good
    return {
        "good": good,
        "flagged": flagged,
        "score": good - flagged,
        **{r: int(flags[r].sum()) for r in REASONS},
    }
```

- [ ] **Step 5: Point `tune/figures.py` at the moved metrics**

Delete the moved definitions and `SIZE_FRAC` from `figures.py`, then add near the other imports:

```python
from cafast.autotune.score import (
    SIZE_FRAC,
    n_parts,
    size_above,
    trace_pnr,
)
from cafast.autotune.score import footprint_image as _image
from cafast.autotune.score import peak_mask as _mask
```

Run `grep -n "_row_of_entries" src/cafast/tune/*.py`. If `figures.py` still uses it, import it from `score` too.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_autotune_score.py tests/test_tune.py -v`
Expected: all PASS (`test_size_above_and_parts` in `test_tune.py` still uses `figures.size_above`).

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/autotune src/cafast/tune/figures.py tests/test_autotune_score.py
git commit -m "Add label-free unit checks (PNR, one piece, cell size, duplicates) for auto-tuning"
```

---

### Task 3: Stage-by-stage sweep (`cafast.autotune.sweep`)

**Files:**
- Create: `src/cafast/autotune/sweep.py`
- Test: `tests/test_autotune_sweep.py`

**Interfaces:**
- Consumes: `pipeline.seeds_unmerged`, `pipeline.seeds_merged`, `pipeline.init_units` (Task 1); `score.Checks`, `score.unit_flags`, `score.summarize`, `score.size_above` (Task 2); `window2.Window2Session` (`.fac`, `.base`, `.factor_corr(noise_freq, timer)`), `window2._norm`; `cnmf.run_cnmf`; `report.StageTimer`.
- Produces:
  - `SEARCHED: tuple[str, ...] = ("seeds_diff_thres", "pnr_thres", "merge_corr", "init_merge_corr", "inita_corr", "sparse_penal", "loop_merge_corr")`
  - `DERIVED: tuple[str, ...] = ("seeds_max_wnd", "inita_wnd", "dl_wnd", "merge_dist", "size_thres")`
  - `STEP_OF: dict[str, str]` (searched key → step name where it varies)
  - `@dataclass(frozen=True) class Grids` (one tuple field per `SEARCHED` key, plus `n_confirm: int = 3`; method `n_trials() -> int`)
  - `derived_params(base: Params, diameter: float) -> Params`
  - `@dataclass class Trial(step: str, values: dict[str, Any], counts: dict[str, int], score: dict[str, int] | None, size10_median: float | None, seconds: float, error: str | None = None)`
  - `ranked(trials: list[Trial], base: Params) -> list[Trial]` (successful trials, best first, distinct values)
  - `best(trials: list[Trial], base: Params, step: str) -> Trial` (raises `RuntimeError` naming `step` if none succeeded)
  - `at_edge(value: Any, grid: tuple) -> bool`
  - `class Sweep(session: Window2Session, checks: Checks, grids: Grids = Grids(), on_trial: Callable[[Trial], None] | None = None)` with `.base: Params`, `.trials: list[Trial]`, `.final: Trial | None`, `.run() -> Params`

- [ ] **Step 1: Write the failing tests** (`tests/test_autotune_sweep.py`)

```python
import dataclasses

import numpy as np
import pytest
import scipy.sparse as sps

from cafast import window2
from cafast.autotune import score, sweep
from cafast.params import Params

D = 7.0  # synthetic cells: Gaussian sigma 3 px
SMALL = sweep.Grids(
    seeds_diff_thres=(2.0, 3.0),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_penal=(0.003, 0.01),
    loop_merge_corr=(0.9,),
    n_confirm=2,
)


def _trial(score_: int | None, **values) -> sweep.Trial:
    sc = None if score_ is None else {"score": score_, "good": score_, "flagged": 0}
    return sweep.Trial("seeds", values, {}, sc, None, 0.0)


def test_derived_params_round_and_scale():
    p = sweep.derived_params(Params(), 7.5)
    assert (p.seeds_max_wnd, p.inita_wnd, p.dl_wnd, p.merge_dist) == (8, 8, 8, 8)
    assert p.size_thres == (11, 177)  # 0.25 and 4 x pi * 3.75^2 = 44.2
    tiny = sweep.derived_params(Params(), 0.6)
    assert tiny.dl_wnd == 1 and tiny.size_thres[0] >= 1
    with pytest.raises(ValueError, match="diameter"):
        sweep.derived_params(Params(), 0)


def test_best_prefers_score_then_fewer_changes_then_order():
    base = Params()
    a = _trial(5, merge_corr=0.9, pnr_thres=base.pnr_thres)
    b = _trial(5, merge_corr=base.merge_corr, pnr_thres=base.pnr_thres)
    c = _trial(7, merge_corr=0.7, pnr_thres=2.0)
    assert sweep.best([a, b], base, "seeds") is b
    assert sweep.best([a, b, c], base, "seeds") is c
    assert sweep.best([_trial(None), a], base, "seeds") is a
    with pytest.raises(RuntimeError, match="seeds"):
        sweep.best([_trial(None)], base, "seeds")


def test_at_edge():
    assert sweep.at_edge(0.002, (0.002, 0.005, 0.01))
    assert not sweep.at_edge(0.005, (0.002, 0.005, 0.01))
    assert sweep.at_edge(1.0, ("auto", 1.0, 2.0))
    assert not sweep.at_edge("auto", ("auto", 1.0, 2.0))
    assert not sweep.at_edge(0.8, (0.8,))  # one-value grid: no edge


def test_sweep_runs_every_step(tiny_run):
    sess = window2.Window2Session(tiny_run)
    seen: list[sweep.Trial] = []
    sw = sweep.Sweep(sess, score.Checks(D), SMALL, on_trial=seen.append)
    prop = sw.run()
    steps = [t.step for t in sw.trials]
    assert steps.count("seeds") == 4
    assert steps.count("merge") == 1 and steps.count("inita") == 1
    assert 1 <= steps.count("confirm") <= 2
    assert steps.count("sparse") == 2 and steps.count("loop_merge") == 1
    assert seen == sw.trials
    assert len(sw.trials) <= SMALL.n_trials()
    assert prop.dl_wnd == 7 and prop.size_thres == sweep.derived_params(sess.base, D).size_thres
    assert prop.sparse_penal in SMALL.sparse_penal
    assert sw.final is not None and sw.final.score is not None
    for k in window2.FACTOR_PARAMS:
        assert getattr(prop, k) == getattr(sess.base, k)


def test_reused_init_gives_the_same_cnmf(tiny_run):
    """The CNMF grid reuses one init A, C: running it twice must not change the result."""
    sess = window2.Window2Session(tiny_run)
    sw = sweep.Sweep(sess, score.Checks(D), SMALL)
    sw.fc = sess.factor_corr(sw.base.noise_freq, sw.timer)
    from cafast.report import StageTimer

    A, C = sw._init(sw.base, StageTimer())
    A0, C0 = A.copy(), C.copy()
    r1 = sw._cnmf(sw.base, StageTimer(), A, C)
    r2 = sw._cnmf(sw.base, StageTimer(), A, C)
    assert np.array_equal(A.toarray(), A0.toarray()) and np.array_equal(C, C0)
    assert np.array_equal(r1[0].toarray(), r2[0].toarray()) and np.array_equal(r1[1], r2[1])


def test_failed_settings_are_recorded_and_skipped(tiny_run):
    sess = window2.Window2Session(tiny_run)
    grids = dataclasses.replace(SMALL, pnr_thres=("auto", 1e9))
    sw = sweep.Sweep(sess, score.Checks(D), grids)
    sw.run()
    failed = [t for t in sw.trials if t.error]
    assert failed and all("no seeds survived" in t.error for t in failed)
    assert all(t.values["pnr_thres"] == 1e9 for t in failed)


def test_all_failed_step_raises(tiny_run):
    sess = window2.Window2Session(tiny_run)
    grids = dataclasses.replace(SMALL, pnr_thres=(1e9,))
    with pytest.raises(RuntimeError, match="seeds"):
        sweep.Sweep(sess, score.Checks(D), grids).run()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_autotune_sweep.py -v`
Expected: FAIL with `ImportError: cannot import name 'sweep'`

- [ ] **Step 3: Implement `src/cafast/autotune/sweep.py`**

```python
"""Stage-by-stage search of the window-2 parameters; every setting tried is a row of the map.

Steps: ``seeds`` (seeds_diff_thres x pnr_thres, scored on the init units), ``merge``
(merge_corr x init_merge_corr from the best seeds, cached after pnr_refine), ``inita``
(inita_corr), ``confirm`` (top init settings through CNMF), ``sparse`` (sparse_penal on the
chosen init), ``loop_merge`` (loop_merge_corr). Factors and FactorCorr are loaded once.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sps

from cafast import corr, pipeline
from cafast.autotune import score
from cafast.cnmf import run_cnmf
from cafast.params import Params
from cafast.report import StageTimer
from cafast.window2 import Window2Session, _norm

SEARCHED = (
    "seeds_diff_thres",
    "pnr_thres",
    "merge_corr",
    "init_merge_corr",
    "inita_corr",
    "sparse_penal",
    "loop_merge_corr",
)
DERIVED = ("seeds_max_wnd", "inita_wnd", "dl_wnd", "merge_dist", "size_thres")
STEP_OF = {
    "seeds_diff_thres": "seeds",
    "pnr_thres": "seeds",
    "merge_corr": "merge",
    "init_merge_corr": "merge",
    "inita_corr": "inita",
    "sparse_penal": "sparse",
    "loop_merge_corr": "loop_merge",
}
INIT_STEPS = ("seeds", "merge", "inita")
FAILED = (ValueError, RuntimeError)  # no seeds survived, non-finite CNMF, singular solves

# ---------------------------------------------------------------------------
# Grids and derived parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grids:
    seeds_diff_thres: tuple[float, ...] = (2.0, 2.5, 3.0, 4.0)
    pnr_thres: tuple[float | str, ...] = ("auto", 1.0, 1.5, 2.0, 3.0)
    merge_corr: tuple[float, ...] = (0.7, 0.75, 0.8, 0.85, 0.9)
    init_merge_corr: tuple[float, ...] = (0.7, 0.8, 0.9)
    inita_corr: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    sparse_penal: tuple[float, ...] = (0.002, 0.003, 0.005, 0.007, 0.01, 0.02)
    loop_merge_corr: tuple[float, ...] = (0.85, 0.9, 0.95)
    n_confirm: int = 3

    def n_trials(self) -> int:
        """Upper bound on the number of settings ``Sweep.run`` tries."""
        return (
            len(self.seeds_diff_thres) * len(self.pnr_thres)
            + len(self.merge_corr) * len(self.init_merge_corr)
            + len(self.inita_corr)
            + self.n_confirm
            + len(self.sparse_penal)
            + len(self.loop_merge_corr)
        )


def derived_params(base: Params, diameter: float) -> Params:
    """``base`` with the size parameters set from the cell ``diameter`` (px)."""
    score.Checks(diameter)  # validates
    d = max(1, round(diameter))
    area = math.pi * (diameter / 2) ** 2
    lo, hi = max(1, round(0.25 * area)), max(2, round(4 * area))
    return dataclasses.replace(
        base, seeds_max_wnd=d, inita_wnd=d, dl_wnd=d, merge_dist=d, size_thres=(lo, hi)
    )


# ---------------------------------------------------------------------------
# Trials and choosing between them
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    step: str
    values: dict[str, Any]  # every SEARCHED key
    counts: dict[str, int]
    score: dict[str, int] | None  # None when the setting failed
    size10_median: float | None
    seconds: float
    error: str | None = None


def _n_changed(values: dict[str, Any], base: Params) -> int:
    return sum(_norm(v) != _norm(getattr(base, k)) for k, v in values.items())


def ranked(trials: list[Trial], base: Params) -> list[Trial]:
    """Successful trials, best first (score, then fewer changes, then order); distinct values."""
    ok = [(i, t) for i, t in enumerate(trials) if t.score is not None]
    ok.sort(key=lambda it: (-it[1].score["score"], _n_changed(it[1].values, base), it[0]))
    out, seen = [], set()
    for _, t in ok:
        key = tuple(sorted((k, str(v)) for k, v in t.values.items()))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def best(trials: list[Trial], base: Params, step: str) -> Trial:
    r = ranked(trials, base)
    if not r:
        raise RuntimeError(f"auto-tune step {step!r}: every setting failed")
    return r[0]


def at_edge(value: Any, grid: tuple) -> bool:
    """True if a numeric ``value`` is the lowest or highest number of a grid with > 1 numbers."""
    nums = sorted(v for v in grid if not isinstance(v, str))
    return not isinstance(value, str) and len(nums) > 1 and value in (nums[0], nums[-1])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

Stage = Callable[[Params, StageTimer], tuple[sps.csr_matrix, np.ndarray]]


@dataclass
class Sweep:
    session: Window2Session
    checks: score.Checks
    grids: Grids = Grids()
    on_trial: Callable[[Trial], None] | None = None
    trials: list[Trial] = field(default_factory=list)
    final: Trial | None = None
    timer: StageTimer = field(default_factory=StageTimer)
    fc: corr.FactorCorr | None = None

    def __post_init__(self) -> None:
        self.base = derived_params(self.session.base, self.checks.diameter)

    # ---- stages --------------------------------------------------------------------

    def _params(self, values: dict[str, Any]) -> Params:
        return dataclasses.replace(self.base, **values)

    def _init(
        self, p: Params, timer: StageTimer, s_pnr: pd.DataFrame | None = None
    ) -> tuple[sps.csr_matrix, np.ndarray]:
        fac = self.session.fac
        if s_pnr is None:
            s_pnr = pipeline.seeds_unmerged(fac, p, timer)
        s = pipeline.seeds_merged(fac, self.fc, s_pnr, p, timer)
        return pipeline.init_units(fac, self.fc, s, p, timer)

    def _cnmf(
        self, p: Params, timer: StageTimer, A: sps.csr_matrix, C: np.ndarray
    ) -> tuple[sps.csr_matrix, np.ndarray]:
        res = run_cnmf(self.session.fac, A, C, p, self.fc.traces_corr)
        timer.counts.update(res.counts)
        timer.counts["units_final"] = res.A.shape[0]
        return res.A, res.YrA

    def _full(self, p: Params, timer: StageTimer) -> tuple[sps.csr_matrix, np.ndarray]:
        return self._cnmf(p, timer, *self._init(p, timer))

    # ---- one setting ---------------------------------------------------------------

    def _try(self, step: str, values: dict[str, Any], stage: Stage) -> Trial:
        t0, timer = time.perf_counter(), StageTimer()
        try:
            A, traces = stage(self._params(values), timer)
        except FAILED as e:
            sc, med, err = None, None, f"{type(e).__name__}: {e}"
        else:
            fac = self.session.fac
            flags = score.unit_flags(A, traces, fac.H, fac.W_, self.checks)
            sc, err = score.summarize(flags), None
            med = float(np.median(flags["size10"])) if len(flags) else None
        t = Trial(step, dict(values), dict(timer.counts), sc, med, time.perf_counter() - t0, err)
        self.trials.append(t)
        if self.on_trial is not None:
            self.on_trial(t)
        return t

    def _step(self, *steps: str) -> list[Trial]:
        return [t for t in self.trials if t.step in steps]

    # ---- the search ----------------------------------------------------------------

    def run(self) -> Params:
        """Run every step; returns the proposed parameters (``self.final`` is its trial)."""
        g, base = self.grids, self.base
        self.fc = self.session.factor_corr(base.noise_freq, self.timer)
        v = {k: getattr(base, k) for k in SEARCHED}
        for diff, pnr in product(g.seeds_diff_thres, g.pnr_thres):
            self._try("seeds", {**v, "seeds_diff_thres": diff, "pnr_thres": pnr}, self._init)
        v = best(self._step("seeds"), base, "seeds").values
        s_pnr = pipeline.seeds_unmerged(self.session.fac, self._params(v), StageTimer())

        def cached(p: Params, t: StageTimer) -> tuple[sps.csr_matrix, np.ndarray]:
            return self._init(p, t, s_pnr)

        for mc, imc in product(g.merge_corr, g.init_merge_corr):
            self._try("merge", {**v, "merge_corr": mc, "init_merge_corr": imc}, cached)
        v = best(self._step("merge"), base, "merge").values
        for ic in g.inita_corr:
            self._try("inita", {**v, "inita_corr": ic}, cached)
        for t in ranked(self._step(*INIT_STEPS), base)[: g.n_confirm]:
            self._try("confirm", t.values, self._full)
        v = best(self._step("confirm"), base, "confirm").values
        A, C = self._init(self._params(v), StageTimer())

        def on_init(p: Params, t: StageTimer) -> tuple[sps.csr_matrix, np.ndarray]:
            return self._cnmf(p, t, A, C)

        for sp in g.sparse_penal:
            self._try("sparse", {**v, "sparse_penal": sp}, on_init)
        top = best(self._step("sparse"), base, "sparse")
        for lm in g.loop_merge_corr:
            self._try("loop_merge", {**top.values, "loop_merge_corr": lm}, on_init)
        self.final = best([top, *self._step("loop_merge")], base, "loop_merge")
        return self._params(self.final.values)
```

Note: `Sweep` is a dataclass so `grids: Grids = Grids()` is a frozen default (safe). `self.base` is set in `__post_init__`; add `base: Params = field(init=False)` above `trials` so type checkers and `dataclasses.fields` know it.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_autotune_sweep.py -v`
Expected: all PASS. If `test_failed_settings_are_recorded_and_skipped` fails because a non-`ValueError` is raised for an empty seed set further down (for example in `initA`), add that exception type to `FAILED` only if it is a data condition (not a bug) and say why in a comment.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/autotune/sweep.py tests/test_autotune_sweep.py
git commit -m "Add the stage-by-stage window-2 sweep that scores each setting with the unit checks"
```

---

### Task 4: Outputs and the `cafast autotune` command

**Files:**
- Create: `src/cafast/autotune/run.py`
- Modify: `src/cafast/cli.py` (new command after `cnmf`)
- Test: `tests/test_autotune_sweep.py` (append)

**Interfaces:**
- Consumes: Tasks 2–3; `window2.Window2Session.run(params, name) -> Path`.
- Produces:
  - `run.AUTO = "auto"`, `run.DEFAULT_GRIDS: Grids` (module global so tests and the page can shrink it)
  - `run.sweep_table(trials: list[Trial]) -> pd.DataFrame` with columns `step, <SEARCHED...>, good, flagged, score, low_pnr, in_pieces, bad_size, duplicate, seeds_init, seeds_pnr, seeds_merged, units_init_merge, units_final, size10_median, seconds, error`
  - `run.autotune(run_dir: Path, diameter: float, *, checks: Checks | None = None, grids: Grids | None = None, name: str = AUTO, session: Window2Session | None = None, on_trial: Callable[[Trial], None] | None = None) -> Path`
  - `run.load_doc(rerun_dir: Path) -> dict | None` (reads `autotune.json`)
  - `run.checks_of(doc: dict) -> Checks`
  - Files in the rerun dir: `sweep.csv`, `unit_flags.csv`, `autotune.json` with keys `diameter, checks, grids, derived, proposed, run_values, at_edge, score, n_trials, wall_s`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_autotune_sweep.py`)

```python
import json

import pandas as pd
from typer.testing import CliRunner

from cafast import cli
from cafast.autotune import run as at


def test_autotune_writes_a_rerun_whose_score_matches(tiny_run):
    out = at.autotune(tiny_run, D, grids=SMALL, name="auto_t")
    assert out == tiny_run / "reruns" / "auto_t"
    doc = json.loads((out / "autotune.json").read_text())
    assert doc["diameter"] == D and set(doc["proposed"]) == set(sweep.SEARCHED)
    assert doc["derived"]["dl_wnd"] == 7
    assert set(doc["at_edge"]) <= set(sweep.SEARCHED)
    # the cached sweep and the fresh rerun agree
    A = sps.load_npz(out / "cnmf" / "A.npz")
    YrA = np.load(out / "cnmf" / "YrA.npy")
    fresh = score.summarize(score.unit_flags(A, YrA, 64, 64, at.checks_of(doc)))
    assert fresh == doc["score"]
    table = pd.read_csv(out / "sweep.csv")
    assert len(table) == doc["n_trials"] and {"step", "score", "sparse_penal"} <= set(table)
    flags = pd.read_csv(out / "unit_flags.csv")
    assert len(flags) == A.shape[0] and "good" in flags


def test_cli_autotune(tiny_run, monkeypatch):
    monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL)
    res = CliRunner().invoke(
        cli.app, ["autotune", str(tiny_run), "--diameter", "7", "--name", "auto_cli"]
    )
    assert res.exit_code == 0, res.output
    assert "sparse_penal" in res.output and "good" in res.output
    assert (tiny_run / "reruns" / "auto_cli" / "autotune.json").exists()


@pytest.mark.parametrize("d", ["0", "-2"])
def test_cli_autotune_rejects_bad_diameter(tiny_run, d):
    res = CliRunner().invoke(cli.app, ["autotune", str(tiny_run), "--diameter", d])
    assert res.exit_code == 1 and "diameter" in res.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_autotune_sweep.py -k "autotune" -v`
Expected: FAIL with `ImportError: cannot import name 'run'`

- [ ] **Step 3: Implement `src/cafast/autotune/run.py`**

```python
"""``cafast autotune``: sweep, then write the proposal as ``RUN_DIR/reruns/NAME/``.

Besides the normal rerun files the directory gets ``sweep.csv`` (one row per setting tried),
``unit_flags.csv`` (the checks on the proposal's final units) and ``autotune.json``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sps

from cafast.autotune import score
from cafast.autotune.sweep import DERIVED, SEARCHED, Grids, Sweep, Trial, at_edge
from cafast.window2 import Window2Session

AUTO = "auto"
DEFAULT_GRIDS = Grids()
DOC = "autotune.json"
COUNT_COLS = ("seeds_init", "seeds_pnr", "seeds_merged", "units_init_merge", "units_final")
SCORE_COLS = ("good", "flagged", "score", *score.REASONS)


def sweep_table(trials: list[Trial]) -> pd.DataFrame:
    rows = []
    for t in trials:
        sc = t.score or {}
        rows.append(
            {
                "step": t.step,
                **{k: t.values[k] for k in SEARCHED},
                **{k: sc.get(k) for k in SCORE_COLS},
                **{k: t.counts.get(k) for k in COUNT_COLS},
                "size10_median": t.size10_median,
                "seconds": round(t.seconds, 2),
                "error": t.error,
            }
        )
    return pd.DataFrame(rows)


def _json_value(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def checks_of(doc: dict[str, Any]) -> score.Checks:
    return score.Checks(**doc["checks"])


def load_doc(rerun_dir: Path) -> dict[str, Any] | None:
    p = Path(rerun_dir) / DOC
    return json.loads(p.read_text()) if p.exists() else None


def autotune(
    run_dir: Path,
    diameter: float,
    *,
    checks: score.Checks | None = None,
    grids: Grids | None = None,
    name: str = AUTO,
    session: Window2Session | None = None,
    on_trial: Callable[[Trial], None] | None = None,
) -> Path:
    """Sweep window 2 of ``run_dir`` and write the proposal to ``run_dir/reruns/name``."""
    t0 = time.perf_counter()
    checks = checks or score.Checks(diameter)
    grids = grids or DEFAULT_GRIDS
    session = session or Window2Session(run_dir)
    sw = Sweep(session, checks, grids, on_trial)
    params = sw.run()
    out = session.run(params, name)
    sweep_table(sw.trials).to_csv(out / "sweep.csv", index=False)
    A = sps.load_npz(out / "cnmf" / "A.npz")
    YrA = np.load(out / "cnmf" / "YrA.npy")
    flags = score.unit_flags(A, YrA, session.fac.H, session.fac.W_, checks)
    flags.to_csv(out / "unit_flags.csv", index=False)
    doc = {
        "diameter": diameter,
        "checks": asdict(checks),
        "grids": {k: _json_value(v) for k, v in asdict(grids).items()},
        "derived": {k: _json_value(getattr(params, k)) for k in DERIVED},
        "proposed": {k: getattr(params, k) for k in SEARCHED},
        "run_values": {k: _json_value(getattr(session.base, k)) for k in SEARCHED + DERIVED},
        "at_edge": [k for k in SEARCHED if at_edge(getattr(params, k), getattr(grids, k))],
        "score": score.summarize(flags),
        "n_trials": len(sw.trials),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / DOC).write_text(json.dumps(doc, indent=1))
    return out
```

- [ ] **Step 4: Add the CLI command** (in `src/cafast/cli.py`, after `cnmf`)

```python
@app.command()
def autotune(
    run_dir: Annotated[Path, typer.Argument(help="A finished run (holds factors/).")],
    diameter: Annotated[float, typer.Option(help="Cell diameter in px (factor frame).")],
    name: Annotated[str, typer.Option(help="Output name under RUN_DIR/reruns/.")] = "auto",
) -> None:
    """Propose window-2 parameters for RUN_DIR (sweep, then a rerun in RUN_DIR/reruns/NAME/)."""
    import json

    from cafast.autotune import run as at
    from cafast.autotune.sweep import Trial

    def show(t: Trial) -> None:
        res = t.error if t.score is None else f"good {t.score['good']}  flagged {t.score['flagged']}"
        typer.echo(f"{t.step:<10} {res}  ({t.seconds:.1f} s)")

    try:
        out = at.autotune(run_dir, diameter, name=name, on_trial=show)
    except (ValueError, RuntimeError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    doc = json.loads((out / at.DOC).read_text())
    typer.echo(str(out))
    for k, v in doc["proposed"].items():
        edge = "  [at edge]" if k in doc["at_edge"] else ""
        typer.echo(f"{k:<17} {doc['run_values'][k]!s:>8} -> {v!s:<8}{edge}")
    s = doc["score"]
    typer.echo(f"good {s['good']}  flagged {s['flagged']}  ({doc['wall_s']} s)")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_autotune_sweep.py -v`
Expected: all PASS. (`CliRunner` mixes stderr into `output` by default; if the installed typer keeps them apart, assert on `res.stderr` for the bad-diameter test.)

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/autotune/run.py src/cafast/cli.py tests/test_autotune_sweep.py
git commit -m "Add cafast autotune, which writes the proposal, the sweep map and unit flags as a rerun"
```

---

### Task 5: Tune page: propose, proposal panel, scores in the run list

**Files:**
- Create: `src/cafast/tune/autofig.py`
- Modify: `src/cafast/tune/figures.py` (`summary`, `make_run_figures`), `src/cafast/tune/server.py`, `src/cafast/tune/page.py`, `src/cafast/tune/static/tune.js`, `src/cafast/tune/static/tune.css`
- Test: `tests/test_tune.py` (append)

**Interfaces:**
- Consumes: `run.autotune`, `run.load_doc`, `run.checks_of`, `run.AUTO`, `run.DEFAULT_GRIDS`, `sweep.SEARCHED`, `sweep.STEP_OF`, `score.unit_flags`, `score.summarize`.
- Produces:
  - `autofig.best_per_value(df: pd.DataFrame, key: str) -> pd.DataFrame` (one row per value of `key` in its step: the best-scoring trial at that value, in grid order)
  - `autofig.sweep_curves(sweep_csv: Path, doc: dict, out_dir: Path) -> list[str]` (writes `sweep_<key>.png` per searched key, returns the file names)
  - `figures.summary(run, mn, checks: Checks | None = None)` adds `good`, `flagged` when `checks` is given
  - `figures.make_run_figures(run, mn, checks: Checks | None = None)` also draws the sweep curves when `run.dir / "sweep.csv"` exists
  - `TuneApp.checks() -> Checks | None` (from `reruns/auto/autotune.json`), `TuneApp.start_autotune(diameter: float) -> str`
  - route `POST /api/autotune` (form `diameter`) → `{"name": "auto"}`, 400 on a bad diameter, 409 when busy

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tune.py`)

```python
from cafast.autotune import run as at  # noqa: E402
from cafast.autotune import sweep  # noqa: E402

SMALL_GRIDS = sweep.Grids(
    seeds_diff_thres=(2.0, 3.0),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_penal=(0.003, 0.01),
    loop_merge_corr=(0.9,),
    n_confirm=2,
)


def test_best_per_value_and_curves(tmp_path):
    from cafast.tune import autofig

    df = pd.DataFrame(
        {
            "step": ["seeds"] * 4 + ["sparse"] * 2,
            "seeds_diff_thres": [2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            "pnr_thres": ["auto", "1.0", "auto", "1.0", "auto", "auto"],
            "sparse_penal": [0.01] * 4 + [0.003, 0.01],
            "score": [3, 5, 4, None, 6, 2],
            "good": [5, 6, 5, None, 7, 4],
            "flagged": [2, 1, 1, None, 1, 2],
        }
    )
    b = autofig.best_per_value(df, "seeds_diff_thres")
    assert b["seeds_diff_thres"].tolist() == [2.0, 3.0] and b["score"].tolist() == [5, 4]
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    doc = {"proposed": {"seeds_diff_thres": 2.0, "pnr_thres": "1.0", "sparse_penal": 0.003}}
    names = autofig.sweep_curves(csv, doc, tmp_path)
    assert names == ["sweep_seeds_diff_thres.png", "sweep_pnr_thres.png", "sweep_sparse_penal.png"]
    assert (tmp_path / names[0]).read_bytes()[:4] == b"\x89PNG"


def test_server_autotune_roundtrip(tune_server, monkeypatch):
    app, base = tune_server
    _wait(base)
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/autotune", {"diameter": "abc"})
    assert e.value.code == 400
    monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL_GRIDS)
    assert _post(base + "/api/autotune", {"diameter": "7"})["name"] == "auto"
    st = _wait(base, timeout=300)
    assert st["state"] == "done", st
    assert app.checks() is not None and app.checks().diameter == 7
    page_html = _get(base + "/?run=auto")[2].decode()
    assert "Proposal" in page_html and "sweep_sparse_penal.png" in page_html
    assert "good / flagged" in page_html
    status, ctype, png = _get(base + "/fig/auto/sweep_sparse_penal.png")
    assert status == 200 and png[:4] == b"\x89PNG"
    summ = json.loads((app.reruns_dir / "base" / "summary.json").read_text())
    assert "good" in summ and "flagged" in summ  # every run rescored with the new checks
```

Also add `import pandas as pd` and `import urllib.error` to the imports at the top of `tests/test_tune.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tune.py -k "curves or autotune" -v`
Expected: FAIL with `ImportError: cannot import name 'autofig'`

- [ ] **Step 3: Implement `src/cafast/tune/autofig.py` (sweep curves)**

```python
"""Figures for auto-tune: one small sweep curve per searched parameter (more added in Task 6-7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cafast.autotune.sweep import SEARCHED, STEP_OF

GOOD, FLAGGED, CHOSEN = "#4cc9f0", "#f72585", "#ffd166"
DPI = 110


def _order(values: pd.Series) -> list[Any]:
    """Grid order: ``auto`` first, then numbers ascending."""
    uniq = list(dict.fromkeys(values.tolist()))
    words = [v for v in uniq if isinstance(v, str) and not _is_num(v)]
    nums = sorted((v for v in uniq if v not in words), key=float)
    return words + nums


def _is_num(v: Any) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def best_per_value(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per value of ``key`` in its step: the best-scoring successful trial, in grid order."""
    d = df[(df["step"] == STEP_OF[key]) & df["score"].notna()]
    if d.empty:
        return d
    top = d.sort_values("score", ascending=False, kind="stable").drop_duplicates(key)
    order = {v: i for i, v in enumerate(_order(top[key]))}
    return top.sort_values(key, key=lambda s: s.map(order)).reset_index(drop=True)


def _curve(df: pd.DataFrame, key: str, chosen: Any, path: Path) -> None:
    b = best_per_value(df, key)
    fig = Figure(figsize=(2.8, 1.5), dpi=DPI, facecolor="#111")
    ax = fig.add_subplot()
    ax.set_facecolor("#111")
    x = np.arange(len(b))
    ax.plot(x, b["good"], "-o", color=GOOD, ms=3, lw=1.2, label="good")
    ax.plot(x, b["flagged"], "-o", color=FLAGGED, ms=3, lw=1.2, label="flagged")
    labels = [str(v) for v in b[key]]
    if str(chosen) in labels:
        ax.axvline(labels.index(str(chosen)), color=CHOSEN, lw=1, ls="--")
    ax.set_xticks(x, labels, fontsize=6, color="#ccc")
    ax.tick_params(axis="y", labelsize=6, colors="#ccc")
    for s in ax.spines.values():
        s.set_color("#444")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, facecolor=fig.get_facecolor())


def sweep_curves(sweep_csv: Path, doc: dict[str, Any], out_dir: Path) -> list[str]:
    """Write ``sweep_<key>.png`` for every searched key that has rows; returns the file names."""
    df = pd.read_csv(sweep_csv)
    names = []
    for key in SEARCHED:
        if key not in df or best_per_value(df, key).empty:
            continue
        name = f"sweep_{key}.png"
        _curve(df, key, doc["proposed"].get(key), Path(out_dir) / name)
        names.append(name)
    return names
```

(`pnr_thres` comes back from the CSV as strings like `"auto"` / `"1.0"`; `_order` and `str(chosen)` handle both. Match the colors to the existing figure palette in `figures.py`, e.g. `CAFAST`, if those are defined there.)

- [ ] **Step 4: Summaries and figures with checks** (`src/cafast/tune/figures.py`)

```python
def summary(
    run: RunData, mn: MinianOverlay | None, checks: Checks | None = None
) -> dict[str, Any]:
    ...  # existing body
    if checks is not None:
        s = summarize(unit_flags(run.A, run.YrA, run.H, run.W, checks))
        out |= {"good": s["good"], "flagged": s["flagged"]}
    return out


def make_run_figures(
    run: RunData, mn: MinianOverlay | None, checks: Checks | None = None
) -> dict[str, Any]:
    ...  # existing figure calls
    sweep_csv, doc = run.dir / "sweep.csv", load_doc(run.dir)
    if sweep_csv.exists() and doc is not None:
        autofig.sweep_curves(sweep_csv, doc, out)
    s = summary(run, mn, checks)
    (run.dir / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def write_summary(run: RunData, mn: MinianOverlay | None, checks: Checks | None) -> None:
    """Rewrite ``summary.json`` only (after new checks), without drawing figures."""
    (run.dir / "summary.json").write_text(json.dumps(summary(run, mn, checks), indent=1))
```

Imports: `from cafast.autotune.score import Checks, summarize, unit_flags`, `from cafast.autotune.run import load_doc`, `from cafast.tune import autofig`. If importing `cafast.autotune.run` from `figures` causes a cycle (it imports `window2`, not `tune`, so it should not), move `load_doc` into `score.py`.

- [ ] **Step 5: Server** (`src/cafast/tune/server.py`)

Add to `TuneApp`:

```python
    def checks(self) -> Checks | None:
        """The unit checks of the last auto-tune (``reruns/auto/autotune.json``), if any."""
        doc = at.load_doc(self.reruns_dir / at.AUTO)
        return at.checks_of(doc) if doc is not None else None

    def start_autotune(self, diameter: float) -> str:
        Checks(diameter)  # ValueError -> HTTP 400 before the worker starts
        self._start(self._autotune, at.AUTO, diameter)
        return at.AUTO

    def _autotune(self, name: str, diameter: float) -> None:
        n = at.DEFAULT_GRIDS.n_trials()
        done: list[object] = []

        def progress(t: Trial) -> None:
            done.append(t)
            self._set(stage=f"{t.step}: setting {len(done)} of ~{n}")

        at.autotune(self.run_dir, diameter, name=name, session=self.session, on_trial=progress)
        self._figures(name)
        self._rescore()

    def _rescore(self) -> None:
        """Rewrite every run's summary with the current checks (good / flagged columns)."""
        self._set(stage="rescoring runs")
        checks = self.checks()
        for d in sorted(self.reruns_dir.glob("*")):
            if (d / "report.json").exists() and (d / "stages" / "index.json").exists():
                run = self.run_data(d.name)
                figures.write_summary(run, self.overlay(run), checks)
```

Change `_figures` to pass `self.checks()` into `make_run_figures`. In `_startup`, also treat a summary as stale when `self.checks()` is not `None` and `'"good"'` is not in its text. In `do_POST` add:

```python
            elif self.path == "/api/autotune":
                try:
                    d = float(form.get("diameter", ""))
                except ValueError:
                    raise ValueError("diameter: expected a number of px") from None
                self._json({"name": app.start_autotune(d)})
```

Imports: `from cafast.autotune import run as at`, `from cafast.autotune.score import Checks`, `from cafast.autotune.sweep import Trial`. Update the module docstring's route list.

- [ ] **Step 6: Page** (`src/cafast/tune/page.py`)

Sidebar, at the very top of `_sidebar` (before the run form):

```python
def _propose_form(app: TuneApp) -> str:
    ch = app.checks()
    d = "" if ch is None else fmt_value(ch.diameter)
    return (
        '<form id="atform"><h2>Auto-tune</h2>'
        '<label><b>cell diameter (px)</b><input type="text" name="diameter" '
        f'value="{_e(d)}" placeholder="e.g. 8"><span class="hint">in the factor frame; '
        "sets dl_wnd, merge_dist, seeds_max_wnd, inita_wnd, size_thres</span></label>"
        '<button type="submit">Propose parameters (~5 min)</button></form>'
    )
```

Main column, before the runs table:

```python
def _proposal(app: TuneApp) -> str:
    doc = at.load_doc(app.reruns_dir / at.AUTO)
    if doc is None:
        return ""
    s = doc["score"]
    reasons = ", ".join(f"{s[r]} {r.replace('_', ' ')}" for r in score.REASONS if s[r])
    rows = []
    for k in sweep.SEARCHED:
        a, b = doc["run_values"][k], doc["proposed"][k]
        cls = ' class="changed"' if str(a) != str(b) else ""
        edge = '<span class="badge">at edge: extend the grid</span>' if k in doc["at_edge"] else ""
        rows.append(
            f"<tr><td class='l'><b>{k}</b></td><td>{_e(fmt_value(a))}</td>"
            f"<td{cls}>{_e(fmt_value(b))}</td><td>{_img(app, at.AUTO, f'sweep_{k}.png')}</td>"
            f"<td class='l'>{edge}</td></tr>"
        )
    derived = ", ".join(f"{k} {fmt_value(v)}" for k, v in doc["derived"].items())
    return (
        f"<h2>Proposal ({at.AUTO})</h2><div class='note'>{s['good']} good / {s['flagged']} "
        f"flagged ({_e(reasons) or 'none'}) · diameter {_e(fmt_value(doc['diameter']))} px · "
        f"derived: {_e(derived)} · {doc['n_trials']} settings in {doc['wall_s']} s</div>"
        "<div class='note'>Curves: best good (blue) and flagged (red) count at each value; "
        "dashed = proposed. To override: view auto (its values fill the form), edit, Rerun; "
        "the new run opens compared with auto.</div>"
        "<table class='proposal'><thead><tr><th class='l'>parameter</th><th>run</th>"
        f"<th>proposed</th><th>sweep</th><th></th></tr></thead><tbody>{''.join(rows)}"
        "</tbody></table>"
    )
```

In `render`, prepend `_proposal(app)` to `main`. In `_runs_table`, when `app.checks()` is not `None`, add a header `"good / flagged"` after `"units"` and the cell `f"{_num(r.get('good'))} / {_num(r.get('flagged'))}"` at the same position. Keep the header/cell index arithmetic for the Minian columns right by building both lists in one place. Imports: `from cafast.autotune import run as at, score, sweep`.

- [ ] **Step 7: JS and CSS**

`tune.js`, after the `#runform` handler:

```js
$('#atform').addEventListener('submit', async e => { e.preventDefault();
  try { const j = await post('/api/autotune', e.target); say('auto-tune started'); poll(j.name, true); }
  catch(err){ say(err.message, true); } });
```

`tune.css`: `table.proposal img { width: 260px; }`, `td.changed { color: #ffd166; font-weight: 600; }`, `.badge { background: #f7258533; color: #f72585; padding: 0 4px; border-radius: 3px; font-size: 11px; }`.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tune.py -v`
Expected: all PASS.

- [ ] **Step 9: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/tune tests/test_tune.py
git commit -m "Show the auto-tune proposal with sweep curves and good/flagged counts on the tune page"
```

---

### Task 6: Flagged-units gallery and a flag column in the units table

**Files:**
- Modify: `src/cafast/tune/autofig.py`, `src/cafast/tune/figures.py` (`unit_table`), `src/cafast/tune/server.py` (route), `src/cafast/tune/page.py`, `static/tune.js`, `static/tune.css`
- Test: `tests/test_tune.py` (append)

**Interfaces:**
- Produces:
  - `autofig.thumb_png(run: RunData, k: int, pad: int = 12) -> bytes`: a crop of the max projection around unit `k`, upscaled 4× (nearest), with the footprint outline at 30 % of its peak; OpenCV only (fast)
  - `figures.unit_table(run, mn, checks: Checks | None = None)` adds a `flag` column (reasons joined by `", "`, empty when good)
  - route `GET /thumb/RUN/K.png`
  - page `_flagged(app, view) -> str`

- [ ] **Step 1: Write the failing tests**

```python
def test_thumb_and_flag_column(tiny_rerun):
    from cafast.autotune.score import Checks
    from cafast.tune import autofig

    run = figures.load_run(tiny_rerun)
    png = autofig.thumb_png(run, 0)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] == img.shape[1] == 4 * (2 * 12 + 1)
    table = figures.unit_table(run, None, Checks(7))
    assert "flag" in table and len(table) == run.A.shape[0]


def test_page_shows_flagged_gallery(tune_server, monkeypatch):
    app, base = tune_server
    _wait(base)
    if app.checks() is None:  # test order: run auto-tune if the roundtrip test has not
        monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL_GRIDS)
        _post(base + "/api/autotune", {"diameter": "7"})
        _wait(base, timeout=300)
    page_html = _get(base + "/?run=auto")[2].decode()
    assert "Flagged units" in page_html and "<th>flag</th>" in page_html
    status, ctype, png = _get(base + "/thumb/auto/0.png")
    assert status == 200 and ctype == "image/png"
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/thumb/auto/99999.png")
    assert e.value.code == 404
```

(Add `import cv2` to the test imports. The 4 × 25 px shape assumes the crop is padded at the frame border, see Step 3.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tune.py -k "thumb or gallery" -v`
Expected: FAIL with `AttributeError: module 'cafast.tune.autofig' has no attribute 'thumb_png'`

- [ ] **Step 3: Implement `thumb_png`** (append to `autofig.py`)

```python
import cv2

from cafast.autotune.score import footprint_image, peak_mask
from cafast.init import centroids

THUMB_SCALE = 4
OUTLINE = 0.3


def _gray8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def thumb_png(run: Any, k: int, pad: int = 12) -> bytes:
    """Max-projection crop (2·pad+1 px, edge-padded) around unit ``k`` with its outline, 4x."""
    cy, cx = (int(round(v)) for v in centroids(run.A[k], run.H, run.W)[0])
    big = np.pad(_gray8(run.max_proj), pad, mode="constant")
    fp = np.pad(peak_mask(footprint_image(run.A, k, run.H, run.W), OUTLINE), pad)
    sl = np.s_[cy : cy + 2 * pad + 1, cx : cx + 2 * pad + 1]
    s = THUMB_SCALE
    crop = cv2.resize(big[sl], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(fp[sl], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cs, -1, (240, 201, 76), 1)  # BGR of #4cc9f0
    return cv2.imencode(".png", rgb)[1].tobytes()
```

(A footprint with no pixels has a NaN centroid. `_flagged` never asks for a thumbnail of such a unit; it lists it as text instead, see Step 5.)

- [ ] **Step 4: `unit_table` flag column** (`figures.py`)

```python
def unit_table(
    run: RunData, mn: MinianOverlay | None, checks: Checks | None = None
) -> pd.DataFrame:
    ...  # existing body up to the Minian block
    if checks is not None:
        f = unit_flags(run.A, run.YrA, run.H, run.W, checks)
        df["flag"] = [", ".join(r for r in REASONS if row[r]) for _, row in f.iterrows()]
    ...
```

- [ ] **Step 5: Route, gallery, JS**

`server.py` `do_GET`: add a branch next to `/unit/` that parses `parts[2]` with the same regex, checks `int(m[1]) < run.A.shape[0]`, and sends `autofig.thumb_png(run, k)` as `image/png`.

`page.py`:

```python
LABELS = {
    "low_pnr": "dim trace (PNR)",
    "in_pieces": "in pieces",
    "bad_size": "wrong size",
    "duplicate": "duplicate",
}


def _flagged(app: TuneApp, view: str) -> str:
    checks = app.checks()
    if checks is None:
        return ""
    run = app.run_data(view)
    f = score.unit_flags(run.A, run.YrA, run.H, run.W, checks)
    groups = []
    for r in score.REASONS:
        ks = f.index[f[r]].tolist()
        if not ks:
            continue
        imgs = "".join(
            f'<img loading="lazy" class="thumb" data-k="{k}" src="/thumb/{_e(view)}/{k}.png" '
            f'title="unit {k}: pnr {f.pnr[k]:.1f}, size10 {f.size10[k]}, '
            f'main piece {f.main_piece[k]:.2f}">'
            if f.size10[k] > 0
            else f'<span class="note">unit {k} (empty)</span>'
            for k in ks
        )
        groups.append(f"<h4>{LABELS[r]} ({len(ks)})</h4><div class='thumbs'>{imgs}</div>")
    n = int((~f.good).sum())
    return (
        f"<div class='fig'><h3>Flagged units ({n} of {len(f)})</h3><div class='why'>checks of "
        f"the last auto-tune (diameter {fmt_value(checks.diameter)} px); click a unit to show "
        f"it below</div>{''.join(groups)}</div>"
    )
```

In `render`, insert `_flagged(app, view)` before `_units(app, view)`, and pass `app.checks()` into `figures.unit_table` inside `_units`.

`tune.js`:

```js
document.querySelectorAll('img.thumb').forEach(im => im.addEventListener('click', () => {
  $('#unitimg').src = '/unit/' + encodeURIComponent(VIEW) + '/' + im.dataset.k + '.png'; $('#unitimg').hidden = false;
  $('#unitimg').scrollIntoView({block: 'nearest'}); }));
```

`tune.css`: `.thumbs { display: flex; flex-wrap: wrap; gap: 4px; } img.thumb { width: 100px; cursor: pointer; image-rendering: pixelated; }`.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tune.py -v`
Expected: all PASS.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/tune tests/test_tune.py
git commit -m "Add a flagged-units gallery and a flag column to the tune page"
```

---

### Task 7: Measure the cell diameter on the page

**Files:**
- Modify: `src/cafast/tune/autofig.py`, `src/cafast/tune/server.py`, `src/cafast/tune/page.py` (`_propose_form`), `static/tune.js`, `static/tune.css`
- Test: `tests/test_tune.py` (append)

**Interfaces:**
- Produces: `autofig.max_proj_png(max_proj: np.ndarray) -> bytes` (8-bit grayscale at native resolution, 1 image px = 1 factor px); route `GET /maxproj.png`.

- [ ] **Step 1: Write the failing test**

```python
def test_maxproj_png_native_size(tune_server):
    app, base = tune_server
    status, ctype, png = _get(base + "/maxproj.png")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert status == 200 and ctype == "image/png"
    assert img.shape == (app.session.fac.H, app.session.fac.W_) and img.dtype == np.uint8
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tune.py::test_maxproj_png_native_size -v`
Expected: FAIL with HTTP 404.

- [ ] **Step 3: Implement**

`autofig.py`:

```python
def max_proj_png(max_proj: np.ndarray) -> bytes:
    """The max projection as an 8-bit PNG, one image pixel per factor pixel."""
    return cv2.imencode(".png", _gray8(np.asarray(max_proj, np.float32)))[1].tobytes()
```

`server.py` `do_GET`: `elif url.path == "/maxproj.png": self._send(200, "image/png", autofig.max_proj_png(app.session.fac.max_proj))`.

`page.py` `_propose_form`: after the diameter label add

```python
        '<div id="measure"><img id="mpimg" src="/maxproj.png" draggable="false">'
        '<svg id="mpline"><line x1="0" y1="0" x2="0" y2="0"/></svg></div>'
        '<span class="hint">drag across one cell to measure its diameter</span>'
```

`tune.js`:

```js
(() => { const im = $('#mpimg'), ln = $('#mpline line'); if(!im) return; let a = null;
  const pt = e => { const r = im.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const set = (p, q) => { ln.setAttribute('x1', p[0]); ln.setAttribute('y1', p[1]); ln.setAttribute('x2', q[0]); ln.setAttribute('y2', q[1]); };
  im.addEventListener('mousedown', e => { a = pt(e); set(a, a); });
  im.addEventListener('mousemove', e => { if(a) set(a, pt(e)); });
  window.addEventListener('mouseup', e => { if(!a) return; const b = pt(e);
    const px = Math.hypot(b[0] - a[0], b[1] - a[1]) * im.naturalWidth / im.clientWidth;
    if(px >= 1) $('#atform [name=diameter]').value = px.toFixed(1); a = null; });
})();
```

`tune.css`: `#measure { position: relative; } #mpimg { width: 100%; image-rendering: pixelated; cursor: crosshair; display: block; } #mpline { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; } #mpline line { stroke: #ffd166; stroke-width: 2; }`.

- [ ] **Step 4: Run the tests, then check the page by hand**

Run: `.venv/bin/python -m pytest tests/test_tune.py -v` → all PASS.
Then start the page on the tiny test run or on WL25 (`.venv/bin/cafast tune RUN_DIR --no-browser`), open it in the browser pane, drag across a cell, and check that the diameter field fills in and that Propose → progress → proposal panel → gallery → clicking a thumbnail all work.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/cafast/tune tests/test_tune.py
git commit -m "Let the tune page measure the cell diameter on the max projection"
```

---

### Task 8: WL25 calibration, acceptance, docs

**Files:**
- Create: `bench/autotune_check.py`
- Modify: `README.md`, `src/cafast/autotune/score.py` (only if calibration changes a default)
- Obsidian: `CaFast auto-tuning.md`

**Interfaces:**
- Consumes: `figures.load_run`, `minian.load` (Minian overlay: `ui`, `mi`), `score.unit_flags`, `run.load_doc`, `run.checks_of`.

- [ ] **Step 1: Prerequisite (the user runs it):** rebuild the WL25 factors with the FFT noise fix (`d616477`). Reading the raw AVIs hangs from Claude sessions, so ask the user to run this in their own terminal:

```bash
.venv/bin/cafast run ~/minian_testing_local/WL25_20251204/<raw folder used by cafast_destriped_v2_clean> --workers 6 --name cafast_destriped_v3_fft
```

The raw folder is in `cafast_destriped_v2_clean/params.json` under `raw_dir`; use the same params file (`--params` with that `params.json`) so only the noise estimate changes.

- [ ] **Step 2: Write `bench/autotune_check.py`**

```python
"""Calibrate and accept auto-tune on a session with a Minian reference.

For each given rerun: pass rate of every check among Minian-matched vs unmatched units, plus
the acceptance numbers (matched / Minian units, median size10, flagged).
Usage: python bench/autotune_check.py RUN_DIR --minian NPZ [--runs base auto] [--diameter D]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cafast.autotune import run as at
from cafast.autotune import score
from cafast.tune import figures, minian


def check_rates(run_dir: Path, name: str, npz: Path, checks: score.Checks) -> dict:
    run = figures.load_run(run_dir / "reruns" / name)
    mn = minian.load(npz, run)
    f = score.unit_flags(run.A, run.YrA, run.H, run.W, checks)
    matched = np.zeros(len(f), bool)
    matched[mn.ui] = True
    rates = {
        r: {
            "fail_matched": float(f[r][matched].mean()) if matched.any() else None,
            "fail_unmatched": float(f[r][~matched].mean()) if (~matched).any() else None,
        }
        for r in score.REASONS
    }
    return {
        "name": name,
        "units": len(f),
        "matched": int(matched.sum()),
        "minian_units": int(mn.A.shape[0]),
        "size10_median": float(np.median(f["size10"])) if len(f) else None,
        **score.summarize(f),
        "rates": rates,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--minian", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=["base", at.AUTO])
    ap.add_argument("--diameter", type=float, default=None)
    a = ap.parse_args()
    doc = at.load_doc(a.run_dir / "reruns" / at.AUTO)
    checks = score.Checks(a.diameter) if a.diameter else at.checks_of(doc)
    out = [check_rates(a.run_dir, n, a.minian, checks) for n in a.runs]
    text = json.dumps(out, indent=1)
    print(text)
    (a.run_dir / "reruns" / at.AUTO / "acceptance.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run auto-tune and the check on the rebuilt WL25 run**

```bash
RUN=~/minian_testing_local/WL25_20251204/processed/cafast_destriped_v3_fft
.venv/bin/cafast cnmf $RUN --name base
time .venv/bin/cafast autotune $RUN --diameter 8
.venv/bin/python bench/autotune_check.py $RUN --minian ~/minian_testing_local/WL25_20251204/processed/minian_v2_timing/minian_units.npz
```

(d = 8 px: Minian's median footprint is 51 px above 10 % of peak → √(51·4/π) ≈ 8.)

- [ ] **Step 4: Calibrate.** A check is working if its `fail_unmatched` is clearly higher than its `fail_matched`. If a check fails more than ~10 % of the Minian-matched units (likely `low_pnr`, given the unchecked default of 12), change that default in `score.Checks` to the value that keeps ~95 % of matched units, re-run Step 3, and record the old → new value in the commit message. Do not tune the thresholds to the acceptance numbers themselves.

- [ ] **Step 5: Acceptance** (from the design): auto `matched` ≥ 110 of 165, auto `size10_median` ≤ base, auto `flagged` < base `flagged`, wall time about 4–5 min. If a criterion fails, stop and report to the user with the `acceptance.json` and `sweep.csv` numbers. Do not change the grids or the score rule without the user's OK. Then run `cafast autotune` on WL27 (no Minian; `~/minian_testing_local/WL27_20251205/`, see its README for the run dir) and look at the flagged gallery and at the units on the striped oval artifact (rows 56–93, cols 120–151).

- [ ] **Step 6: README** — add under the `cafast tune` section:

```markdown
### Auto-tune

`cafast autotune RUN_DIR --diameter D` (or "Propose parameters" on `cafast tune`) sweeps the
window-2 parameters stage by stage (~5 min on a 30-min session), scores each setting with
label-free unit checks (trace PNR, one piece, cell size from D, no duplicates; score = good -
flagged) and writes the proposal to `RUN_DIR/reruns/auto/` with `sweep.csv` (the map),
`unit_flags.csv` and `autotune.json`. D is the cell diameter in px of the factor frame.
```

- [ ] **Step 7: Full check, commit, note**

```bash
.venv/bin/ruff format src tests bench && .venv/bin/ruff check --fix src tests bench
.venv/bin/python -m pytest -q
git add bench/autotune_check.py README.md src/cafast/autotune/score.py
git commit -m "Add the WL25 auto-tune calibration check and document cafast autotune"
```

Update the Obsidian note `CaFast auto-tuning.md` (Science vault): set the header to "v1 built", tick the open action "Implementation plan → build v1", and add the WL25 acceptance numbers and any calibrated threshold in 2–3 lines. Keep it short.
