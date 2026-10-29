"""``minime autotune``: sweep, then write the proposal as ``RUN_DIR/reruns/NAME/``.

Besides the normal rerun files the directory gets ``sweep.csv`` (one row per setting tried),
``unit_flags.csv`` (the checks on the proposal's final units) and ``autotune.json``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sps

from minime.autotune import score
from minime.autotune.sweep import (
    DERIVED,
    MIN_DIST_FRAC,
    SEARCHED,
    STEP_OF,
    TUNE_FRAMES,
    Grids,
    Sweep,
    Trial,
    at_edge,
    derived_keys,
    relative_grids,
)
from minime.window2 import Window2Session

AUTO = "auto"
DEFAULT_GRIDS: Grids | None = None  # None: ``relative_grids`` of the run
DOC = "autotune.json"
COUNT_COLS = ("seeds_init", "seeds_pnr", "seeds_merged", "units_init_merge", "units_final")
MIN_UNITS_FOR_ESTIMATE = 10
DUP_CORR = score.Checks(1.0).dup_corr  # the default, for the CLI and the page
SCORE_COLS = ("good", "flagged", "lost", "score", *score.REASONS)


# ----------------------------------------------------------------------------------------
# Tables and the JSON document
# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------
# Sweep table helpers (the curves of the tune page)
# ----------------------------------------------------------------------------------------
def _is_num(v: Any) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def _order(values: pd.Series) -> list[Any]:
    """Grid order: ``auto`` first, then numbers ascending."""
    uniq = list(dict.fromkeys(values.tolist()))
    words = [v for v in uniq if not _is_num(v)]
    nums = sorted((v for v in uniq if _is_num(v)), key=float)
    return words + nums


def best_per_value(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per value of ``key`` in its step: the best-scoring successful trial, in grid order."""
    d = df[(df["step"] == STEP_OF[key]) & df["score"].notna()]
    if d.empty:
        return d
    top = d.sort_values("score", ascending=False, kind="stable").drop_duplicates(key)
    order = {v: i for i, v in enumerate(_order(top[key]))}
    return top.sort_values(key, key=lambda s: s.map(order)).reset_index(drop=True)


def same_value(a: Any, b: Any) -> bool:
    """``2`` and ``2.0`` and ``"2.0"`` are one value; ``auto`` only equals ``auto``."""
    if _is_num(a) and _is_num(b):
        return float(a) == float(b)
    return str(a) == str(b)


# ----------------------------------------------------------------------------------------
# Input checks
# ----------------------------------------------------------------------------------------
def check_diameter(diameter: float, H: int, W: int) -> None:
    """Raise ``ValueError`` unless ``diameter`` is a finite number of px that fits the FOV."""
    score.Checks(diameter)  # a number > 0
    if not math.isfinite(diameter) or diameter > min(H, W):
        raise ValueError(
            f"diameter must be a finite number of px of at most {min(H, W)} "
            f"(the smaller field-of-view side), got {diameter}"
        )


def grids_of(session: Window2Session, grids: Grids | None = None) -> Grids:
    """``grids``, else ``DEFAULT_GRIDS``, else the grids built from the run's values."""
    if grids or DEFAULT_GRIDS:
        return grids or DEFAULT_GRIDS
    integer = bool(np.issubdtype(session.fac.block_max.dtype, np.integer))
    return relative_grids(session.base, integer_diff=integer)


def n_trials(session: Window2Session, grids: Grids | None = None) -> int:
    """Upper bound on the settings an auto-tune of ``session`` tries."""
    return grids_of(session, grids).n_trials()


def estimate_diameter(run_dir: Path) -> float:
    """Cell diameter (px) from the run's final footprints: the equivalent diameter of the
    median footprint size above 10 % of peak (the size the checks measure).

    Footprint size follows the run's ``sparse_snr``, so this is a default to show and
    override, not a measurement. Raises ``ValueError`` with fewer than
    ``MIN_UNITS_FOR_ESTIMATE`` units.
    """
    A = sps.load_npz(Path(run_dir) / "cnmf" / "A.npz")
    if A.shape[0] < MIN_UNITS_FOR_ESTIMATE:
        raise ValueError(
            f"only {A.shape[0]} units in {run_dir}: too few to estimate the cell diameter, "
            "pass --diameter"
        )
    return round(2 * math.sqrt(float(np.median(score.size_above(A))) / math.pi), 1)


def check_not_reserved(name: str) -> None:
    """Raise ``ValueError`` for the name auto-tune writes its proposal to."""
    if name == AUTO:
        raise ValueError(f"run name {AUTO!r} is reserved for auto-tune; pick another name")


# ----------------------------------------------------------------------------------------
# Progress file
# ----------------------------------------------------------------------------------------
RERUN_STEP = "rerun"


class StepLog:
    """One ``progress.jsonl`` row per sweep step (``tune_seeds`` ...) and one for the final
    rerun, in ``StageTimer``'s line format, so ``minime serve`` shows them as stages.

    ``trial`` is the ``on_trial`` callback, ``stage`` the ``on_stage`` one; ``close`` ends
    the open row (``error=True`` marks it failed).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.write_text("")
        self.step: str | None = None
        self.t0 = 0.0
        self.n = 0

    def _write(self, event: str, **extra: Any) -> None:
        doc = {"stage": f"tune_{self.step}", "event": event, "t": time.time(), **extra}
        with self.path.open("a") as f:
            f.write(json.dumps(doc) + "\n")

    def _switch(self, step: str) -> None:
        if step == self.step:
            return
        self.close()
        self.step, self.t0, self.n = step, time.time(), 0
        self._write("start")

    def trial(self, t: Trial) -> None:
        self._switch(t.step)
        self.n += 1

    def stage(self, name: str) -> None:
        self._switch(RERUN_STEP)

    def close(self, error: bool = False) -> None:
        if self.step is None:
            return
        self._write("error" if error else "end", seconds=time.time() - self.t0, n=self.n)
        self.step = None


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------
def autotune(
    run_dir: Path,
    diameter: float | None = None,
    *,
    checks: score.Checks | None = None,
    grids: Grids | None = None,
    name: str = AUTO,
    session: Window2Session | None = None,
    on_trial: Callable[[Trial], None] | None = None,
    dup_corr: float = DUP_CORR,
    min_dist_frac: float = MIN_DIST_FRAC,
    tune_frames: int = TUNE_FRAMES,
    on_stage: Callable[[str], None] | None = None,
) -> Path:
    """Sweep window 2 of ``run_dir`` and write the proposal to ``run_dir/reruns/name``.

    ``diameter`` None: ``estimate_diameter`` of the run. ``dup_corr`` and ``min_dist_frac``
    are the two thresholds the checks and the derived merge distance rest on. The search
    explores on a middle segment of ``tune_frames`` frames (0 = the whole session) and decides
    on the whole session. ``on_stage`` follows the final rerun's stages.
    """
    t0 = time.perf_counter()
    session = session or Window2Session(run_dir)
    source = "given" if diameter is not None else "footprints"
    if diameter is None:
        diameter = estimate_diameter(run_dir)
    check_diameter(diameter, session.fac.H, session.fac.W_)
    checks = checks or score.Checks(diameter, dup_corr=dup_corr)
    sw = Sweep(
        session,
        checks,
        grids_of(session, grids),
        on_trial,
        min_dist_frac=min_dist_frac,
        tune_frames=tune_frames,
    )
    params = sw.run()
    grids = sw.grids  # extended where a best value sat at an edge
    out = session.run(params, name, on_stage=on_stage)
    sweep_table(sw.trials).to_csv(out / "sweep.csv", index=False)
    A = sps.load_npz(out / "cnmf" / "A.npz")
    YrA = np.load(out / "cnmf" / "YrA.npy")
    flags = score.unit_flags(A, YrA, session.fac.H, session.fac.W_, checks)
    flags.to_csv(out / "unit_flags.csv", index=False)
    A0 = sps.load_npz(Path(run_dir) / "cnmf" / "A.npz")
    Y0 = np.load(Path(run_dir) / "cnmf" / "YrA.npy")
    ref = score.good_centroids(score.unit_flags(A0, Y0, session.fac.H, session.fac.W_, checks))
    lost = score.lost_units(ref, flags, checks.keep_dist * checks.diameter)
    pairs = {
        which: {
            k: v
            for k, v in score.pair_stats(
                a, y, session.fac.H, session.fac.W_, checks, min_dist_frac
            ).items()
            if not isinstance(v, np.ndarray)
        }
        for which, a, y in (("run", A0, Y0), ("proposal", A, YrA))
    }
    doc = {
        "diameter": diameter,
        "diameter_source": source,
        "min_dist_frac": min_dist_frac,
        "tune_frames": tune_frames,
        "segment": None if sw.segment is None else list(sw.segment),  # frames explored on
        "frames": session.fac.T,
        "pairs": pairs,  # what dup_corr and min_dist_frac act on, run and proposal
        "checks": asdict(checks),
        "grids": {k: _json_value(v) for k, v in asdict(grids).items()},
        "derived": {k: _json_value(getattr(params, k)) for k in DERIVED},
        "kept": [k for k in DERIVED if k not in derived_keys(session.base, diameter)],
        "proposed": {k: getattr(params, k) for k in SEARCHED},
        "run_values": {k: _json_value(getattr(session.base, k)) for k in SEARCHED + DERIVED},
        "at_edge": [k for k in SEARCHED if at_edge(getattr(params, k), getattr(grids, k))],
        "score": score.summarize(flags, lost),
        "run_score": sw.run_trial.score,  # None when the run's own parameters failed
        "better_than_run": sw.better_than_run,
        "search_error": sw.search_error,
        "n_trials": len(sw.trials),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / DOC).write_text(json.dumps(doc, indent=1))
    return out
