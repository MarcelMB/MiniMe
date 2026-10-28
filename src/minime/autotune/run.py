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
    SEARCHED,
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
SCORE_COLS = ("good", "flagged", "score", *score.REASONS)


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


def check_not_reserved(name: str) -> None:
    """Raise ``ValueError`` for the name auto-tune writes its proposal to."""
    if name == AUTO:
        raise ValueError(f"run name {AUTO!r} is reserved for auto-tune; pick another name")


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------
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
    session = session or Window2Session(run_dir)
    check_diameter(diameter, session.fac.H, session.fac.W_)
    checks = checks or score.Checks(diameter)
    sw = Sweep(session, checks, grids_of(session, grids), on_trial)
    params = sw.run()
    grids = sw.grids  # extended where a best value sat at an edge
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
        "kept": [k for k in DERIVED if k not in derived_keys(session.base, diameter)],
        "proposed": {k: getattr(params, k) for k in SEARCHED},
        "run_values": {k: _json_value(getattr(session.base, k)) for k in SEARCHED + DERIVED},
        "at_edge": [k for k in SEARCHED if at_edge(getattr(params, k), getattr(grids, k))],
        "score": score.summarize(flags),
        "run_score": sw.run_trial.score,  # None when the run's own parameters failed
        "better_than_run": sw.better_than_run,
        "search_error": sw.search_error,
        "n_trials": len(sw.trials),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / DOC).write_text(json.dumps(doc, indent=1))
    return out
