"""Stage-by-stage search of the window-2 parameters; every setting tried is a row of the map.

Steps: ``seeds`` (seeds_diff_thres x pnr_thres, scored on the init units), ``merge``
(merge_corr x init_merge_corr from the best seeds, cached after pnr_refine), ``inita``
(inita_corr), ``confirm`` (top init settings through CNMF), ``sparse`` (sparse_snr on the
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

from minime import corr, pipeline
from minime.autotune import score
from minime.cnmf import run_cnmf
from minime.params import Params
from minime.report import StageTimer
from minime.window2 import Window2Session, _norm

SEARCHED = (
    "seeds_diff_thres",
    "pnr_thres",
    "merge_corr",
    "init_merge_corr",
    "inita_corr",
    "sparse_snr",
    "loop_merge_corr",
)
DERIVED = ("seeds_max_wnd", "inita_wnd", "dl_wnd", "merge_dist", "size_thres")
STEP_OF = {
    "seeds_diff_thres": "seeds",
    "pnr_thres": "seeds",
    "merge_corr": "merge",
    "init_merge_corr": "merge",
    "inita_corr": "inita",
    "sparse_snr": "sparse",
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
    sparse_snr: tuple[float, ...] = (0.4, 0.6, 1.0, 1.4, 2.0, 4.0)
    loop_merge_corr: tuple[float, ...] = (0.85, 0.9, 0.95)
    n_confirm: int = 3

    def n_trials(self) -> int:
        """Upper bound on the number of settings ``Sweep.run`` tries."""
        return (
            len(self.seeds_diff_thres) * len(self.pnr_thres)
            + len(self.merge_corr) * len(self.init_merge_corr)
            + len(self.inita_corr)
            + self.n_confirm
            + len(self.sparse_snr)
            + len(self.loop_merge_corr)
        )


def derived_params(base: Params, diameter: float) -> Params:
    """``base`` with the size parameters set from the cell ``diameter`` (px).

    ``size_thres`` has no upper bound: CNMF compares it with the nnz of the *dilated* support
    (cell plus ``dl_wnd`` halo), not the cell size, so a cell-based bound would drop real units.
    Oversized units are still flagged by the score's ``bad_size`` check.
    """
    score.Checks(diameter)  # validates
    d = max(1, round(diameter))
    area = math.pi * (diameter / 2) ** 2
    lo = max(1, round(0.25 * area))
    return dataclasses.replace(
        base, seeds_max_wnd=d, inita_wnd=d, dl_wnd=d, merge_dist=d, size_thres=(lo, None)
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
        errors = list(dict.fromkeys(t.error for t in trials if t.error))[:3]
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise RuntimeError(f"auto-tune step {step!r}: every setting failed{detail}")
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
    base: Params = field(init=False)
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

        for sp in g.sparse_snr:
            self._try("sparse", {**v, "sparse_snr": sp}, on_init)
        top = best(self._step("sparse"), base, "sparse")
        for lm in g.loop_merge_corr:
            self._try("loop_merge", {**top.values, "loop_merge_corr": lm}, on_init)
        self.final = best([top, *self._step("loop_merge")], base, "loop_merge")
        return self._params(self.final.values)
