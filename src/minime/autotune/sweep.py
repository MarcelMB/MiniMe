"""Stage-by-stage search of the window-2 parameters; every setting tried is a row of the map.

The run's own parameters are trial zero (through CNMF); the proposal is never worse than
them. Grids are built from the run's values (``relative_grids``), so every grid holds the
run's value, and a grid whose best value sits at its edge is extended outward. Steps:
``seeds`` (seeds_diff_thres x pnr_thres, scored on the init units), ``merge`` (merge_corr x
init_merge_corr from the best seeds, cached after pnr_refine), ``inita`` (inita_corr),
``sparse`` (sparse_snr) and ``loop_merge`` (loop_merge_corr) on the init of the best
init-level setting, then ``confirm`` (the run's searched init values and the best init settings
per distinct merge_corr through CNMF, at the tuned sparse_snr and loop_merge_corr). The
init-level scores only prune within a step; CNMF decides. Factors and FactorCorr are loaded
once.

On a session longer than ``tune_frames`` the steps run on one contiguous middle segment of
that length (``Factors.subset``; every stage scales with frames), and ``decide`` reruns the
run's values and the ``n_decide`` best settings on the whole session: the proposal is chosen
there.
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

from minime import corr, pipeline, seeds
from minime.autotune import score
from minime.cnmf import run_cnmf
from minime.factors import Factors
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
CORR_KEYS = ("merge_corr", "init_merge_corr", "inita_corr", "loop_merge_corr")
DERIVED = ("seeds_max_wnd", "inita_wnd", "dl_wnd", "merge_dist", "merge_min_dist", "size_thres")
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
# Grids from the run's values
# ---------------------------------------------------------------------------

SCALE = (0.5, 0.7, 1.0, 1.4, 2.0)  # thresholds and sparse_snr: run's value x these
CORR_OFFSETS = (-0.1, -0.05, 0.0, 0.05, 0.1)  # correlations: run's value + these
CORR_RANGE = (0.5, 0.98)
PNR_ABSOLUTE = (1.0, 1.5, 2.0, 3.0)  # pnr_thres grid when the run uses "auto"
# merge_min_dist = MIN_DIST_FRAC x diameter: no two cells sit this close, only a split one. Set
# from WL25 (Minian pairs >= 0.41 D) and the synthetic sessions (real cells at 0.3-0.38 D by
# design still gained precision); a default to show and override, not a law
MIN_DIST_FRAC = 0.4
# frames the search explores on (a middle segment; 0 = the whole session). ~22 min at 30 fps:
# WL25 (39822 frames) stays whole; a contiguous segment keeps the seed windows and the corr
# low-pass on continuous time and skips the bleaching transient at the start
TUNE_FRAMES = 40000


@dataclass(frozen=True)
class Grids:
    """The values each step tries; ``relative_grids`` builds them from the run's values."""

    seeds_diff_thres: tuple[float, ...]
    pnr_thres: tuple[float | str, ...]
    merge_corr: tuple[float, ...]
    init_merge_corr: tuple[float, ...]
    inita_corr: tuple[float, ...]
    sparse_snr: tuple[float, ...]
    loop_merge_corr: tuple[float, ...]
    n_confirm: int = 5
    n_decide: int = 3  # segment search: best settings rerun on the whole session
    # extensions per parameter while its best value sits at the grid edge; a run that starts
    # far off needs several (V4: sparse_snr 3.74 -> 0.96 used two and was still at the edge)
    max_extend: int = 4
    integer_diff: bool = True  # seeds_diff_thres in whole steps (uint8 block maxima)

    def n_trials(self) -> int:
        """Upper bound on the number of settings ``Sweep.run`` tries."""
        return (
            1  # the run's own parameters
            + len(self.seeds_diff_thres) * len(self.pnr_thres)
            + len(self.merge_corr) * len(self.init_merge_corr)
            + len(self.inita_corr)
            + 1  # the run's searched values on the search base
            + self.n_confirm
            + len(self.sparse_snr)
            + len(self.loop_merge_corr)
            + self.max_extend * len(SEARCHED)
            + 1
            + self.n_decide  # decide: the run's values and the best on the whole session
        )


def _round(v: float, integer: bool) -> float:
    return max(1, round(v)) if integer else float(f"{v:.3g}")


def _unique(values: list) -> tuple:
    strs = [v for v in values if isinstance(v, str)]
    nums = sorted({v for v in values if not isinstance(v, str)})
    return (*strs, *nums)


def _corr_grid(v: float) -> tuple[float, ...]:
    lo, hi = CORR_RANGE
    return _unique([v, *(round(min(hi, max(lo, v + o)), 3) for o in CORR_OFFSETS)])


def relative_grids(base: Params, integer_diff: bool = True, **overrides: Any) -> Grids:
    """Grids around the run's values: thresholds x ``SCALE``, correlations + ``CORR_OFFSETS``
    (clipped to ``CORR_RANGE``), ``pnr_thres`` with "auto" added (absolute values when the
    run uses "auto"). Every grid holds the run's own value."""
    pnr = base.pnr_thres
    pnr_grid = PNR_ABSOLUTE if isinstance(pnr, str) else [_round(pnr * s, False) for s in SCALE]
    g = {
        "seeds_diff_thres": _unique(
            [_round(base.seeds_diff_thres * s, integer_diff) for s in SCALE]
        ),
        "pnr_thres": ("auto", *_unique([*pnr_grid, *([] if isinstance(pnr, str) else [pnr])])),
        "sparse_snr": _unique(
            [base.sparse_snr, *(_round(base.sparse_snr * s, False) for s in SCALE)]
        ),
        **{k: _corr_grid(getattr(base, k)) for k in CORR_KEYS},
    }
    if not integer_diff:
        g["seeds_diff_thres"] = _unique([base.seeds_diff_thres, *g["seeds_diff_thres"]])
    return Grids(**g, integer_diff=integer_diff, **overrides)


def extend(
    spec: tuple[float, ...], grid: tuple, value: Any, *, integer: bool = False, corr: bool = False
) -> float | None:
    """The next value outward when ``value`` is the lowest or highest number of ``grid``.

    ``spec`` is ``CORR_OFFSETS`` (one more offset step) or ``SCALE`` (one more factor).
    None when ``value`` is inside the grid, is a string, or the next value is not new
    (clipped to ``CORR_RANGE``, rounded onto an existing value).
    """
    nums = sorted(v for v in grid if not isinstance(v, str))
    if isinstance(value, str) or not nums or value not in (nums[0], nums[-1]):
        return None
    up = value == nums[-1]
    if corr:
        step = spec[-1] - spec[-2] if up else spec[0] - spec[1]
        new = round(min(CORR_RANGE[1], max(CORR_RANGE[0], value + step)), 3)
    else:
        new = _round(value * (spec[-1] / spec[-2] if up else spec[0] / spec[1]), integer)
    return None if new in nums else new


def derived_keys(base: Params, diameter: float, defaults: Params | None = None) -> list[str]:
    """The size parameters still at their built-in default, i.e. the ones derived from the
    cell ``diameter``; a value the run changed is kept."""
    score.Checks(diameter)  # validates
    d = defaults or Params()
    return [k for k in DERIVED if _norm(getattr(base, k)) == _norm(getattr(d, k))]


def derived_params(
    base: Params,
    diameter: float,
    defaults: Params | None = None,
    min_dist_frac: float = MIN_DIST_FRAC,
) -> Params:
    """``base`` with the size parameters it left at their defaults set from ``diameter`` (px).

    ``size_thres`` has no upper bound: CNMF compares it with the nnz of the *dilated* support
    (cell plus ``dl_wnd`` halo), not the cell size, so a cell-based bound would drop real units.
    Oversized units are still flagged by the score's ``bad_size`` check.
    """
    keys = derived_keys(base, diameter, defaults)
    d = max(1, round(diameter))
    area = math.pi * (diameter / 2) ** 2
    lo = max(1, round(0.25 * area))
    new = {
        "seeds_max_wnd": d,
        "inita_wnd": d,
        "dl_wnd": d,
        "merge_dist": d,
        "merge_min_dist": round(min_dist_frac * diameter, 1),
        "size_thres": (lo, None),
    }
    return dataclasses.replace(base, **{k: new[k] for k in keys})


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


def confirm_candidates(trials: list[Trial], base: Params, n: int) -> list[Trial]:
    """Up to ``n`` init settings for CNMF: the best of each distinct ``merge_corr`` first
    (the init-level score cannot judge the merge), then the next best overall."""
    r = ranked(trials, base)
    first: dict[Any, Trial] = {}
    for t in r:
        first.setdefault(t.values["merge_corr"], t)
    out = list(first.values())
    out += [t for t in r if t not in out]
    return out[:n]


def middle_segment(edges: np.ndarray, n: int) -> tuple[int, int] | None:
    """``[t0, t1)`` of about ``n`` frames centred in the session, on the block ``edges``;
    None when ``n`` is 0 or the session is not longer than ``n``."""
    T = int(edges[-1])
    if n < 0:
        raise ValueError(f"tune_frames must be >= 0, got {n}")
    if n == 0 or n >= T:
        return None
    t0 = int(edges[np.abs(edges - (T - n) // 2).argmin()])
    t1 = int(edges[np.abs(edges - (t0 + n)).argmin()])
    return (t0, t1) if t1 > t0 else None


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
    grids: Grids | None = None  # default: relative_grids of the run
    on_trial: Callable[[Trial], None] | None = None
    min_dist_frac: float = MIN_DIST_FRAC
    # a setting with more than cap_factor x the run's seeds after PNR, or x its init units,
    # fails fast: it is no candidate, and on a 2 h session one such trial ran for 20 min
    cap_factor: float = 5.0
    tune_frames: int = TUNE_FRAMES
    base: Params = field(init=False)  # the run's params with the default sizes derived
    trials: list[Trial] = field(default_factory=list)
    run_trial: Trial | None = None
    final: Trial | None = None
    better_than_run: bool = False
    search_error: str | None = None  # why the search stopped when it fell back to the run
    timer: StageTimer = field(default_factory=StageTimer)
    fc: corr.FactorCorr | None = None
    fac: Factors = field(init=False)  # what the steps run on: the segment or the session
    segment: tuple[int, int] | None = field(init=False)  # frames explored on, None = all
    _pnr: np.ndarray = field(init=False, repr=False)  # per-pixel seed PNR, NaN = not yet read

    def __post_init__(self) -> None:
        self.base = derived_params(
            self.session.base, self.checks.diameter, min_dist_frac=self.min_dist_frac
        )
        self.fac = self.session.fac
        self.segment = middle_segment(self.fac.block_edges, self.tune_frames)
        self._pnr = np.full(self.fac.P, np.nan)
        if self.grids is None:
            integer = bool(np.issubdtype(self.fac.block_max.dtype, np.integer))
            self.grids = relative_grids(self.session.base, integer_diff=integer)

    # ---- stages --------------------------------------------------------------------

    def _pnr_of(self, s: pd.DataFrame) -> np.ndarray:
        """``seeds.seed_pnr`` of ``s`` from the per-pixel cache; reads only unseen pixels.

        The seeds step tries every threshold pair, and the trace read behind the PNR is the
        step's whole cost (274 s per setting on a 2 h session), while a pixel's PNR does not
        depend on the thresholds.
        """
        pix = seeds.seed_pixels(self.fac, s)
        new = np.isnan(self._pnr[pix])
        if new.any():
            self._pnr[pix[new]] = seeds.seed_pnr(self.fac, s[new])
        return self._pnr[pix]

    def _params(self, values: dict[str, Any], base: Params | None = None) -> Params:
        return dataclasses.replace(base or self.base, **values)

    def _cap(self, what: str, n: int) -> None:
        """Raise ``ValueError`` when ``n`` exceeds ``cap_factor`` x the run's own count."""
        ref = self.run_trial.counts.get(what) if self.run_trial is not None else None
        if ref and n > self.cap_factor * ref:
            raise ValueError(f"too many {what}: {n} > {self.cap_factor:g} x the run's {ref}")

    def _init(
        self, p: Params, timer: StageTimer, s_pnr: pd.DataFrame | None = None
    ) -> tuple[sps.csr_matrix, np.ndarray]:
        fac = self.fac
        if s_pnr is None:
            s_pnr = pipeline.seeds_unmerged(fac, p, timer, pnr_of=self._pnr_of)
            self._cap("seeds_pnr", len(s_pnr))
        s = pipeline.seeds_merged(fac, self.fc, s_pnr, p, timer)
        self._cap("seeds_merged", len(s))
        return pipeline.init_units(fac, self.fc, s, p, timer)

    def _cnmf(
        self, p: Params, timer: StageTimer, A: sps.csr_matrix, C: np.ndarray
    ) -> tuple[sps.csr_matrix, np.ndarray]:
        res = run_cnmf(self.fac, A, C, p, self.fc.traces_corr)
        timer.counts.update(res.counts)
        timer.counts["units_final"] = res.A.shape[0]
        return res.A, res.YrA

    def _full(self, p: Params, timer: StageTimer) -> tuple[sps.csr_matrix, np.ndarray]:
        return self._cnmf(p, timer, *self._init(p, timer))

    # ---- one setting ---------------------------------------------------------------

    def _try(
        self, step: str, values: dict[str, Any], stage: Stage, base: Params | None = None
    ) -> Trial:
        t0, timer = time.perf_counter(), StageTimer()
        try:
            A, traces = stage(self._params(values, base), timer)
        except FAILED as e:
            sc, med, err = None, None, f"{type(e).__name__}: {e}"
        else:
            fac = self.fac
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

    def _explore(
        self, step: str, v: dict[str, Any], keys: tuple[str, ...], stage: Stage
    ) -> dict[str, Any]:
        """Try the grids of ``keys`` on top of ``v``; while the best value of a key sits at
        its grid's edge, add the next value outward (``max_extend`` times). Returns the best
        values of the step; ``self.grids`` records the extended grids."""
        grids = {k: list(getattr(self.grids, k)) for k in keys}
        for combo in product(*grids.values()):
            self._try(step, {**v, **dict(zip(keys, combo, strict=True))}, stage)
        for k in keys:
            for _ in range(self.grids.max_extend):
                bv = best(self._step(step), self.base, step).values
                spec = CORR_OFFSETS if k in CORR_KEYS else SCALE
                integer = k == "seeds_diff_thres" and self.grids.integer_diff
                new = extend(spec, tuple(grids[k]), bv[k], integer=integer, corr=k in CORR_KEYS)
                if new is None:
                    break
                grids[k].append(new)
                self._try(step, {**bv, k: new}, stage)
        self.grids = dataclasses.replace(self.grids, **{k: _unique(g) for k, g in grids.items()})
        return best(self._step(step), self.base, step).values

    # ---- the search ----------------------------------------------------------------

    def run(self) -> Params:
        """Run every step; returns the proposed parameters (``self.final`` is its trial).

        When every setting of a step fails (sizes derived from a wrong ``diameter``, say),
        the run's own parameters are proposed if they worked; otherwise the error is raised.
        """
        run = self.session.base
        if self.segment is None:
            self.fc = self.session.factor_corr(self.base.noise_freq, self.timer)
        else:
            self.fac = self.session.fac.subset(*self.segment)
            with self.timer("corr_setup"):
                self.fc = corr.FactorCorr(self.fac, self.base.noise_freq)
        run_values = {k: getattr(run, k) for k in SEARCHED}
        self.run_trial = self._try("run", run_values, self._full, base=run)
        top: list[Trial] = []
        try:
            top = self._search(run_values)
        except RuntimeError as e:
            if self.run_trial.score is None:
                raise
            self.search_error = str(e)
        found = (
            self._decide(run_values, top) if self.segment is not None else (top[0] if top else None)
        )
        if found is not None and (
            self.run_trial.score is None or found.score["score"] > self.run_trial.score["score"]
        ):
            self.final, self.better_than_run = found, True
            return self._params(found.values)
        self.final, self.better_than_run = self.run_trial, False
        return run

    def _decide(self, run_values: dict[str, Any], top: list[Trial]) -> Trial | None:
        """Rerun the run's values and the segment's ``n_decide`` best settings on the whole
        session; the run's becomes ``run_trial``. Returns the best of the others there (None
        if all failed): segment scores within a few points of each other split by 25 on the
        session (V4 2 h)."""
        self.fc = None  # free the segment's before the session's is built
        self.fac = self.session.fac
        self.fc = self.session.factor_corr(self.base.noise_freq, self.timer)
        self._pnr = np.full(self.fac.P, np.nan)  # the segment's PNR is not the session's
        self.run_trial = None  # no caps: the session has more seeds than the segment
        run = self._try("decide", run_values, self._full, base=self.session.base)
        others = [
            self._try("decide", t.values, self._full)
            for t in [t for t in top if t.values != run_values][: self.grids.n_decide]
        ]
        self.run_trial = run
        r = ranked(others, self.base)
        if run.score is None and not r:
            raise RuntimeError(
                f"auto-tune: the run's parameters failed on the whole session ({run.error})"
            )
        return r[0] if r else None

    def _tail(self, v: dict[str, Any]) -> dict[str, Any]:
        """``sparse`` then ``loop_merge`` on the init of ``v`` (built once, CNMF only)."""
        A, C = self._init(self._params(v), StageTimer())

        def on_init(p: Params, t: StageTimer) -> tuple[sps.csr_matrix, np.ndarray]:
            return self._cnmf(p, t, A, C)

        v = self._explore("sparse", v, ("sparse_snr",), on_init)
        return self._explore("loop_merge", v, ("loop_merge_corr",), on_init)

    def _search(self, run_values: dict[str, Any]) -> list[Trial]:
        """Every step; returns the CNMF-level trials, best first."""
        g, base = self.grids, self.base
        v = self._explore("seeds", run_values, ("seeds_diff_thres", "pnr_thres"), self._init)
        s_pnr = pipeline.seeds_unmerged(
            self.fac, self._params(v), StageTimer(), pnr_of=self._pnr_of
        )

        def cached(p: Params, t: StageTimer) -> tuple[sps.csr_matrix, np.ndarray]:
            return self._init(p, t, s_pnr)

        v = self._explore("merge", v, ("merge_corr", "init_merge_corr"), cached)
        self._explore("inita", v, ("inita_corr",), cached)
        candidates = confirm_candidates(self._step(*INIT_STEPS), base, g.n_confirm)
        # sparse_snr and loop_merge_corr first, on the best init: at the run's own (V4 2 h:
        # sparse_snr 3.74, ~130 units) the inits tie within 5, at the tuned one (0.96, ~500
        # units) they split by up to 85 and the merge settings keep their order on the session
        tail = self._tail(candidates[0].values)
        tuned = {k: tail[k] for k in ("sparse_snr", "loop_merge_corr")}
        # the run's own init values on the search base among them: the init-level scores
        # cannot be trusted to keep them (WL25: they beat every candidate by 17)
        tried = [t.values for t in self.trials if t.step in ("sparse", "loop_merge")]
        for values in [run_values, *(t.values for t in candidates)]:
            values = {**values, **tuned}
            if values not in tried:
                tried.append(values)
                self._try("confirm", values, self._full)
        return ranked(self._step("confirm", "sparse", "loop_merge"), base)
