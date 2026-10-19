"""Rerun window 2 (seeds -> init -> CNMF) from a run's saved factors, without passes 1-2.

``RUN_DIR/reruns/NAME/`` gets ``params.json``, ``report.json``, ``cnmf/`` (same files as a full
run), ``stages/`` (seed tables and the footprints after every step, for ``cafast tune``) and
symlinks ``factors`` / ``motion`` to the source run, so tools that read a run dir read it too.

Parameters that shaped the factors (preprocessing, motion, tiling, compression) cannot change
here; ``seeds_wnd`` / ``seeds_stp`` can, as long as their windows fall on the block edges pass
1 stored. Everything else (seeds, PNR, merges, init, CNMF) is free.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import scipy.sparse as sps

import cafast
from cafast import corr, pipeline, seeds
from cafast.factors import Factors
from cafast.params import Params
from cafast.report import StageTimer

RERUNS = "reruns"
# parameters that shaped the saved factors: a change needs ``cafast run``
FACTOR_PARAMS = (
    "median_ksize",
    "tophat_wnd",
    "motion_max_shift",
    "motion_down",
    "template_frames",
    "motion_before_tophat",
    "motion_bandpass",
    "tile",
    "svd_k0",
    "svd_power_iter",
    "rank_rho_min",
    "rank_z",
)
SEED_WINDOW_PARAMS = ("seeds_wnd", "seeds_stp")  # must fall on the stored block edges
EXECUTION_PARAMS = ("workers", "sub_batch")
WINDOW2_PARAMS = tuple(
    f.name for f in fields(Params) if f.name not in FACTOR_PARAMS + EXECUTION_PARAMS
)
LINKED = ("factors", "motion")

# ---------------------------------------------------------------------------
# Inputs and checks
# ---------------------------------------------------------------------------


def load_factors(run_dir: Path) -> Factors:
    return Factors.load(Path(run_dir) / "factors")


def run_params(run_dir: Path, override: Path | None = None) -> Params:
    """The run's ``params.json``, updated by the (partial) JSON ``override`` if given."""
    base = Params.from_json(Path(run_dir) / "params.json")
    return base if override is None else Params.from_json(override, base)


def raw_dir(run_dir: Path) -> Path | None:
    """The raw folder recorded in the run's ``params.json`` (``None`` if it is not recorded)."""
    raw = json.loads((Path(run_dir) / "params.json").read_text()).get("raw_dir")
    return Path(raw) if raw else None


def _norm(v: Any) -> Any:
    return list(v) if isinstance(v, tuple | list) else v


def check_factor_params(fac: Factors, params: Params) -> None:
    """Raise ``ValueError`` if ``params`` would need different factors than ``fac``."""
    stored = fac.meta.get("params", {})
    diff = [
        f"{k} {stored[k]!r} -> {getattr(params, k)!r}"
        for k in FACTOR_PARAMS
        if k in stored and _norm(stored[k]) != _norm(getattr(params, k))
    ]
    if diff:
        raise ValueError(
            "parameters differ from the ones the factors were built with: "
            + ", ".join(diff)
            + " (these need a full rerun: cafast run)"
        )
    windows = seeds.rolling_windows(fac.T, params.seeds_wnd, params.seeds_stp)
    try:
        seeds.window_block_ranges(fac.block_edges, windows)
    except ValueError:
        raise ValueError(
            f"seeds_wnd {params.seeds_wnd} / seeds_stp {params.seeds_stp}: the seed windows do "
            "not fall on the block edges pass 1 stored (built with seeds_wnd "
            f"{stored.get('seeds_wnd')}, seeds_stp {stored.get('seeds_stp')}); "
            "a change needs a full rerun: cafast run"
        ) from None


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _write_params(out: Path, params: Params, run_dir: Path) -> None:
    src = json.loads((run_dir / "params.json").read_text())
    doc = {
        **params.asdict(),
        "raw_dir": src.get("raw_dir"),
        "raw_hash": src.get("raw_hash"),
        "source_run": str(run_dir.resolve()),
        "cafast_version": cafast.__version__,
    }
    (out / "params.json").write_text(json.dumps(doc, indent=1))


def _link_inputs(out: Path, run_dir: Path) -> None:
    """Relative symlinks ``out/factors`` and ``out/motion`` to the source run's dirs."""
    rel = Path(os.path.relpath(run_dir.resolve(), out.resolve()))
    for name in LINKED:
        if (run_dir / name).is_dir():
            (out / name).symlink_to(rel / name, target_is_directory=True)


def _write_stages(d: Path, w2: pipeline.Window2Result) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, df in w2.seeds.items():
        df.to_csv(d / f"{name}.csv", index=False)
    for name, A in w2.footprints.items():
        sps.save_npz(d / f"A_{name}.npz", sps.csr_matrix(A))
    (d / "index.json").write_text(json.dumps({"footprints": list(w2.footprints)}))


def footprint_stages(rerun_dir: Path) -> dict[str, sps.csr_matrix]:
    """Footprints after each window-2 step, in pipeline order (``initA`` ... ``merge_{n-1}``)."""
    d = Path(rerun_dir) / "stages"
    names = json.loads((d / "index.json").read_text())["footprints"]
    return {n: sps.load_npz(d / f"A_{n}.npz").tocsr() for n in names}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class Window2Session:
    """A run's factors held in memory, for repeated window-2 reruns (``cafast tune``).

    The ``FactorCorr`` of the last ``noise_freq`` is kept, so a rerun that leaves it unchanged
    skips the correlation setup.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.fac = load_factors(self.run_dir)
        self.base = run_params(self.run_dir)
        self._fc: corr.FactorCorr | None = None

    def factor_corr(self, noise_freq: float, timer: StageTimer) -> corr.FactorCorr:
        if self._fc is None or self._fc.noise_freq != noise_freq:
            self._fc = None  # free the old one first
            with timer("corr_setup"):
                self._fc = corr.FactorCorr(self.fac, noise_freq)
        return self._fc

    def out_dir(self, name: str) -> Path:
        pipeline.check_run_name(name)
        return self.run_dir / RERUNS / name

    def run(
        self, params: Params | None, name: str, on_stage: Callable[[str], None] | None = None
    ) -> Path:
        """Rerun window 2 with ``params`` (default: the run's own) into ``reruns/name``."""
        t0 = time.perf_counter()
        params = self.base if params is None else params
        out = self.out_dir(name)
        check_factor_params(self.fac, params)
        pipeline._reset_dir(out)
        _write_params(out, params, self.run_dir)
        _link_inputs(out, self.run_dir)
        timer = StageTimer(on_stage)
        try:
            fc = self.factor_corr(params.noise_freq, timer)
            w2 = pipeline.run_window2(self.fac, params, timer, fc)
            with timer("write_outputs"):
                pipeline._write_cnmf(out / "cnmf", w2.cnmf, self.fac.H, self.fac.W_)
                _write_stages(out / "stages", w2)
        except Exception:
            (out / "error.txt").write_text(traceback.format_exc())
            raise
        timer.to_json(
            out / "report.json",
            {
                "source_run": str(self.run_dir.resolve()),
                "wall_total": time.perf_counter() - t0,
                "version": cafast.__version__,
            },
        )
        return out


def rerun(run_dir: Path, params: Params | None = None, name: str = "rerun") -> Path:
    """Rerun window 2 of ``run_dir`` into ``run_dir/reruns/name``; returns that directory."""
    pipeline.check_run_name(name)
    return Window2Session(run_dir).run(params, name)
