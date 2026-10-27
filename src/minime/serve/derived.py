"""Results the stage pages need that a run did not save: computed once into the cache."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from minime import corr, motion, pipeline
from minime.factors import Factors
from minime.params import Params
from minime.report import StageTimer
from minime.serve.runs import RunView
from minime.serve.traces import overview

SEED_STAGES = ("init", "pnr", "merged")


def seeds(run: RunView) -> dict[str, np.ndarray] | None:
    """Seed positions (n, 2) per step: from the run if it saved them, else from the cache."""
    if run.has_saved_seeds:
        return {k: np.load(run.path / "seeds" / f"{k}.npy") for k in SEED_STAGES}
    if all(run.cache.has(f"seeds_{k}.npy") for k in SEED_STAGES):
        return {k: run.cache.load_npy(f"seeds_{k}.npy") for k in SEED_STAGES}
    return None


def init_A(run: RunView) -> sps.csr_matrix | None:
    if run.has_saved_init:
        return sps.load_npz(run.path / "init" / "A.npz").tocsr()
    if run.cache.has("init_A.npz"):
        return sps.load_npz(run.cache.dir / "init_A.npz").tocsr()
    return None


def compute_seeds_and_init(run: RunView) -> None:
    """Rerun the seed and init stages on the saved factors (~2.5 min on 3 h) into the cache."""
    fac = run.factors
    p = Params.from_json(run.path / "params.json")
    fc = corr.FactorCorr(fac, p.noise_freq)
    timer = StageTimer()
    kept: dict[str, np.ndarray] = {}
    s = pipeline._seeds(fac, fc, p, timer, keep=kept)
    A0, _ = pipeline.init_units(fac, fc, s, p, timer)
    tmp = run.cache.path("init_A.tmp.npz")
    sps.save_npz(tmp, sps.csr_matrix(A0, dtype=np.float32))
    tmp.replace(run.cache.dir / "init_A.npz")
    for k in SEED_STAGES:  # seeds last: their presence marks the computation done
        run.cache.save_npy(f"seeds_{k}.npy", kept[k])


def block_residual(fac: Factors, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Residual (dy, dx) and correlation of each corrected ``block_max`` vs the whole-run max.

    A consistent registration gives ~0 px for every block; a block locked to a slightly
    wrong offset shows up as a step here even when its per-frame quality looks normal.
    """
    ref = motion._mean_template(fac.max_proj[None].astype(np.float32), down=1)
    shifts, cor = [], []
    for i in range(0, len(fac.block_max), chunk):
        blk = fac.block_max[i : i + chunk].astype(np.float32)
        shifts.append(motion.estimate(blk, ref))
        cor.append(motion.frame_quality(blk, ref))
    return np.concatenate(shifts), np.concatenate(cor)


def residual(run: RunView) -> np.ndarray | None:
    return run.cache.load_npy("residual.npy") if run.cache.has("residual.npy") else None


def compute_residual(run: RunView) -> None:
    sh, cor = block_residual(run.factors)
    run.cache.save_npy("residual.npy", np.column_stack([sh, cor]).astype(np.float32))


def _overview_name(kind: str, bins: int) -> str:
    return f"overview_{kind}_{int(bins)}.npy"


def traces_overview(run: RunView, kind: str, bins: int) -> np.ndarray | None:
    name = _overview_name(kind, bins)
    return run.cache.load_npy(name) if run.cache.has(name) else None


def compute_overview(run: RunView, kind: str, bins: int) -> None:
    run.cache.save_npy(_overview_name(kind, bins), overview(run.traces(kind), bins))
