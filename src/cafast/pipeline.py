"""End-to-end run: raw AVI folder -> ``processed/<run_name>/`` (motion, factors, cnmf, report).

Stage order (spec section 5): probe, hash, motion sidecar check (reuse its shifts, or build a
template), pass 1, sidecar save, pass 2, factors save, store removal, then window 2 on the
factors: seeds_init, pnr_refine, gmm_refine (opt-in), seeds_merge, initA, initC, unit_merge, CNMF.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

import cafast
from cafast import corr, init, io, motion, pass1, pass2, seeds
from cafast.cnmf import CnmfResult, run_cnmf
from cafast.factors import Factors
from cafast.params import Params
from cafast.report import StageTimer, dir_bytes, peak_rss
from cafast.tiles import TileStore, tile_grid

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def out_dir_for(raw_dir: Path, run_name: str, out_root: Path | None = None) -> Path:
    """``(out_root or raw_dir/../processed) / run_name``."""
    root = Path(out_root) if out_root is not None else Path(raw_dir).parent / "processed"
    return root / run_name


def _check_out_dir(raw_dir: Path, out: Path) -> None:
    """Refuse an output directory that is, or contains, the raw folder (it gets deleted)."""
    raw, o = Path(raw_dir).resolve(), Path(out).resolve()
    if raw == o or o in raw.parents:
        raise ValueError(f"output dir {o} would contain the raw dir {raw}; choose another")


def _reset_dir(out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Window 1: motion, pass 1, pass 2
# ---------------------------------------------------------------------------


def _template_from_sidecar(
    sc: motion.MotionSidecar, info: io.RawInfo, params: Params
) -> motion.Template:
    """``Template`` rebuilt from the stored template image (used for frame quality only)."""
    return motion.Template(np.fft.rfft2(sc.template), params.motion_down, (info.height, info.width))


def _save_sidecar(
    out: Path,
    p1: pass1.Pass1Result,
    template_img: np.ndarray,
    params: Params,
    raw_hash: str,
    workers: int,
) -> None:
    motion.save_sidecar(
        out / "motion",
        motion.MotionSidecar(
            shifts=p1.shifts,
            template=template_img,
            quality=p1.quality,
            params=params.motion_params(),
            raw_hash=raw_hash,
            meta={"workers": workers, "pass1_seconds": p1.seconds},
        ),
    )


# ---------------------------------------------------------------------------
# Window 2: seeds and initialisation on factors
# ---------------------------------------------------------------------------


def _filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[df[col].to_numpy(bool)].reset_index(drop=True)


def _seeds(fac: Factors, fc: corr.FactorCorr, params: Params, timer: StageTimer) -> pd.DataFrame:
    p, n = params, timer.counts
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
    with timer("pnr_refine"):
        s = _filter(seeds.pnr_refine(fac, s, thres=p.pnr_thres), "mask_pnr")
    n["seeds_pnr"] = len(s)
    if p.gmm_refine:
        with timer("gmm_refine"):
            s = _filter(seeds.gmm_refine(fac, s), "mask_gmm")
    n["seeds_gmm"] = len(s)  # equals seeds_pnr when gmm_refine is off
    with timer("seeds_merge"):
        s = _filter(corr.seeds_merge(fc, s, fac.max_proj, p.merge_dist, p.merge_corr), "mask_mrg")
    n["seeds_merged"] = len(s)
    return s


def _init_units(
    fac: Factors, fc: corr.FactorCorr, s: pd.DataFrame, params: Params, timer: StageTimer
) -> tuple[sps.csr_matrix, np.ndarray]:
    with timer("initA"):
        A = init.initA(fc, s, params.inita_corr, params.inita_wnd)
    timer.counts["units_initA"] = A.shape[0]
    with timer("initC"):
        C = init.initC(fac, A)
    with timer("unit_merge"):
        A, C, _ = init.unit_merge(A, C, fc.traces_corr, params.init_merge_corr)
    timer.counts["units_init_merge"] = A.shape[0]
    return A, C


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _write_cnmf(d: Path, res: CnmfResult, H: int, W: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    A = sps.csr_matrix(res.A, dtype=np.float32)
    sps.save_npz(d / "A.npz", A)
    np.save(d / "C.npy", np.asarray(res.C, dtype=np.float32))
    np.save(d / "YrA.npy", np.asarray(res.YrA, dtype=np.float32))
    np.save(d / "b.npy", np.asarray(res.b, dtype=np.float32).reshape(H, W))
    np.save(d / "f.npy", np.asarray(res.f, dtype=np.float32))
    cent = init.centroids(A, H, W) if A.shape[0] else np.zeros((0, 2))
    peak = res.YrA.max(axis=1) if A.shape[0] else np.zeros(0, np.float32)
    pd.DataFrame(
        {
            "unit_id": np.arange(A.shape[0]),
            "centroid_h": cent[:, 0],
            "centroid_w": cent[:, 1],
            "size_px": np.diff(A.indptr),
            "peak": peak,
        }
    ).to_csv(d / "units.csv", index=False)


def _write_params(out: Path, params: Params, raw_dir: Path, raw_hash: str) -> None:
    doc = {
        **params.asdict(),
        "raw_dir": str(Path(raw_dir).resolve()),
        "raw_hash": raw_hash,
        "cafast_version": cafast.__version__,
    }
    (out / "params.json").write_text(json.dumps(doc, indent=1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    raw_dir: Path,
    params: Params,
    run_name: str = "cafast",
    out_root: Path | None = None,
    keep_store: bool = False,
    workers: int | None = None,
) -> Path:
    """Run the whole pipeline on ``raw_dir``; returns the (fresh) output directory."""
    t_start = time.perf_counter()
    raw_dir = Path(raw_dir)
    workers = params.workers if workers is None else int(workers)
    out = out_dir_for(raw_dir, run_name, out_root)
    _check_out_dir(raw_dir, out)
    timer = StageTimer()

    with timer("probe"):
        info = io.probe(raw_dir)
    with timer("hash"):
        raw_hash = io.hash_files(info.segments)
    # read a valid sidecar into memory before the output dir is wiped
    sc = motion.load_if_valid(out / "motion", raw_hash, params.motion_params(), info.frames)
    _reset_dir(out)

    with timer("template"):
        if sc is None:
            template = pass1.build_template(info, params)
            template_img = motion.template_image(template)
        else:
            template = _template_from_sidecar(sc, info, params)
            template_img = sc.template
    grid = tile_grid(info.height, info.width, params.tile)
    store = TileStore(out / "tiles", grid, pass1.n_parts_for(info, workers))
    with timer("pass1"):
        p1 = pass1.run_pass1(
            info, store, template, params, workers, shifts=None if sc is None else sc.shifts
        )
    with timer("sidecar_save"):
        _save_sidecar(out, p1, template_img, params, raw_hash, workers)
    with timer("pass2"):
        tiles = pass2.run_pass2(store, grid, params, workers)
    fac = pass2.build_factors(info, grid, p1, tiles, params)
    with timer("factors_save"):
        fac.save(out / "factors")
    store_peak = store.nbytes()
    if not keep_store:
        store.remove()

    with timer("corr_setup"):
        fc = corr.FactorCorr(fac, params.noise_freq)
    s = _seeds(fac, fc, params, timer)
    A, C = _init_units(fac, fc, s, params, timer)
    with timer("cnmf"):
        res = run_cnmf(fac, A, C, params, fc.traces_corr)
    timer.counts.update(res.counts)
    timer.counts["units_final"] = res.A.shape[0]

    with timer("write_outputs"):
        _write_cnmf(out / "cnmf", res, info.height, info.width)
        _write_params(out, params, raw_dir, raw_hash)
    timer.to_json(
        out / "report.json",
        {
            "motion": {
                "sidecar_reused": sc is not None,
                "quality_median": float(np.median(p1.quality)),
                "max_abs_shift_px": float(np.abs(p1.shifts).max()),
            },
            "bytes": {
                "raw": int(sum(p.stat().st_size for p in info.segments)),
                "store_peak": int(store_peak),
                "factors": dir_bytes(out / "factors"),
                "cnmf": dir_bytes(out / "cnmf"),
            },
            "pass1_cpu_seconds": p1.seconds,
            "peak_rss": peak_rss(),
            "peak_rss_children": peak_rss(children=True),
            "wall_total": time.perf_counter() - t_start,
            "workers": workers,
            "version": cafast.__version__,
        },
    )
    return out
