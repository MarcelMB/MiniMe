"""End-to-end run: raw AVI folder -> ``processed/<run_name>/`` (motion, factors, cnmf, report).

Stage order (spec section 5): probe, hash, motion sidecar check (reuse its shifts, or build a
template), pass 1, sidecar save, pass 2, factors save, store removal, then window 2 on the
factors: seeds_init, pnr_refine, gmm_refine (opt-in), seeds_merge, initA, initC, unit_merge, CNMF.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

import minime
from minime import corr, init, io, motion, pass1, pass2, seeds
from minime.cnmf import CnmfResult, run_cnmf
from minime.factors import Factors
from minime.params import Params
from minime.report import StageTimer, dir_bytes, peak_rss
from minime.tiles import TileStore, tile_grid

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def check_run_name(run_name: str) -> None:
    """A run name must be one plain path component: not empty, ``.``, ``..`` or with a separator.

    ``--name ""`` or ``--name .`` would otherwise resolve to ``processed/`` itself and the
    wipe below would delete every run in it; ``--name ../x`` would escape it.
    """
    seps = {"/", "\\", os.sep} | ({os.altsep} if os.altsep else set())
    if run_name in ("", ".", "..") or any(c in run_name for c in seps):
        raise ValueError(f"invalid run name {run_name!r}: use a single, plain directory name")


def out_dir_for(raw_dir: Path, run_name: str, out_root: Path | None = None) -> Path:
    """``(out_root or raw_dir/../processed) / run_name``.

    ``raw_dir`` is made absolute first: ``Path(".").parent`` is ``.``, which would put
    ``processed/`` inside the raw folder.
    """
    check_run_name(run_name)
    root = (
        Path(out_root)
        if out_root is not None
        else Path(os.path.abspath(raw_dir)).parent / "processed"
    )
    return root / run_name


def _check_out_dir(raw_dir: Path, out: Path) -> None:
    """Refuse an output directory that is, contains, or lies inside the raw folder.

    The output dir gets deleted, and the raw folder is read-only.
    """
    raw, o = Path(raw_dir).resolve(), Path(out).resolve()
    if raw == o or o in raw.parents:
        raise ValueError(f"output dir {o} would contain the raw dir {raw}; choose another")
    if raw in o.parents:
        raise ValueError(f"output dir {o} lies inside the raw dir {raw}; choose another")


def _is_minime_run(out: Path) -> bool:
    """``out`` holds a ``params.json`` or ``report.json`` that a minime run wrote."""
    for name, keys in (
        # older runs: caanfast, cafast, fastminian
        (
            "params.json",
            {"minime_version", "caanfast_version", "cafast_version", "fastminian_version"},
        ),
        ("report.json", {"stages"}),
    ):
        try:
            doc = json.loads((out / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and keys & set(doc):
            return True
    return False


def _check_wipe(out: Path) -> None:
    """Refuse to delete ``out`` unless it is missing, empty, or a previous minime run."""
    if not out.exists():
        return
    if not out.is_dir():
        raise ValueError(f"output path {out} exists and is not a directory")
    if any(out.iterdir()) and not _is_minime_run(out):
        raise ValueError(
            f"output dir {out} is not empty and holds no minime params.json / report.json; "
            "refusing to delete it (choose another --name)"
        )


def _reset_dir(out: Path) -> None:
    _check_wipe(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Window 1: motion, pass 1, pass 2
# ---------------------------------------------------------------------------


def _template_from_sidecar(
    sc: motion.MotionSidecar, info: io.RawInfo, params: Params
) -> motion.Template:
    """``Template`` rebuilt from the stored template image (used for frame quality only)."""
    return motion.Template(
        np.fft.rfft2(sc.template),
        params.motion_down,
        (info.height, info.width),
        (int(sc.template.shape[0]), int(sc.template.shape[1])),
    )


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


def _seeds(
    fac: Factors,
    fc: corr.FactorCorr,
    params: Params,
    timer: StageTimer,
    keep: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Seeds through init -> PNR (-> GMM) -> merge; ``keep`` gets each step's positions."""
    st: dict[str, pd.DataFrame] = {}
    s = seeds_merged(fac, fc, seeds_unmerged(fac, params, timer, st), params, timer, st)
    if keep is not None:
        keep.update(seed_positions(st))
    return s


def seed_positions(stages: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """``(n, 2)`` positions after init, PNR and merge from the ``seeds_unmerged`` tables."""
    return {
        "init": _positions(stages["seeds_init"]),
        "pnr": _positions(_filter(stages["seeds_pnr"], "mask_pnr")),
        "merged": _positions(stages["seeds_merged"]),
    }


def _positions(s: pd.DataFrame) -> np.ndarray:
    return s[["height", "width"]].to_numpy(np.float32).reshape(-1, 2)


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


@dataclass
class Window2Result:
    """Window 2 on the factors, with every intermediate stage the tuning figures show."""

    seeds: dict[str, pd.DataFrame]  # seeds_init, seeds_pnr (all, with pnr), seeds_merged
    footprints: dict[str, sps.csr_matrix]  # initA, init_merge, spatial_{i}, merge_{i}
    cnmf: CnmfResult


def run_window2(
    fac: Factors, params: Params, timer: StageTimer, fc: corr.FactorCorr | None = None
) -> Window2Result:
    """Seeds, initA/initC, unit merge and the CNMF loop on ``fac`` (``fc`` built if not given)."""
    if fc is None:
        with timer("corr_setup"):
            fc = corr.FactorCorr(fac, params.noise_freq)
    st: dict[str, pd.DataFrame] = {}
    fp: dict[str, sps.csr_matrix] = {}
    s = seeds_merged(fac, fc, seeds_unmerged(fac, params, timer, st), params, timer, st)
    A, C = init_units(fac, fc, s, params, timer, fp)
    with timer("cnmf"):
        res = run_cnmf(fac, A, C, params, fc.traces_corr, record=fp.__setitem__)
    timer.counts.update(res.counts)
    timer.counts["units_final"] = res.A.shape[0]
    return Window2Result(st, fp, res)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _write_seeds(d: Path, kept: dict[str, np.ndarray]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, pos in kept.items():
        np.save(d / f"{name}.npy", pos)


def _write_init(d: Path, A: sps.csr_matrix) -> None:
    d.mkdir(parents=True, exist_ok=True)
    sps.save_npz(d / "A.npz", sps.csr_matrix(A, dtype=np.float32))


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
        "minime_version": minime.__version__,
    }
    (out / "params.json").write_text(json.dumps(doc, indent=1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    raw_dir: Path,
    params: Params,
    run_name: str = "minime",
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
    _check_wipe(out)
    timer = StageTimer()

    with timer("probe"):
        info = io.probe(raw_dir)
    with timer("hash"):
        raw_hash = io.hash_files(info.segments)
    # read a valid sidecar into memory before the output dir is wiped
    sc = motion.load_if_valid(out / "motion", raw_hash, params.motion_params(), info.frames)
    _reset_dir(out)
    _write_params(out, params, raw_dir, raw_hash)  # first: marks the dir as a minime run
    timer.attach_progress(out / "progress.jsonl")

    with timer("template"):
        if sc is None:
            template = pass1.build_template(info, params)
            template_img = motion.template_image(template)
        else:
            template = _template_from_sidecar(sc, info, params)
            template_img = sc.template
    grid = tile_grid(info.height, info.width, params.tile)
    store = TileStore(out / "tiles", grid, pass1.n_parts_for(info, workers))
    try:
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
    finally:  # a failed run must not leave a raw-sized tile store behind
        if not keep_store:
            store.remove()

    with timer("corr_setup"):
        fc = corr.FactorCorr(fac, params.noise_freq)
    kept: dict[str, np.ndarray] = {}
    s = _seeds(fac, fc, params, timer, keep=kept)
    _write_seeds(out / "seeds", kept)
    A, C = init_units(fac, fc, s, params, timer)
    _write_init(out / "init", A)
    with timer("cnmf"):
        res = run_cnmf(fac, A, C, params, fc.traces_corr)
    timer.counts.update(res.counts)
    timer.counts["units_final"] = res.A.shape[0]

    with timer("write_outputs"):
        _write_cnmf(out / "cnmf", res, info.height, info.width)
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
            "version": minime.__version__,
        },
    )
    return out
