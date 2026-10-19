"""Time Minian v2.0 on a real session with that session's parameter JSON.

Run under the Minian venv (needs minian + dask), not the cafast venv::

    <minian_v2.0>/.venv/bin/python bench/minian_real.py VIDEO PARAMS_JSON OUT_DIR

``VIDEO`` is the Minian input movie (already preprocessed: Minian's own glow removal,
denoise and tophat are skipped, as in the modified WL25 pipeline). ``PARAMS_JSON`` is the
session's Minian parameter file (``parameters.{seeds_init, pnr_refine, ks_refine,
seeds_merge, initialize, init_merge, get_noise, first_spatial, first_merge,
second_spatial, second_merge}``). Stages match cafast's: motion, seeds, PNR (+ ks_refine,
timed on its own), seed merge, initA/initC, unit merge, background, then
2 x [spatial, background, traces, merge]. No temporal update, no ``drop_correlated_pairs``
or post-hoc filters. Cluster as in ``run_minian_baseline.py``: 4 workers x 2 threads, 4 GB.

Writes ``OUT_DIR/timing.json`` (per-stage seconds, unit counts, peak disk) and
``OUT_DIR/minian_units.npz`` (``A (K, H*W)``, ``C (K, T)`` = final traces, ``max_proj``,
``shifts (T, 2)``). Intermediates go to ``OUT_DIR/minian_intermediate`` and are deleted
unless ``--keep``.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import minian  # noqa: E402

# DiskMonitor / Timer live next to the synthetic Minian baseline in the minian_v2.0 checkout
sys.path.insert(0, str(Path(minian.__file__).resolve().parents[1] / "scripts" / "bench_browser"))
from run_minian_baseline import DiskMonitor, Timer  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(video: Path, prm: dict, out: Path, n_workers: int, mem_limit: str, tm: Timer) -> dict:
    import dask
    import numpy as np
    from dask.distributed import Client, LocalCluster
    from minian.cnmf import (
        compute_trace,
        get_noise_fft,
        unit_merge,
        update_background,
        update_spatial,
    )
    from minian.initialization import (
        initA,
        initC,
        ks_refine,
        pnr_refine,
        seeds_init,
        seeds_merge,
    )
    from minian.motion_correction import apply_transform, estimate_motion
    from minian.utilities import TaskAnnotation, get_optimal_chk, load_videos, save_minian

    intpath = str(out / "minian_intermediate")
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    mem = "1GB"  # rechunk memory limit, as in run_minian_baseline.py

    def save(x, name: str, **kw):
        return save_minian(x.rename(name), intpath, overwrite=True, **kw)

    dask.config.set({"distributed.worker.memory.terminate": False})
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit=mem_limit,
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=None,
    )
    cluster.scheduler.add_plugin(TaskAnnotation())
    client = Client(cluster)
    info: dict = {}
    try:
        # --- load + motion ----------------------------------------------------
        def _load() -> tuple:
            varr = load_videos(
                str(video.parent), pattern=re.escape(video.name) + "$", dtype=np.uint8
            )
            chk, _ = get_optimal_chk(varr, dtype=float)
            varr = save(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}), "varr")
            return varr, chk

        varr, chk = tm("load_videos+save_varr", _load)
        info["shape"] = dict(varr.sizes)
        motion = tm(
            "estimate_motion+save",
            lambda: save(
                estimate_motion(varr, dim="frame").chunk({"frame": chk["frame"]}), "motion"
            ),
        )
        Y_fm = tm(
            "apply_transform+save_Y_fm",
            lambda: save(apply_transform(varr, motion, fill=0).astype(np.float32), "Y_fm_chk"),
        )
        hw = {"frame": -1, "height": chk["height"], "width": chk["width"]}
        Y_hw = tm("rechunk_Y_hw", lambda: save(Y_fm, "Y_hw_chk", mem_limit=mem, chunks=hw))

        # --- seeds + initialisation --------------------------------------------
        max_proj = tm("max_proj", lambda: save(Y_fm.max("frame"), "max_proj").compute())
        seeds = tm("seeds_init", lambda: seeds_init(Y_fm, **prm["seeds_init"]))
        info["seeds_init"] = len(seeds)
        seeds, _, _ = tm("pnr_refine", lambda: pnr_refine(Y_hw, seeds, **prm["pnr_refine"]))
        seeds = seeds[seeds["mask_pnr"]].reset_index(drop=True)
        info["seeds_pnr"] = len(seeds)
        seeds = tm("ks_refine", lambda: ks_refine(Y_hw, seeds, **prm["ks_refine"]))
        seeds = seeds[seeds["mask_ks"]].reset_index(drop=True)
        info["seeds_ks"] = len(seeds)
        seeds = tm("seeds_merge", lambda: seeds_merge(Y_hw, max_proj, seeds, **prm["seeds_merge"]))
        seeds = seeds[seeds["mask_mrg"]].reset_index(drop=True)
        info["seeds_merged"] = len(seeds)
        A = tm("initA+save", lambda: save(initA(Y_hw, seeds, **prm["initialize"]), "A_init"))
        uc = {"unit_id": 1, "frame": -1}
        C = tm("initC+save", lambda: save(initC(Y_fm, A), "C_init", mem_limit=mem, chunks=uc))

        def _merge(A, C, thres: float, tag: str) -> tuple:
            A, C = unit_merge(A, C, thres_corr=thres)[:2]
            # "mrg_" keeps these off the stores A and C are lazily read from (A_init, C_init):
            # save_minian(overwrite=True) deletes its target before computing into it
            A, C = save(A, f"A_mrg_{tag}"), save(C, f"C_mrg_{tag}")
            fc = {"unit_id": -1, "frame": chk["frame"]}
            return A, C, save(C, f"C_chk_mrg_{tag}", mem_limit=mem, chunks=fc)

        init_thres = prm["init_merge"]["thres_corr"]
        A, C, C_chk = tm("unit_merge_init", lambda: _merge(A, C, init_thres, "init"))
        info["units_init_merge"] = int(A.sizes["unit_id"])

        def _bg(A, C_chk, tag: str) -> tuple:
            b, f = update_background(Y_fm, A, C_chk)
            return save(b, f"b_{tag}"), save(f, f"f_{tag}")

        tm("update_background_init", lambda: _bg(A, C_chk, "init"))
        sn = tm(
            "get_noise_fft",
            lambda: save(get_noise_fft(Y_hw, **prm["get_noise"]), "sn_spatial"),
        )

        # --- CNMF: 2 x (spatial, background, traces, merge) ---------------------
        def _iteration(it: int, sp: dict, merge_thres: float, A, C, C_chk) -> tuple:
            def _spatial() -> tuple:
                A_new, mask, nf = update_spatial(Y_hw, A, C, sn, **sp)
                C_new = save(C_chk.sel(unit_id=mask) * nf, f"C_chk_sp{it}")
                return save(A_new, f"A_sp{it}"), C_new

            A, C_chk = tm(f"update_spatial_{it}", _spatial)
            info[f"units_after_spatial_{it}"] = int(A.sizes["unit_id"])
            b, f = tm(f"update_background_{it}", lambda: _bg(A, C_chk, f"sp{it}"))
            YrA = tm(
                f"compute_trace_{it}",
                lambda: save(
                    compute_trace(Y_fm, A, b, C_chk, f), f"YrA_{it}", mem_limit=mem, chunks=uc
                ),
            )
            A, C, C_chk = tm(f"unit_merge_{it}", lambda: _merge(A, YrA, merge_thres, f"m{it}"))
            info[f"units_after_merge_{it}"] = int(A.sizes["unit_id"])
            return A, C, C_chk

        for it, (sp, mg) in enumerate(
            (("first_spatial", "first_merge"), ("second_spatial", "second_merge"))
        ):
            A, C, C_chk = _iteration(it, prm[sp], prm[mg]["thres_corr"], A, C, C_chk)

        K = int(A.sizes["unit_id"])
        At = A.transpose("unit_id", "height", "width")
        np.savez(
            out / "minian_units.npz",
            A=At.values.reshape(K, -1).astype(np.float32),
            C=C.transpose("unit_id", "frame").values.astype(np.float32),
            max_proj=np.asarray(max_proj.values, np.float32),
            shifts=motion.transpose("frame", "shift_dim").values.astype(np.float32),
        )
        info["units_final"] = K
    finally:
        client.close()
        cluster.close()
    return info


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("video", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem-limit", default="4GB")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--keep", action="store_true", help="keep intermediates")
    a = ap.parse_args()

    prm = json.loads(a.params.read_text())["parameters"]
    for k in ("first_spatial", "second_spatial"):
        prm[k]["size_thres"] = tuple(prm[k]["size_thres"])
    prm["get_noise"]["noise_range"] = tuple(prm["get_noise"]["noise_range"])

    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    mon = DiskMonitor(a.out, a.min_free_gb)
    mon.start()
    tm = Timer(mon)
    t0 = time.perf_counter()
    info = run(a.video, prm, a.out, a.workers, a.mem_limit, tm)
    total = time.perf_counter() - t0
    mon.stop()
    mon.join()
    res = {
        "pipeline": "minian v2.0 (local), WL25 params; no Minian preprocessing, no temporal update",
        "cluster": f"LocalCluster {a.workers} workers x 2 threads, {a.mem_limit}/worker",
        "video": str(a.video),
        "params": str(a.params),
        "total_seconds": total,
        "stages": tm.rows,
        "peak_disk_bytes": mon.peak,
        **info,
    }
    (a.out / "timing.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"total {total:.1f} s, peak disk {mon.peak / 1e9:.2f} GB -> {a.out / 'timing.json'}")
    if not a.keep:
        shutil.rmtree(a.out / "minian_intermediate", ignore_errors=True)


if __name__ == "__main__":
    main()
