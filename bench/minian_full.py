"""Time the full Minian v2.0 pipeline on raw AVIs with a cafast run's parameters.

Run under the Minian venv (needs minian + dask), not the cafast venv::

    <minian_v2.0>/.venv/bin/python bench/minian_full.py RAW_DIR CAFAST_PARAMS_JSON OUT_DIR

Follows ``pipeline.ipynb``: load, glow removal + median denoise + tophat, motion, seeds,
PNR, seed merge, initA/initC, unit merge, background, noise, then n_iter x [spatial,
background, traces, merge]. ``ks_refine`` and the temporal update are skipped, as cafast
has neither. Parameters come from the cafast ``params.json`` (``sparse_penal`` =
``sparse_snr / sqrt(T)``); cafast's motion-before-tophat and tree seed merge have no
Minian equivalent, Minian's own order and connected-component merge are used.

Inputs that are no longer needed (``varr`` after preprocessing, ``varr_ref`` after the
motion warp) are deleted to keep the peak disk down. A monitor thread records peak disk of
``OUT_DIR`` and peak RSS of this process tree every ``--poll`` s and aborts if free disk
falls under ``--min-free-gb``. Writes ``OUT_DIR/timing.json`` after every stage (so a
crash keeps the stages that finished) and ``OUT_DIR/minian_units.npz`` at the end.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import psutil  # noqa: E402

# ---------------------------------------------------------------------------
# Monitor + timer
# ---------------------------------------------------------------------------


def dir_bytes(path: Path) -> int:
    total, stack = 0, [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        else:
                            total += e.stat(follow_symlinks=False).st_blocks * 512
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass
    return total


def tree_rss() -> int:
    me = psutil.Process()
    total = 0
    for p in [me, *me.children(recursive=True)]:
        with contextlib.suppress(psutil.Error):
            total += p.memory_info().rss
    return total


class Monitor(threading.Thread):
    def __init__(self, path: Path, min_free_gb: float, poll: float) -> None:
        super().__init__(daemon=True)
        self.path, self.min_free, self.poll = path, min_free_gb * 1e9, poll
        self.stage = "start"
        self.peak_disk = self.peak_rss = 0
        self.stage_disk: dict[str, int] = {}
        self.stage_rss: dict[str, int] = {}
        self.min_free_seen = float("inf")
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            size, rss = dir_bytes(self.path), tree_rss()
            free = shutil.disk_usage(self.path).free
            self.peak_disk, self.peak_rss = max(self.peak_disk, size), max(self.peak_rss, rss)
            self.stage_disk[self.stage] = max(self.stage_disk.get(self.stage, 0), size)
            self.stage_rss[self.stage] = max(self.stage_rss.get(self.stage, 0), rss)
            self.min_free_seen = min(self.min_free_seen, free)
            if free < self.min_free:
                print(f"!! free disk {free / 1e9:.1f} GB < limit in {self.stage}, aborting")
                os._exit(3)
            self._halt.wait(self.poll)

    def stop(self) -> None:
        self._halt.set()


class Timer:
    def __init__(self, mon: Monitor, out: Path, header: dict[str, Any]) -> None:
        self.mon, self.out, self.header = mon, out, header
        self.rows: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}
        self.t0 = time.perf_counter()

    def __call__(self, name: str, fn: Callable[[], Any]) -> Any:
        self.mon.stage = name
        print(f"{time.strftime('%H:%M:%S')} {name} ...", flush=True)
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        self.rows.append({"stage": name, "seconds": dt, "disk_after_bytes": dir_bytes(self.out)})
        print(f"{time.strftime('%H:%M:%S')} {name:<28s} {dt:9.1f} s", flush=True)
        self.dump("running")
        return res

    def dump(self, status: str) -> None:
        doc = {
            **self.header,
            "status": status,
            "elapsed_seconds": time.perf_counter() - self.t0,
            "stages": self.rows,
            "peak_disk_bytes": self.mon.peak_disk,
            "peak_rss_bytes": self.mon.peak_rss,
            "stage_peak_disk_bytes": self.mon.stage_disk,
            "stage_peak_rss_bytes": self.mon.stage_rss,
            "free_disk_min_bytes": self.mon.min_free_seen,
            **self.info,
        }
        (self.out / "timing.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def minian_params(cp: dict[str, Any], T: int) -> dict[str, Any]:
    """Minian keyword arguments per step from a cafast ``params.json``."""
    nf = cp["noise_freq"]
    return {
        "denoise": {"method": "median", "ksize": cp["median_ksize"]},
        "background": {"method": "tophat", "wnd": cp["tophat_wnd"]},
        "motion": {"dim": "frame"},
        "seeds_init": {
            "wnd_size": cp["seeds_wnd"],
            "method": "rolling",
            "stp_size": cp["seeds_stp"],
            "max_wnd": cp["seeds_max_wnd"],
            "diff_thres": cp["seeds_diff_thres"],
        },
        "pnr_refine": {"noise_freq": nf, "thres": cp["pnr_thres"]},
        "seeds_merge": {
            "thres_dist": cp["merge_dist"],
            "thres_corr": cp["merge_corr"],
            "noise_freq": nf,
        },
        "initialize": {"thres_corr": cp["inita_corr"], "wnd": cp["inita_wnd"], "noise_freq": nf},
        "init_merge": cp["init_merge_corr"],
        "loop_merge": cp["loop_merge_corr"],
        "get_noise": {"noise_range": (nf, 0.5)},
        "spatial": {
            "dl_wnd": cp["dl_wnd"],
            "sparse_penal": cp["sparse_snr"] / math.sqrt(T),
            "size_thres": tuple(cp["size_thres"]),
        },
        "n_iter": cp["n_iter"],
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(raw: Path, cp: dict[str, Any], out: Path, a: argparse.Namespace, tm: Timer) -> None:
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
    from minian.initialization import initA, initC, pnr_refine, seeds_init, seeds_merge
    from minian.motion_correction import apply_transform, estimate_motion
    from minian.preprocessing import denoise, remove_background
    from minian.utilities import TaskAnnotation, get_optimal_chk, load_videos, save_minian

    intpath = out / "minian_intermediate"
    os.environ["MINIAN_INTERMEDIATE"] = str(intpath)
    mem = "1GB"  # rechunk memory limit, as in pipeline.ipynb

    def save(x, name: str, **kw):
        return save_minian(x.rename(name), str(intpath), overwrite=True, **kw)

    def drop(name: str) -> None:
        shutil.rmtree(intpath / f"{name}.zarr", ignore_errors=True)

    dask.config.set({"distributed.worker.memory.terminate": False})
    cluster = LocalCluster(
        n_workers=a.workers,
        memory_limit=a.mem_limit,
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=None,
    )
    cluster.scheduler.add_plugin(TaskAnnotation())
    client = Client(cluster)
    info = tm.info
    try:
        # --- load + preprocessing --------------------------------------------
        def _load() -> tuple:
            varr = load_videos(str(raw), pattern=r"[0-9]+\.avi$", dtype=np.uint8)
            chk, _ = get_optimal_chk(varr, dtype=float)
            return save(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}), "varr"), chk

        varr, chk = tm("load_videos+save_varr", _load)
        T = int(varr.sizes["frame"])
        prm = minian_params(cp, T)
        info.update(shape=dict(varr.sizes), chk=chk, minian_params=prm)

        def _pre():
            v = varr - varr.min("frame").compute()
            v = denoise(v, **prm["denoise"])
            return save(remove_background(v, **prm["background"]), "varr_ref")

        varr_ref = tm("glow+denoise+tophat+save", _pre)
        drop("varr")

        # --- motion ----------------------------------------------------------
        motion = tm(
            "estimate_motion+save",
            lambda: save(
                estimate_motion(varr_ref, **prm["motion"]).chunk({"frame": chk["frame"]}), "motion"
            ),
        )
        Y_fm = tm(
            "apply_transform+save_Y_fm",
            lambda: save(apply_transform(varr_ref, motion, fill=0).astype(np.float32), "Y_fm_chk"),
        )
        drop("varr_ref")
        hw = {"frame": -1, "height": chk["height"], "width": chk["width"]}
        Y_hw = tm("rechunk_Y_hw", lambda: save(Y_fm, "Y_hw_chk", mem_limit=mem, chunks=hw))

        # --- seeds + initialisation -------------------------------------------
        max_proj = tm("max_proj", lambda: save(Y_fm.max("frame"), "max_proj").compute())
        seeds = tm("seeds_init", lambda: seeds_init(Y_fm, **prm["seeds_init"]))
        info["seeds_init"] = len(seeds)
        seeds, _, _ = tm("pnr_refine", lambda: pnr_refine(Y_hw, seeds, **prm["pnr_refine"]))
        seeds = seeds[seeds["mask_pnr"]].reset_index(drop=True)
        info["seeds_pnr"] = len(seeds)
        seeds = tm("seeds_merge", lambda: seeds_merge(Y_hw, max_proj, seeds, **prm["seeds_merge"]))
        seeds = seeds[seeds["mask_mrg"]].reset_index(drop=True)
        info["seeds_merged"] = len(seeds)
        A = tm("initA+save", lambda: save(initA(Y_hw, seeds, **prm["initialize"]), "A_init"))
        uc = {"unit_id": 1, "frame": -1}
        C = tm("initC+save", lambda: save(initC(Y_fm, A), "C_init", mem_limit=mem, chunks=uc))

        def _merge(A, C, thres: float, tag: str) -> tuple:
            A, C = unit_merge(A, C, thres_corr=thres)[:2]
            A, C = save(A, f"A_mrg_{tag}"), save(C, f"C_mrg_{tag}")
            fc = {"unit_id": -1, "frame": chk["frame"]}
            return A, C, save(C, f"C_chk_mrg_{tag}", mem_limit=mem, chunks=fc)

        A, C, C_chk = tm("unit_merge_init", lambda: _merge(A, C, prm["init_merge"], "init"))
        info["units_init_merge"] = int(A.sizes["unit_id"])

        def _bg(A, C_chk, tag: str) -> tuple:
            b, f = update_background(Y_fm, A, C_chk)
            return save(b, f"b_{tag}"), save(f, f"f_{tag}")

        tm("update_background_init", lambda: _bg(A, C_chk, "init"))
        sn = tm(
            "get_noise_fft", lambda: save(get_noise_fft(Y_hw, **prm["get_noise"]), "sn_spatial")
        )

        # --- CNMF: n_iter x (spatial, background, traces, merge) ---------------
        def _iteration(it: int, A, C, C_chk) -> tuple:
            def _spatial() -> tuple:
                A_new, mask, nf = update_spatial(Y_hw, A, C, sn, **prm["spatial"])
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
            A, C, C_chk = tm(
                f"unit_merge_{it}", lambda: _merge(A, YrA, prm["loop_merge"], f"m{it}")
            )
            info[f"units_after_merge_{it}"] = int(A.sizes["unit_id"])
            return A, C, C_chk

        for it in range(prm["n_iter"]):
            A, C, C_chk = _iteration(it, A, C, C_chk)

        K = int(A.sizes["unit_id"])
        np.savez(
            out / "minian_units.npz",
            A=A.transpose("unit_id", "height", "width").values.reshape(K, -1).astype(np.float32),
            C=C.transpose("unit_id", "frame").values.astype(np.float32),
            max_proj=np.asarray(max_proj.values, np.float32),
            shifts=motion.transpose("frame", "shift_dim").values.astype(np.float32),
        )
        info["units_final"] = K
    finally:
        client.close()
        cluster.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("raw", type=Path)
    ap.add_argument("params", type=Path, help="cafast params.json")
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem-limit", default="4GB")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--keep", action="store_true", help="keep intermediates")
    a = ap.parse_args()

    cp = json.loads(a.params.read_text())
    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    mon = Monitor(a.out, a.min_free_gb, a.poll)
    mon.start()
    header = {
        "pipeline": "minian v2.0 pipeline.ipynb order; cafast params; no ks_refine, no temporal",
        "cluster": f"LocalCluster {a.workers} workers x 2 threads, {a.mem_limit}/worker",
        "raw": str(a.raw),
        "cafast_params": str(a.params),
    }
    tm = Timer(mon, a.out, header)
    status = "failed"
    try:
        run(a.raw, cp, a.out, a, tm)
        status = "done"
    finally:
        mon.stop()
        tm.dump(status)
        print(f"{status}: {time.perf_counter() - tm.t0:.0f} s -> {a.out / 'timing.json'}")
        if not a.keep:
            shutil.rmtree(a.out / "minian_intermediate", ignore_errors=True)


if __name__ == "__main__":
    main()
