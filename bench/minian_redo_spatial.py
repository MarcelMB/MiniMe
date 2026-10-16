"""Re-run Minian's first ``update_spatial`` on the kept intermediates of a baseline run.

Run under the Minian venv (needs ``minian``, xarray, dask.distributed)::

    <minian_v2.0>/.venv/bin/python bench/minian_redo_spatial.py RUN_OUT_DIR --scratch SCRATCH_DIR

Why: ``run_minian_baseline.py --keep`` saves the normalised ``A_new`` with
``save_minian(A_new.rename("A_new"), intpath, overwrite=True)`` while that lazy array still
reads from ``A_new.zarr``, so the store ends up all NaN. This script recomputes ``A_new`` from
the unharmed ``Y_hw_chk``, ``A``, ``C`` and ``sn_spatial`` stores with the baseline's
parameters (dl_wnd 10, sparse_penal 0.01, size_thres (25, None)) and leaves it in
``SCRATCH_DIR/A_new.zarr`` (un-normalised; normalisation does not change centroids). The kept
intermediates are only read. Export with
``bench/minian_export.py RUN_OUT_DIR A_new C_new SCRATCH_DIR``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import xarray as xr

P_SPATIAL = {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)}


def _open(inter: Path, name: str) -> xr.DataArray:
    # consolidated=False: rechunker leaves stale consolidated metadata (wrong chunk grid)
    return xr.open_zarr(inter / f"{name}.zarr", consolidated=False)[name]


def redo(run_out: Path, scratch: Path, n_workers: int = 4, mem_limit: str = "4GB") -> None:
    import dask
    from dask.distributed import Client, LocalCluster

    inter = run_out / "minian_intermediate"
    scratch.mkdir(parents=True, exist_ok=True)
    link = scratch / "C.zarr"  # update_spatial reads C from $MINIAN_INTERMEDIATE/C.zarr/C
    if not link.exists():
        link.symlink_to((inter / "C.zarr").resolve())
    os.environ["MINIAN_INTERMEDIATE"] = str(scratch)
    from minian.cnmf import update_spatial
    from minian.utilities import TaskAnnotation

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
    try:
        Y_hw_chk, A, C, sn = (_open(inter, n) for n in ("Y_hw_chk", "A", "C", "sn_spatial"))
        t0 = time.time()
        A_new, mask, _norm_fac = update_spatial(Y_hw_chk, A, C, sn, **P_SPATIAL)
        C_new = _open(inter, "C_new")
        same = bool((C_new.unit_id.values == A_new.unit_id.values).all())
        print(
            f"kept {int(mask.sum())} of {mask.size} units in {time.time() - t0:.0f} s; "
            f"unit ids match C_new: {same}",
            flush=True,
        )
    finally:
        client.close()
        cluster.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_out", type=Path, help="baseline output dir holding minian_intermediate/")
    p.add_argument("--scratch", type=Path, required=True, help="dir that receives A_new.zarr")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mem-limit", default="4GB")
    a = p.parse_args()
    redo(a.run_out, a.scratch, a.workers, a.mem_limit)


if __name__ == "__main__":
    main()
