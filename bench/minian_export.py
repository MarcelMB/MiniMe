"""Export Minian's units from its zarr intermediates into a plain ``minian_units.npz``.

Run under the Minian venv (needs xarray + zarr), not the caanfast venv::

    <minian_v2.0>/.venv/bin/python bench/minian_export.py RUN_OUT_DIR

``RUN_OUT_DIR`` holds ``minian_intermediate/{A_new,C_new}.zarr`` and ``minian/motion.zarr``.
Writes ``RUN_OUT_DIR/minian_units.npz`` with ``A (K, H*W) float32`` (dense, row-major),
``C (K, T) float32`` (Minian's trace after the spatial update) and ``shifts (T, 2) float32``
in (height, width) order. Optional extra arguments ``A_NAME C_NAME`` export another pair of
intermediates (e.g. ``A C``) to ``minian_units_<A_NAME>.npz``; a third ``A_DIR`` reads
``A_NAME.zarr`` from that directory instead of ``minian_intermediate`` (``consolidated=False``
because rechunker leaves stale consolidated metadata behind).

``--final MINIAN_DIR [OUT_NPZ]`` exports a finished Minian output folder (``A.zarr``,
``C.zarr``, ``max_proj.zarr``) to ``A, C, max_proj`` for ``bench/compare_minian.py``
(default ``MINIAN_DIR/../minian_final.npz``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


def export(
    run_out: Path, a_name: str = "A_new", c_name: str = "C_new", a_dir: Path | None = None
) -> Path:
    inter = run_out / "minian_intermediate"
    A = xr.open_zarr(Path(a_dir or inter) / f"{a_name}.zarr", consolidated=False)[a_name].transpose(
        "unit_id", "height", "width"
    )
    C = xr.open_zarr(inter / f"{c_name}.zarr", consolidated=False)[c_name].transpose(
        "unit_id", "frame"
    )
    motion = xr.open_zarr(run_out / "minian" / "motion.zarr", consolidated=False)[
        "motion"
    ].transpose("frame", "shift_dim")
    dims = [str(d) for d in motion["shift_dim"].values]
    if dims != ["height", "width"]:
        raise ValueError(f"unexpected shift_dim order {dims}")
    K, h, w = A.shape
    A_dense = A.values.reshape(K, h * w).astype(np.float32)
    if np.isnan(A_dense).any():
        raise ValueError(f"{a_name} contains NaN (store clobbered or incomplete); not exporting")
    out = run_out / ("minian_units.npz" if a_name == "A_new" else f"minian_units_{a_name}.npz")
    np.savez(
        out,
        A=A_dense,
        C=C.values.astype(np.float32),
        shifts=motion.values.astype(np.float32),
    )
    return out


def export_final(minian_dir: Path, out: Path | None = None) -> Path:
    """``A (K, H*W)``, ``C (K, T)``, ``max_proj (H, W)`` of a finished Minian output folder."""

    def var(name: str) -> xr.DataArray:
        return xr.open_zarr(minian_dir / f"{name}.zarr", consolidated=False)[name]

    A = var("A").transpose("unit_id", "height", "width")
    K, h, w = A.shape
    out = out or minian_dir.parent / "minian_final.npz"
    np.savez(
        out,
        A=A.values.reshape(K, h * w).astype(np.float32),
        C=var("C").transpose("unit_id", "frame").values.astype(np.float32),
        max_proj=var("max_proj").transpose("height", "width").values.astype(np.float32),
    )
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--final":
        print(export_final(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None))
        sys.exit()
    # optional: A_NAME C_NAME [A_DIR]; A_DIR holds a re-computed A_new.zarr (see README)
    names = sys.argv[2:4]
    a_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    print(export(Path(sys.argv[1]), *names, a_dir=a_dir))
