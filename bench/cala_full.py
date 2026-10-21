"""Time cala (Aharoni-Lab/cala, online OnACID-style) on raw AVIs at full field of view.

Run under a cala venv (cala main + its pinned ``noob``), not the cafast venv::

    <cala>/.venv/bin/python bench/cala_full.py RAW_DIR OUT_DIR [--max-files 1]

The tube is ``bench/cala_tube.yaml``: cala's ``long_recording.yaml`` without its crop to
100 x 100 px and its 20 -> 5 fps downsample, so every frame of the full field of view goes
through preprocessing, the per-frame trace / statistics / footprint update and detection.
cala has no batch stages; the run is one loop over frames, timed in blocks of
``--block`` frames together with the number of components (per-frame cost grows with K).

A monitor thread records peak RSS of this process and the size of ``OUT_DIR`` (the trace
zarr grows with T) every ``--poll`` s. ``OUT_DIR/timing.json`` is rewritten after every
block, so an interrupted run keeps its throughput curve. At the end the footprints and the
full traces are written to ``OUT_DIR/cala_units.npz`` as ``A (K, H*W)``, ``C (K, T)`` and
``max_proj (H, W)`` of the preprocessed frames, the format ``bench/compare_minian.py``
reads. ``C`` is NaN before a component was detected (cala does not back-fill).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import yaml
from natsort import natsorted

TUBE = Path(__file__).with_name("cala_tube.yaml")

# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            with contextlib.suppress(FileNotFoundError):
                total += os.stat(os.path.join(root, f)).st_blocks * 512
    return total


class Monitor(threading.Thread):
    def __init__(self, path: Path, poll: float) -> None:
        super().__init__(daemon=True)
        self.path, self.poll = path, poll
        self.peak_rss = self.peak_disk = 0
        self._halt = threading.Event()

    def run(self) -> None:
        me = psutil.Process()
        while not self._halt.is_set():
            with contextlib.suppress(psutil.Error):
                self.peak_rss = max(self.peak_rss, me.memory_info().rss)
            self.peak_disk = max(self.peak_disk, dir_bytes(self.path))
            self._halt.wait(self.poll)

    def stop(self) -> None:
        self._halt.set()


# ---------------------------------------------------------------------------
# Tube
# ---------------------------------------------------------------------------


def avi_files(raw: Path, max_files: int | None) -> list[Path]:
    files = natsorted(raw.glob("*.avi"), key=lambda p: p.name)
    return files[:max_files] if max_files else files


def count_frames(files: list[Path]) -> tuple[int, int, int]:
    n = 0
    for f in files:
        cap = cv2.VideoCapture(str(f))
        n += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    return n, h, w


def tube_spec(files: list[Path], traces_zarr: Path) -> dict[str, Any]:
    spec = yaml.safe_load(TUBE.read_text())
    spec["nodes"]["source"]["params"] = {"files": [str(f) for f in files]}
    spec["assets"]["traces"]["params"]["zarr_path"] = str(traces_zarr)
    return spec


def export_units(state: Any, max_proj: np.ndarray, out: Path) -> dict[str, Any]:
    """Footprints and full traces in the pixel layout of the preprocessed frames."""
    fps = state.assets["footprints"].obj.array
    trs = state.assets["traces"].obj.full_array()
    if fps is None or trs is None:
        np.savez(
            out, A=np.zeros((0, max_proj.size), np.float32), C=np.zeros((0, 0)), max_proj=max_proj
        )
        return {"units_final": 0}
    ids = [str(i) for i in fps["id_"].values]
    trs = trs.assign_coords(id_=trs["id_"].astype(str)).set_index(component="id_")
    C = trs.sel(component=ids).transpose("component", "frame").values.astype(np.float32)
    a = fps.data
    a = a.todense() if hasattr(a, "todense") else np.asarray(a)
    A = a.reshape(len(ids), -1).astype(np.float32)
    detected = fps["detected_on"].values.astype(np.int64)
    np.savez(out, A=A, C=C, max_proj=max_proj, detected_on=detected)
    return {"units_final": len(ids), "footprint_dims": list(fps.dims)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def patch_cala() -> None:
    """Fix a cala bug that crashes full-FOV runs within ~200 frames (main 50bc60d).

    ``Footprinter`` zeroes footprint pixels below ``ratio_lb`` but keeps them as stored
    entries of the sparse matrix. Two places then index ``A.data`` (all stored entries)
    with masks built from ``A.nonzero()`` (stored entries that are not zero), and the shapes
    disagree: ``residual._find_unlayered_footprints`` and ``footprints.update_shapes``
    (``Ab.data[indptr[m]:indptr[m+1]]`` against ``A_mask[m]``). Dropping the stored zeros
    on a copy first changes no value; the zeroed pixels are outside the footprint either way.
    """
    from cala.nodes.omf import footprints, residual  # noqa: PLC0415 (cala venv only)

    def no_stored_zeros(A: Any) -> Any:
        A = A.copy()
        A.eliminate_zeros()
        return A

    unlayered = residual._find_unlayered_footprints
    residual._find_unlayered_footprints = lambda A: unlayered(no_stored_zeros(A))
    update = footprints.update_shapes
    footprints.update_shapes = lambda Ab, **kw: update(Ab=no_stored_zeros(Ab), **kw)


def run(a: argparse.Namespace) -> None:
    from noob import SynchronousRunner, Tube  # noqa: PLC0415 (cala venv only)

    patch_cala()

    a.out.mkdir(parents=True, exist_ok=True)
    files = avi_files(a.raw, a.max_files)
    T, H, W = count_frames(files)
    traces_zarr = a.out / "traces.zarr"
    shutil.rmtree(traces_zarr, ignore_errors=True)
    (a.out / "tube.yaml").write_text(yaml.safe_dump(tube_spec(files, traces_zarr)))

    mon = Monitor(a.out, a.poll)
    mon.start()
    doc: dict[str, Any] = {
        "pipeline": "cala main 50bc60d, long_recording.yaml at full FOV and full frame rate",
        "raw": str(a.raw),
        "n_files": len(files),
        "frames": T,
        "height": H,
        "width": W,
        "status": "running",
        "blocks": [],
    }

    def dump(t0: float) -> None:
        doc.update(
            elapsed_seconds=time.perf_counter() - t0,
            peak_rss_bytes=mon.peak_rss,
            peak_disk_bytes=mon.peak_disk,
        )
        (a.out / "timing.json").write_text(json.dumps(doc, indent=1))

    t0 = time.perf_counter()
    tube = Tube.from_specification((a.out / "tube.yaml").resolve())
    runner = SynchronousRunner(tube=tube)
    max_proj = None
    tb, done = time.perf_counter(), 0
    try:
        for ret in runner.iter(n=T):
            frame = np.asarray(ret["prep"].array.values, np.float32)
            max_proj = frame if max_proj is None else np.maximum(max_proj, frame)
            done += 1
            if done % a.block == 0 or done == T:
                fps_obj = runner.tube.state.assets["footprints"].obj.array
                K = 0 if fps_obj is None else int(fps_obj.sizes["component"])
                dt = time.perf_counter() - tb
                doc["blocks"].append({"frames": done, "seconds": dt, "units": K})
                print(
                    f"{time.strftime('%H:%M:%S')} {done:>7d}/{T} frames  "
                    f"{a.block / dt if dt else 0:6.1f} fps  K={K}",
                    flush=True,
                )
                tb = time.perf_counter()
                dump(t0)
            if done == T:  # export before the iterator closes: noob's deinit drops the assets
                doc["loop_seconds"] = time.perf_counter() - t0
                te = time.perf_counter()
                doc.update(export_units(runner.tube.state, max_proj, a.out / "cala_units.npz"))
                doc["export_seconds"] = time.perf_counter() - te
        doc["status"] = "done"
    except BaseException as e:
        doc["status"] = f"failed at frame {done}: {type(e).__name__}: {e}"
        raise
    finally:
        mon.stop()
        dump(t0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-files", type=int, default=None, help="first N AVIs only (probe)")
    ap.add_argument("--block", type=int, default=1000, help="frames per timing block")
    ap.add_argument("--poll", type=float, default=10.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
