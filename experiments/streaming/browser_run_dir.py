"""Lay a browser run's outputs out as a MiniMe run directory, so ``minime score`` scores it.

    python experiments/streaming/browser_run_dir.py BROWSER_DIR --raw RAW_DIR [--out DIR]
    minime score DIR --raw RAW_DIR

``BROWSER_DIR`` holds what the ``#/stream`` page offers after a run from the AVIs alone (or
``e2e/stream.spec.ts`` saved): the stitched footprints ``A_{data,indices,indptr}.npy``,
the streaming pass's traces ``C.npy`` ((K, T) float32), ``shifts.npy`` and ``session.json``.
Writes (default: ``BROWSER_DIR/run``) ``cnmf/A.npz``, ``cnmf/YrA.npy``,
``motion/arrays.npz`` (``shifts``), ``factors/meta.json`` (``H``, ``W``, ``T``) and
``params.json`` (``raw_dir`` and the session's parameters), the files
``score.score_pipeline_dir`` reads. The traces are the pass's, on the stitched footprints
(what ``t2_stream.py`` scores as pass 0), not the updated ``Amerged`` ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("browser", type=Path)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.browser / "run"
    session = json.loads((a.browser / "session.json").read_text(encoding="utf-8"))
    H, W = int(session["pass1"]["H"]), int(session["pass1"]["W"])
    C = np.load(a.browser / "C.npy", mmap_mode="r")
    K, T = C.shape
    A = sps.csr_matrix(
        (
            np.load(a.browser / "A_data.npy"),
            np.load(a.browser / "A_indices.npy"),
            np.load(a.browser / "A_indptr.npy"),
        ),
        shape=(K, H * W),
    )
    shifts = np.load(a.browser / "shifts.npy").astype(np.float32)
    assert shifts.shape == (T, 2), shifts.shape
    for d in ("cnmf", "motion", "factors"):
        (out / d).mkdir(parents=True, exist_ok=True)
    sps.save_npz(out / "cnmf" / "A.npz", A)
    np.save(out / "cnmf" / "YrA.npy", np.ascontiguousarray(C, dtype=np.float32))
    np.savez(out / "motion" / "arrays.npz", shifts=shifts)
    (out / "factors" / "meta.json").write_text(json.dumps({"H": H, "W": W, "T": T}, indent=1))
    params = {"raw_dir": str(a.raw.resolve()), "source_run": "browser", **session}
    (out / "params.json").write_text(json.dumps(params, indent=1))
    print(f"{K} units, {T} frames -> {out}")


if __name__ == "__main__":
    main()
