"""Rerun window 2 (seeds, init, CNMF) from a run's saved factors with changed parameters.

    .venv/bin/python bench/window2_rerun.py SRC_RUN OUT_DIR [--set sparse_snr=0.8 ...]

``--set`` takes ``key=value`` pairs (value parsed as JSON, else kept as a string) applied on
top of ``SRC_RUN/params.json``. Writes ``OUT_DIR/{cnmf/, params.json, report.json}`` and
symlinks ``factors/`` and ``motion/`` from ``SRC_RUN``, so ``cafast score OUT_DIR``,
``bench/compare_minian.py`` and ``bench/compare_iters.py`` work on it.

``--set noise_frames=N`` only changes the per-pixel noise ``sigma`` of the factors (``W``,
``V`` and the rank do not depend on it), so it is applied by recomputing ``sigma`` of every
tile from ``SRC_RUN/tiles`` (the kept tile store) instead of redoing pass 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: bench/ is not packaged
from cafast import compress, corr, pipeline  # noqa: E402
from cafast.cnmf import run_cnmf  # noqa: E402
from cafast.factors import Factors  # noqa: E402
from cafast.params import RUN_METADATA_KEYS, Params  # noqa: E402
from cafast.report import StageTimer  # noqa: E402
from cafast.tiles import TileStore  # noqa: E402


def parse_sets(items: list[str]) -> dict[str, Any]:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def resigma(fac: Factors, store_dir: Path, noise_freq: float, frames: int) -> None:
    """Replace every tile's ``sigma`` by the estimate from ``frames`` frames of the store."""
    store = TileStore(store_dir, fac.grid, len(list(store_dir.glob("tile_000.part*"))))
    for b, tf in enumerate(fac.tiles):
        Y = compress.TileMatrix(store.read_tile(b))
        tf.sigma = Y.noise(noise_freq, frames).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--set", dest="sets", nargs="*", default=[])
    a = ap.parse_args()

    raw = json.loads((a.src / "params.json").read_text())
    meta = {k: raw.pop(k) for k in list(raw) if k in RUN_METADATA_KEYS}
    raw.update(parse_sets(a.sets))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "params.json").write_text(json.dumps({**raw, **meta}, indent=1))
    params = Params.from_json(a.out / "params.json")
    for d in ("factors", "motion"):
        link = a.out / d
        if not link.exists() and (a.src / d).exists():
            link.symlink_to((a.src / d).resolve())

    timer = StageTimer()
    with timer("factors_load"):
        fac = Factors.load(a.src / "factors")
    if params.noise_frames:
        with timer("resigma"):
            resigma(fac, a.src / "tiles", params.noise_freq, params.noise_frames)
    with timer("corr_setup"):
        fc = corr.FactorCorr(fac, params.noise_freq)
    s = pipeline._seeds(fac, fc, params, timer)
    A, C = pipeline._init_units(fac, fc, s, params, timer)
    with timer("cnmf"):
        res = run_cnmf(fac, A, C, params, fc.traces_corr)
    timer.counts.update(res.counts)
    timer.counts["units_final"] = res.A.shape[0]
    with timer("write_outputs"):
        pipeline._write_cnmf(a.out / "cnmf", res, fac.H, fac.W_)
    wall = sum(r["seconds"] for r in timer.rows)  # window 2 only (read by compare_minian)
    timer.to_json(a.out / "report.json", {"src": str(a.src), "set": a.sets, "wall_total": wall})
    print(json.dumps(timer.counts, indent=1, default=int))
    for r in timer.rows:
        print(f"{r['stage']:<16s} {r['seconds']:8.1f} s")


if __name__ == "__main__":
    main()
