"""Rerun window 2 (seeds, init, CNMF) from a run's saved factors with a background variant.

    .venv/bin/python bench/background_variants.py run SRC_RUN OUT_DIR [--threads N] [--no-clip]
    .venv/bin/python bench/background_variants.py compare BASE_DIR OTHER_DIR

``run`` writes ``OUT_DIR/{cnmf/, params.json, report.json}`` and symlinks ``factors/`` and
``motion/`` from ``SRC_RUN`` so that ``caanfast score OUT_DIR`` and ``bench/compare_minian.py``
work on it. ``report.json`` holds the ``cnmf`` stage seconds and every ``update_background``
call's seconds. ``compare`` prints (and writes ``OTHER_DIR/compare_bg.json``) the max relative
difference of ``b`` and ``f`` and the per-unit trace correlation of ``C`` and ``YrA`` (units
matched by footprint centroid, 5 px) of ``OTHER_DIR`` against ``BASE_DIR``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: bench/ is not packaged
from bench.score import match_units  # noqa: E402
from caanfast import cnmf, corr, init, pipeline  # noqa: E402
from caanfast.factors import Factors  # noqa: E402
from caanfast.params import RUN_METADATA_KEYS, Params  # noqa: E402
from caanfast.report import StageTimer  # noqa: E402

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _timed_background(calls: list[float], threads: int | None) -> None:
    """Patch ``cnmf.update_background`` to record each call's wall seconds."""
    real = cnmf.update_background
    kw = {"threads": threads} if "threads" in inspect.signature(real).parameters else {}
    if threads is not None and not kw:
        raise SystemExit("this cnmf.update_background has no threads argument")

    @functools.wraps(real)
    def wrapped(*a: Any, **k: Any) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        out = real(*a, **{**kw, **k})
        calls.append(time.perf_counter() - t0)
        return out

    cnmf.update_background = wrapped


def _link(src: Path, dst: Path) -> None:
    if not dst.exists():
        dst.symlink_to(src.resolve(), target_is_directory=True)


def run(src: Path, out: Path, threads: int | None, clip: bool) -> dict[str, Any]:
    doc = json.loads((src / "params.json").read_text())
    params = Params.from_json(src / "params.json")
    if not clip:
        params.background_clip = False
    t_start = time.perf_counter()
    fac = Factors.load(src / "factors")
    timer = StageTimer()
    with timer("corr_setup"):
        fc = corr.FactorCorr(fac, params.noise_freq)
    s = pipeline._seeds(fac, fc, params, timer)
    A, C = pipeline._init_units(fac, fc, s, params, timer)
    calls: list[float] = []
    _timed_background(calls, threads)
    with timer("cnmf"):
        res = cnmf.run_cnmf(fac, A, C, params, fc.traces_corr)
    out.mkdir(parents=True, exist_ok=True)
    pipeline._write_cnmf(out / "cnmf", res, fac.H, fac.W_)
    _link(src / "factors", out / "factors")
    _link(src / "motion", out / "motion")
    meta = {k: doc[k] for k in RUN_METADATA_KEYS if k in doc}
    (out / "params.json").write_text(json.dumps({**params.asdict(), **meta}, indent=1))
    timer.counts.update(res.counts)
    timer.counts["units_final"] = res.A.shape[0]
    extra = {
        "background_calls_s": calls,
        "threads": threads,
        "src": str(src),
        "wall_total": time.perf_counter() - t_start,  # window 2 only (read by compare_minian)
    }
    timer.to_json(out / "report.json", extra)
    cnmf_s = next(r["seconds"] for r in timer.rows if r["stage"] == "cnmf")
    return {"cnmf_s": round(cnmf_s, 2), "background_s": [round(c, 2) for c in calls]}


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _load(d: Path) -> dict[str, Any]:
    c = d / "cnmf"
    b = np.load(c / "b.npy")
    return {
        "A": sps.load_npz(c / "A.npz").tocsr(),
        "C": np.load(c / "C.npy"),
        "YrA": np.load(c / "YrA.npy"),
        "b": b.ravel(),
        "f": np.load(c / "f.npy"),
        "H": b.shape[0],
        "W": b.shape[1],
    }


def _rowcorr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(1) / den


def _rel(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.abs(x.astype(np.float64) - ref).max() / np.abs(ref).max())


def compare(base: Path, other: Path) -> dict[str, Any]:
    p, q = _load(base), _load(other)
    cp = init.centroids(p["A"], p["H"], p["W"])
    cq = init.centroids(q["A"], q["H"], q["W"])
    ui, ti = match_units(cq, cp)
    out: dict[str, Any] = {
        "units_base": p["A"].shape[0],
        "units": q["A"].shape[0],
        "matched": len(ui),
        "b_max_rel": _rel(q["b"], p["b"]),
        "f_max_rel": _rel(q["f"], p["f"]),
        "f_corr": float(np.corrcoef(q["f"], p["f"])[0, 1]),
    }
    for key in ("C", "YrA"):
        r = _rowcorr(q[key][ui].astype(np.float64), p[key][ti].astype(np.float64))
        out[f"{key}_corr_median"] = float(np.nanmedian(r))
        out[f"{key}_corr_min"] = float(np.nanmin(r))
        out[f"{key}_corr_frac_below_0.99"] = float(np.mean(r < 0.99))
    (other / "compare_bg.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("src", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--no-clip", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("other", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        res = run(a.src.expanduser(), a.out.expanduser(), a.threads, not a.no_clip)
    else:
        res = compare(a.base.expanduser(), a.other.expanduser())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
