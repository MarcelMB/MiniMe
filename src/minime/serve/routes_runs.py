"""Routes for folders, runs and their stage data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from minime import init
from minime.serve import derived
from minime.serve.arrays import array_response
from minime.serve.contours import unit_polygons
from minime.serve.frames import SOURCES, FrameSource, tile_map
from minime.serve.paramsmeta import GROUPS, defaults
from minime.serve.runs import RunView, find_runs, is_run
from minime.serve.traces import window, zooms
from minime.tiles import tile_grid

router = APIRouter()

IMAGES = ("max_proj", "background", "rank", "noise")


def _run(request: Request, rid: str) -> RunView:
    try:
        return request.app.state.registry.get(rid)
    except KeyError:
        raise HTTPException(404, f"unknown run {rid}") from None


def _frames(request: Request, run: RunView) -> FrameSource:
    """The run's FrameSource (one per run); those of runs no longer recent are dropped."""
    state = request.app.state
    with state.frames_lock:
        fr = state.frames
        if run.id not in fr or fr[run.id].run is not run:
            fr[run.id] = FrameSource(run)
        keep = set(state.registry.recent()) | {run.id}
        for rid in [r for r in fr if r not in keep]:
            del fr[rid]
        return fr[run.id]


def computing(request: Request, run: RunView, what: str, fn: Callable[[], None]) -> JSONResponse:
    """202 while ``fn`` runs in the background (500 if it failed).

    The key carries the run's version, so a rewritten run never sees an old failure.
    """
    key = f"{run.id}:{run.stamp}:{what}"
    st = request.app.state.background.ensure(key, fn)
    return JSONResponse(st, status_code=500 if st["status"] == "failed" else 202)


@router.get("/fs")
def fs(path: str | None = None) -> dict[str, Any]:
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        raise HTTPException(404, f"not a folder: {p}")
    p = p.resolve()
    entries = []
    try:
        children = sorted(c for c in p.iterdir() if c.is_dir() and not c.name.startswith("."))
    except OSError as e:
        raise HTTPException(403, f"cannot list {p}: {e}") from None
    for c in children:
        try:
            has_avi = any(c.glob("*.avi"))
        except OSError:
            continue
        entries.append({"name": c.name, "path": str(c), "has_avi": has_avi, "is_run": is_run(c)})
    return {"path": str(p), "parent": str(p.parent), "entries": entries}


@router.get("/runs")
def runs(request: Request, root: str | None = None) -> dict[str, Any]:
    base = Path(root).expanduser() if root else Path.home()
    if not base.is_dir():
        raise HTTPException(404, f"not a folder: {base}")
    reg = request.app.state.registry
    out = []
    for p in find_runs(base):
        try:
            out.append(reg.open(p, touch=False).listing())
        except (OSError, ValueError, KeyError) as e:
            out.append({"path": str(p), "name": p.name, "error": f"{type(e).__name__}: {e}"})
    return {"root": str(base.resolve()), "runs": out}


@router.get("/runs/{rid}/summary")
def summary(request: Request, rid: str) -> dict[str, Any]:
    return _run(request, rid).summary()


class FpsBody(BaseModel):
    fps: float


@router.post("/runs/{rid}/fps")
def set_fps(request: Request, rid: str, body: FpsBody) -> dict[str, Any]:
    run = _run(request, rid)
    try:
        run.set_fps(body.fps)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"fps": run.fps}


@router.get("/runs/{rid}/image/{name}")
def image(request: Request, rid: str, name: str) -> Response:
    run = _run(request, rid)
    if name not in IMAGES:
        raise HTTPException(404, f"unknown image {name!r} (one of {', '.join(IMAGES)})")
    try:
        if name == "max_proj":
            a = np.load(run.path / "factors" / "max_proj.npy")
        elif name == "background":
            a = np.load(run.path / "cnmf" / "b.npy")
        elif name == "rank":
            a = tile_map(tile_grid(run.H, run.W, run.tile), run.H, run.W, run.ranks)
        else:
            a = run.factors.sigma_flat().reshape(run.H, run.W)
    except FileNotFoundError as e:
        raise HTTPException(404, f"{name} is not available for this run: {e.filename}") from None
    return array_response(np.asarray(a, np.float32))


@router.get("/runs/{rid}/frame/{t}", response_model=None)
def frame(request: Request, rid: str, t: int, source: str = "recon") -> Response:
    run = _run(request, rid)
    if source not in SOURCES:
        raise HTTPException(400, f"unknown source {source!r} (one of {', '.join(SOURCES)})")
    fs_ = _frames(request, run)
    if not 0 <= t < run.T:
        raise HTTPException(400, f"frame {t} out of range 0..{run.T - 1}")
    if not fs_.available(source):
        try:
            fs_.frame(source, t)
        except LookupError as e:
            raise HTTPException(404, str(e)) from None
    if not fs_.has_contrast(source):
        return computing(request, run, f"contrast:{source}", lambda: fs_.contrast(source))
    return array_response(fs_.frame(source, t))


@router.get("/params/defaults")
def params_defaults() -> dict[str, Any]:
    return {"groups": list(GROUPS), "params": defaults()}


@router.get("/runs/{rid}/motion")
def motion_(request: Request, rid: str) -> Response:
    sc = _run(request, rid).sidecar
    q = np.asarray(sc.quality, np.float32).reshape(-1, 1)
    return array_response(np.hstack([np.asarray(sc.shifts, np.float32), q]))


@router.get("/runs/{rid}/motion/residual", response_model=None)
def motion_residual(request: Request, rid: str) -> Response:
    run = _run(request, rid)
    r = derived.residual(run)
    if r is None:
        return computing(request, run, "residual", lambda: derived.compute_residual(run))
    return array_response(r)


@router.get("/runs/{rid}/seeds", response_model=None)
def seeds_(request: Request, rid: str) -> Response | dict[str, Any]:
    run = _run(request, rid)
    s = derived.seeds(run)
    if s is None:
        return computing(request, run, "seeds", lambda: derived.compute_seeds_and_init(run))
    return {k: np.round(v, 1).tolist() for k, v in s.items()}


@router.get("/runs/{rid}/units", response_model=None)
def units(request: Request, rid: str, stage: str = "final") -> Response | dict[str, Any]:
    run = _run(request, rid)
    if stage == "final":
        A = run.A
        peak = pd.read_csv(run.path / "cnmf" / "units.csv")["peak"].to_numpy()
    elif stage == "init":
        A = derived.init_A(run)
        if A is None:
            return computing(request, run, "seeds", lambda: derived.compute_seeds_and_init(run))
        peak = None
    else:
        raise HTTPException(400, f"stage must be final or init, got {stage!r}")
    cent = init.centroids(A, run.H, run.W) if A.shape[0] else np.zeros((0, 2))
    polys = unit_polygons(A, run.H, run.W)
    sizes = np.diff(A.indptr)
    return {
        "stage": stage,
        "units": [
            {
                "id": k,
                "centroid": [round(float(cent[k, 0]), 2), round(float(cent[k, 1]), 2)],
                "size_px": int(sizes[k]),
                "peak": None if peak is None else float(peak[k]),
                "polygons": polys[k],
            }
            for k in range(A.shape[0])
        ],
    }


def _kind(kind: str) -> str:
    if kind not in ("C", "YrA"):
        raise HTTPException(400, f"kind must be C or YrA, got {kind!r}")
    return kind


@router.get("/runs/{rid}/traces/overview", response_model=None)
def traces_overview(request: Request, rid: str, kind: str = "C", bins: int = 2000) -> Response:
    run = _run(request, rid)
    kind = _kind(kind)
    if not 1 <= bins <= 20000:
        raise HTTPException(400, "bins must be in 1..20000")
    ov = derived.traces_overview(run, kind, bins)
    if ov is None:
        key = f"overview:{kind}:{bins}"
        return computing(request, run, key, lambda: derived.compute_overview(run, kind, bins))
    return array_response(ov)


@router.get("/runs/{rid}/traces/{unit}")
def trace(
    request: Request,
    rid: str,
    unit: int,
    start: int = 0,
    stop: int | None = None,
    kind: str = "C",
    lowpass_hz: float | None = None,
) -> Response:
    run = _run(request, rid)
    C = run.traces(_kind(kind))
    if not 0 <= unit < C.shape[0]:
        raise HTTPException(400, f"unit {unit} out of range 0..{C.shape[0] - 1}")
    stop = run.T if stop is None else stop
    try:
        y = window(C[unit], start, stop, fs=run.fps, lowpass_hz=lowpass_hz)
    except (IndexError, ValueError) as e:
        raise HTTPException(400, str(e)) from None
    return array_response(y)


@router.get("/runs/{rid}/zooms", response_model=None)
def zooms_(request: Request, rid: str) -> Response | dict[str, Any]:
    run = _run(request, rid)
    ov = derived.traces_overview(run, "C", 2000)
    if ov is None:
        key = "overview:C:2000"
        return computing(request, run, key, lambda: derived.compute_overview(run, "C", 2000))
    return zooms(ov[:, 1].max(axis=1), run.T, run.fps)
