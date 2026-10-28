"""Routes for starting and following runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from minime import io
from minime.autotune.sweep import DERIVED, derived_params
from minime.params import Params
from minime.serve import preview
from minime.serve.arrays import array_response
from minime.serve.jobs import FINAL, Job, JobManager
from minime.serve.runs import fps_from_files

router = APIRouter()


def _jobs(request: Request) -> JobManager:
    jm = request.app.state.jobs
    if jm is None:
        raise HTTPException(503, "the job runner is not enabled")
    return jm


def _job(request: Request, jid: str) -> Job:
    try:
        return _jobs(request).get(jid)
    except KeyError:
        raise HTTPException(404, f"unknown job {jid}") from None


class JobBody(BaseModel):
    raw_dir: str
    name: str
    params: dict[str, Any] = {}
    keep_store: bool = False
    overwrite: bool = False


class PreviewBody(BaseModel):
    raw_dir: str
    t: int
    params: dict[str, Any] = {}


def _info(request: Request, path: str) -> tuple[Path, io.RawInfo]:
    """``io.probe`` of ``path``, cached per app: it decodes the whole last segment."""
    raw = Path(path).expanduser()
    if not raw.is_dir() or not any(raw.glob("*.avi")):
        raise HTTPException(404, f"no .avi files in {raw}")
    state = request.app.state
    key = str(raw.resolve())
    with state.raw_infos_lock:
        info = state.raw_infos.get(key)
    if info is None:
        try:
            info = io.probe(raw)
        except OSError as e:
            raise HTTPException(400, f"cannot read the videos in {raw}: {e}") from None
        with state.raw_infos_lock:
            state.raw_infos[key] = info
    return raw, info


def _preview_args(request: Request, body: PreviewBody) -> tuple[io.RawInfo, Params]:
    _, info = _info(request, body.raw_dir)
    if not 0 <= body.t < info.frames:
        raise HTTPException(400, f"frame {body.t} out of range 0..{info.frames - 1}")
    try:
        return info, Params.from_dict(body.params)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/raw/info")
def raw_info(request: Request, path: str) -> dict[str, Any]:
    raw, info = _info(request, path)
    return {
        "path": str(raw.resolve()),
        "segments": len(info.segments),
        "frames": info.frames,
        "height": info.height,
        "width": info.width,
        "fps": fps_from_files(raw),
        "bytes": int(sum(p.stat().st_size for p in info.segments)),
    }


@router.post("/raw/preview", response_model=None)
def raw_preview(request: Request, body: PreviewBody) -> Response:
    """Frame ``t`` through the pass-1 steps, (4, H, W) uint8 (``preview.PANELS``)."""
    info, p = _preview_args(request, body)
    return array_response(preview.frame_panels(info, body.t, p))


@router.post("/raw/motion")
def raw_motion(request: Request, body: PreviewBody) -> dict[str, Any]:
    """Shifts of a short clip around ``t``; the template comes from the clip."""
    info, p = _preview_args(request, body)
    try:
        return preview.motion_check(info, body.t, p)
    except OSError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/params/derived")
def params_derived(diameter: float) -> dict[str, Any]:
    """The size parameters for a cell ``diameter`` in px (``tophat_wnd`` and the ones
    auto-tune derives); the launcher fills them in from the diameter read off a frame."""
    try:
        p = derived_params(Params(), diameter)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"tophat_wnd": max(1, round(diameter)), **{k: getattr(p, k) for k in DERIVED}}


@router.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    return {"jobs": [asdict(j) for j in _jobs(request).list()]}


@router.post("/jobs")
def submit(request: Request, body: JobBody) -> dict[str, Any]:
    try:
        job = _jobs(request).submit(
            body.raw_dir, body.name, body.params, body.keep_store, body.overwrite
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return asdict(job)


@router.get("/jobs/{jid}")
def get_job(request: Request, jid: str) -> dict[str, Any]:
    return asdict(_job(request, jid))


@router.post("/jobs/{jid}/cancel")
def cancel(request: Request, jid: str) -> dict[str, Any]:
    _job(request, jid)
    return asdict(_jobs(request).cancel(jid))


@router.get("/jobs/{jid}/events")
async def events(request: Request, jid: str) -> StreamingResponse:
    jm = _jobs(request)
    _job(request, jid)

    async def gen():  # noqa: ANN202
        n_prog = n_log = 0
        last = None
        while True:
            job = jm.get(jid)
            prog = jm.progress(job)
            for e in prog[n_prog:]:
                yield f"event: stage\ndata: {json.dumps(e)}\n\n"
            n_prog = len(prog)
            lines = jm.log_lines(jid)
            for line in lines[n_log:]:
                yield f"event: log\ndata: {json.dumps(line)}\n\n"
            n_log = len(lines)
            if job.status != last:
                last = job.status
                yield f"event: status\ndata: {json.dumps(asdict(job))}\n\n"
            if job.status in FINAL or await request.is_disconnected():
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")
