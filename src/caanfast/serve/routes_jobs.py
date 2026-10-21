"""Routes for starting and following runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from caanfast import io
from caanfast.serve.jobs import FINAL, Job, JobManager
from caanfast.serve.runs import fps_from_files

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


@router.get("/raw/info")
def raw_info(path: str) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_dir() or not any(raw.glob("*.avi")):
        raise HTTPException(404, f"no .avi files in {raw}")
    try:
        info = io.probe(raw)
    except OSError as e:
        raise HTTPException(400, f"cannot read the videos in {raw}: {e}") from None
    return {
        "path": str(raw.resolve()),
        "segments": len(info.segments),
        "frames": info.frames,
        "height": info.height,
        "width": info.width,
        "fps": fps_from_files(raw),
        "bytes": int(sum(p.stat().st_size for p in info.segments)),
    }


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
