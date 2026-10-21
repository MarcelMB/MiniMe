"""Run queue of ``caanfast serve``: one ``caanfast run`` subprocess at a time."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from caanfast import pipeline
from caanfast.params import Params
from caanfast.serve.cache import cache_root
from caanfast.serve.runs import is_run

FINAL = {"done", "failed", "cancelled", "interrupted"}


@dataclass
class Job:
    id: str
    raw_dir: str
    name: str
    params: dict[str, Any]
    keep_store: bool
    status: str = "queued"  # queued | running | done | failed | cancelled | interrupted
    out_dir: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    error: str = ""


def default_command(job: Job, params_file: Path) -> list[str]:
    cmd = [sys.executable, "-m", "caanfast", "run", job.raw_dir]
    cmd += ["--params", str(params_file), "--name", job.name]
    return cmd + (["--keep-store"] if job.keep_store else [])


class JobManager:
    """Queue persisted in ``<root>/jobs.json``; logs and params files in ``<root>/jobs/``."""

    def __init__(
        self,
        root: Path | None = None,
        command: Callable[[Job, Path], list[str]] | None = None,
    ) -> None:
        self.root = Path(root or cache_root())
        self.dir = self.root / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.command = command or default_command
        self._cv = threading.Condition()
        self._proc: subprocess.Popen | None = None
        self._cancel: set[str] = set()
        self.jobs: dict[str, Job] = {}
        f = self.root / "jobs.json"
        if f.exists():
            for d in json.loads(f.read_text()):
                job = Job(**d)
                if job.status == "running":
                    job.status, job.error = "interrupted", "the server stopped during the run"
                self.jobs[job.id] = job
        self._save()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- public ---------------------------------------------------------------

    def submit(
        self,
        raw_dir: str,
        name: str,
        params: dict[str, Any],
        keep_store: bool = False,
        overwrite: bool = False,
    ) -> Job:
        raw = Path(raw_dir).expanduser()
        if not raw.is_dir() or not any(raw.glob("*.avi")):
            raise ValueError(f"no .avi files in {raw}")
        pipeline.check_run_name(name)
        jid = uuid.uuid4().hex[:10]
        pfile = self.dir / f"{jid}.params.json"
        pfile.write_text(json.dumps(params))
        Params.from_json(pfile)  # raises ValueError on unknown or invalid parameters
        out = pipeline.out_dir_for(raw, name)
        if is_run(out) and not overwrite:
            raise ValueError(f"{out} already holds a run; choose another name or overwrite")
        job = Job(jid, str(raw), name, params, keep_store, out_dir=str(out))
        with self._cv:
            self.jobs[jid] = job
            self._save()
            self._cv.notify_all()
        return job

    def get(self, jid: str) -> Job:
        return self.jobs[jid]

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, jid: str) -> Job:
        with self._cv:
            job = self.jobs[jid]
            if job.status == "queued":
                job.status, job.ended = "cancelled", time.time()
                self._save()
            elif job.status == "running":
                self._cancel.add(jid)  # the worker also checks this right after Popen
                self._terminate()
            self._cv.notify_all()
            return job

    def _terminate(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGTERM)

    def wait(self, jid: str, timeout: float) -> Job:
        t0 = time.time()
        with self._cv:
            while self.jobs[jid].status not in FINAL:
                left = timeout - (time.time() - t0)
                if left <= 0:
                    raise TimeoutError(f"job {jid} still {self.jobs[jid].status}")
                self._cv.wait(min(left, 0.5))
            return self.jobs[jid]

    def log_path(self, jid: str) -> Path:
        return self.dir / f"{jid}.log"

    def log_lines(self, jid: str) -> list[str]:
        p = self.log_path(jid)
        return p.read_text(errors="replace").splitlines() if p.exists() else []

    def progress(self, job: Job) -> list[dict[str, Any]]:
        """Stage events of this job's run (older lines of a previous run are skipped)."""
        p = Path(job.out_dir) / "progress.jsonl"
        if job.started is None or not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written
            if e.get("t", 0) >= job.started - 1:
                out.append(e)
        return out

    # -- worker -----------------------------------------------------------------

    def _save(self) -> None:
        tmp = self.root / "jobs.json.tmp"
        tmp.write_text(json.dumps([asdict(j) for j in self.jobs.values()], indent=1))
        tmp.replace(self.root / "jobs.json")

    def _next(self) -> Job:
        with self._cv:
            while True:
                queued = [j for j in self.jobs.values() if j.status == "queued"]
                if queued:
                    job = min(queued, key=lambda j: j.created)
                    job.status, job.started = "running", time.time()
                    self._save()
                    return job
                self._cv.wait()

    def _worker(self) -> None:
        while True:
            job = self._next()
            pfile = self.dir / f"{job.id}.params.json"
            with self.log_path(job.id).open("wb") as log:
                proc = subprocess.Popen(
                    self.command(job, pfile),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with self._cv:
                    self._proc = proc
                    if job.id in self._cancel:  # cancelled between "running" and Popen
                        self._terminate()
                rc = proc.wait()
            with self._cv:
                self._proc = None
                job.ended = time.time()
                if job.id in self._cancel:
                    job.status = "cancelled"
                elif rc == 0:
                    job.status = "done"
                else:
                    job.status = "failed"
                    job.error = "\n".join(self.log_lines(job.id)[-20:])
                self._save()
                self._cv.notify_all()
