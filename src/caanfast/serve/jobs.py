"""Run queue of ``caanfast serve``: one ``caanfast run`` subprocess at a time."""

from __future__ import annotations

import atexit
import contextlib
import fcntl
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
from typing import IO, Any

from caanfast import pipeline
from caanfast.params import Params
from caanfast.serve.cache import cache_root
from caanfast.serve.runs import is_run

FINAL = {"done", "failed", "cancelled", "interrupted"}
STOPPED = "the server stopped during the run"


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
    overwrite: bool = False
    pid: int | None = None


def default_command(job: Job, params_file: Path) -> list[str]:
    cmd = [sys.executable, "-m", "caanfast", "run", job.raw_dir]
    cmd += ["--params", str(params_file), "--name", job.name]
    return cmd + (["--keep-store"] if job.keep_store else [])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(pid: int, timeout: float = 10.0) -> None:
    """SIGTERM the process group led by ``pid``, SIGKILL it if it outlives ``timeout``."""
    try:
        if os.getpgid(pid) != pid:
            return  # not a job's process group leader (the pid was reused)
    except (ProcessLookupError, PermissionError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, sig)
        t0 = time.time()
        while _alive(pid) and time.time() - t0 < timeout:
            time.sleep(0.05)
        if not _alive(pid):
            return


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
        self._lock_file = self._take_lock()
        self.command = command or default_command
        self._cv = threading.Condition()
        self._proc: subprocess.Popen | None = None
        self._current: Job | None = None
        self._cancel: set[str] = set()
        self._stopping = False
        self.jobs: dict[str, Job] = {}
        for job in self._load():
            if job.status == "running":
                if job.pid is not None and _alive(job.pid):
                    _stop_group(job.pid)  # left over from a server that was killed
                job.status, job.error = "interrupted", STOPPED
                job.ended = job.ended or time.time()
            self.jobs[job.id] = job
        self._save()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def _take_lock(self) -> IO[str]:
        f = (self.root / "jobs.lock").open("w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            raise RuntimeError(f"caanfast serve is already running (jobs in {self.root})") from None
        return f

    def _load(self) -> list[Job]:
        """Jobs of ``jobs.json``; a corrupt file is renamed aside, bad entries skipped."""
        f = self.root / "jobs.json"
        if not f.exists():
            return []
        try:
            docs = json.loads(f.read_text())
            if not isinstance(docs, list):
                raise ValueError("jobs.json is not a list")
        except (OSError, ValueError):
            with contextlib.suppress(OSError):
                f.rename(self.root / f"jobs.json.bad-{time.strftime('%Y%m%d-%H%M%S')}")
            return []
        jobs = []
        for d in docs:
            try:
                jobs.append(Job(**d))
            except TypeError:
                continue
        return jobs

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
        job = Job(jid, str(raw), name, params, keep_store, out_dir=str(out), overwrite=overwrite)
        with self._cv:
            if any(j.out_dir == str(out) and j.status not in FINAL for j in self.jobs.values()):
                raise ValueError(f"a queued or running job already writes to {out}")
            self.jobs[jid] = job
            self._save()
            self._cv.notify_all()
        return job

    def get(self, jid: str) -> Job:
        with self._cv:
            return self.jobs[jid]

    def list(self) -> list[Job]:
        with self._cv:
            jobs = list(self.jobs.values())
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def shutdown(self) -> None:
        """Stop the running job (it becomes ``interrupted``) and pick no further jobs."""
        with self._cv:
            if self._stopping:
                return
            self._stopping = True
            self._terminate()
            self._cv.notify_all()
        self._thread.join(timeout=10)
        with self._cv:
            if self._proc is not None:  # did not stop on SIGTERM
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._proc.pid, signal.SIGKILL)
        self._thread.join(timeout=5)
        with self._cv:
            job = self._current
            if job is not None and job.status == "running":
                job.status, job.error, job.ended = "interrupted", STOPPED, time.time()
                self._save()
            self._cv.notify_all()
        self._lock_file.close()

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

    def _next(self) -> Job | None:
        """The oldest queued job, now ``running``; None once the manager is stopping."""
        with self._cv:
            while not self._stopping:
                queued = [j for j in self.jobs.values() if j.status == "queued"]
                if queued:
                    job = min(queued, key=lambda j: j.created)
                    job.status, job.started = "running", time.time()
                    self._current = job
                    self._save()
                    return job
                self._cv.wait()
            return None

    def _fail(self, job: Job, error: str) -> None:
        with self._cv:
            self._proc = self._current = None
            job.status, job.ended, job.error = "failed", time.time(), error
            self._save()
            self._cv.notify_all()

    def _worker(self) -> None:
        while True:
            job = self._next()
            if job is None:
                return
            pfile = self.dir / f"{job.id}.params.json"
            out = Path(job.out_dir)
            if is_run(out) and not job.overwrite:
                self._fail(job, f"{out} already holds a run; start again with overwrite")
                continue
            try:
                with self.log_path(job.id).open("wb") as log:
                    proc = subprocess.Popen(
                        self.command(job, pfile),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    with self._cv:
                        self._proc = proc
                        job.pid = proc.pid
                        self._save()
                        if job.id in self._cancel or self._stopping:  # between "running" and Popen
                            self._terminate()
                    rc = proc.wait()
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                self._fail(job, f"could not start the run: {e}")
                continue
            with self._cv:
                self._proc = self._current = None
                job.ended = time.time()
                if self._stopping:
                    job.status, job.error = "interrupted", STOPPED
                elif job.id in self._cancel:
                    job.status = "cancelled"
                elif rc == 0:
                    job.status = "done"
                else:
                    job.status = "failed"
                    job.error = "\n".join(self.log_lines(job.id)[-20:])
                self._save()
                self._cv.notify_all()
