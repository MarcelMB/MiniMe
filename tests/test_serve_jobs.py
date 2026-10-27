import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minime.serve.app import create_app
from minime.serve.jobs import JobManager
from minime.testing import synth
from tests.conftest import TINY_PARAMS


def _sleeper(job, params_file):
    return [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "my session" / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return raw


def test_submit_rejects_bad_input(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    with pytest.raises(ValueError, match="no .avi"):
        jm.submit(str(tmp_path), "x", {})
    with pytest.raises(ValueError, match="unknown parameter"):
        jm.submit(str(raw), "x", {"bogus": 1})
    with pytest.raises(ValueError):
        jm.submit(str(raw), "../escape", {})


def test_a_real_run_end_to_end_and_overwrite_guard(tmp_path, raw):
    jm = JobManager(tmp_path / "cache")
    job = jm.submit(str(raw), "t", {**TINY_PARAMS, "size_thres": [9, None]})
    done = jm.wait(job.id, timeout=300)
    assert done.status == "done", jm.log_lines(job.id)[-20:]
    ends = [e["stage"] for e in jm.progress(done) if e["event"] == "end"]
    assert "cnmf" in ends
    assert (raw.parent / "processed" / "t" / "report.json").is_file()
    with pytest.raises(ValueError, match="already holds a run"):
        jm.submit(str(raw), "t", {})
    again = jm.submit(str(raw), "t", {**TINY_PARAMS, "size_thres": [9, None]}, overwrite=True)
    assert jm.wait(again.id, timeout=300).status == "done"


def test_cancel_and_queue(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    a = jm.submit(str(raw), "a", {})
    b = jm.submit(str(raw), "b", {})
    t0 = time.time()
    while jm.get(a.id).status != "running":
        assert time.time() - t0 < 10
        time.sleep(0.05)
    assert jm.get(b.id).status == "queued"
    assert jm.cancel(b.id).status == "cancelled"
    jm.cancel(a.id)
    assert jm.wait(a.id, timeout=10).status == "cancelled"


def test_running_jobs_are_interrupted_after_a_restart(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "j1",
                    "raw_dir": "/r",
                    "name": "n",
                    "params": {},
                    "keep_store": False,
                    "status": "running",
                    "out_dir": "/o",
                    "created": 0.0,
                    "started": 1.0,
                    "ended": None,
                    "error": "",
                }
            ]
        )
    )
    assert JobManager(root, command=_sleeper).get("j1").status == "interrupted"


def test_job_routes_and_events(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    c = TestClient(create_app("tok", cache_root=tmp_path / "cache", jobs=jm))
    c.headers["X-Token"] = "tok"
    r = c.post("/api/jobs", json={"raw_dir": str(raw), "name": "a", "params": {}})
    assert r.status_code == 200
    jid = r.json()["id"]
    assert (
        c.post("/api/jobs", json={"raw_dir": "/nope", "name": "a", "params": {}}).status_code == 400
    )
    assert [j["id"] for j in c.get("/api/jobs").json()["jobs"]] == [jid]
    assert c.post(f"/api/jobs/{jid}/cancel").json()["status"] in ("cancelled", "running")
    jm.wait(jid, timeout=10)
    with c.stream("GET", f"/api/jobs/{jid}/events") as s:
        text = "".join(s.iter_text())
    assert "event: status" in text and '"cancelled"' in text
    assert c.get("/api/jobs/nope").status_code == 404


def test_raw_info(tmp_path, raw):
    c = TestClient(create_app("tok", cache_root=tmp_path / "cache"))
    c.headers["X-Token"] = "tok"
    info = c.get("/api/raw/info", params={"path": str(raw)}).json()
    assert (info["segments"], info["frames"], info["height"], info["width"]) == (3, 300, 64, 64)
    assert info["fps"] == 30.0 and info["bytes"] > 0
    assert c.get("/api/raw/info", params={"path": str(tmp_path)}).status_code == 404


def test_a_run_that_cannot_start_fails_without_killing_the_worker(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=lambda j, p: ["/nonexistent/python"])
    a = jm.submit(str(raw), "a", {})
    done = jm.wait(a.id, timeout=10)
    assert done.status == "failed" and "could not start the run" in done.error
    b = jm.submit(str(raw), "b", {})
    assert jm.wait(b.id, timeout=10).status == "failed"


def _wait_status(jm, jid, status, timeout=10):
    t0 = time.time()
    while jm.get(jid).status != status:
        assert time.time() - t0 < timeout, jm.get(jid)
        time.sleep(0.05)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_shutdown_stops_the_running_job(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    a = jm.submit(str(raw), "a", {})
    b = jm.submit(str(raw), "b", {})
    t0 = time.time()
    while jm.get(a.id).pid is None:
        assert time.time() - t0 < 10
        time.sleep(0.05)
    pid = jm.get(a.id).pid
    jm.shutdown()
    assert not _alive(pid)
    done = jm.get(a.id)
    assert done.status == "interrupted" and "server stopped" in done.error
    time.sleep(0.3)
    assert jm.get(b.id).status == "queued"  # the worker picks nothing after shutdown
    saved = {d["id"]: d for d in json.loads((tmp_path / "cache" / "jobs.json").read_text())}
    assert saved[a.id]["status"] == "interrupted"


def test_app_lifespan_shuts_the_job_runner_down(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    with TestClient(create_app("tok", cache_root=tmp_path / "cache", jobs=jm)) as c:
        c.headers["X-Token"] = "tok"
        jid = c.post("/api/jobs", json={"raw_dir": str(raw), "name": "a"}).json()["id"]
        _wait_status(jm, jid, "running")
    assert jm.get(jid).status == "interrupted"


def test_a_restart_terminates_a_job_process_that_is_still_alive(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    threading.Thread(target=proc.wait, daemon=True).start()  # reap it once it ends
    root = tmp_path / "cache"
    root.mkdir()
    job = {"id": "j1", "raw_dir": "/r", "name": "n", "params": {}, "keep_store": False}
    (root / "jobs.json").write_text(json.dumps([{**job, "status": "running", "pid": proc.pid}]))
    try:
        jm = JobManager(root, command=_sleeper)
        assert jm.get("j1").status == "interrupted"
        t0 = time.time()
        while proc.poll() is None:
            assert time.time() - t0 < 5, "the old job process is still running"
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_one_out_dir_is_written_by_one_job_at_a_time(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    c = TestClient(create_app("tok", cache_root=tmp_path / "cache", jobs=jm))
    c.headers["X-Token"] = "tok"
    a = jm.submit(str(raw), "a", {})
    b = jm.submit(str(raw), "b", {})
    _wait_status(jm, a.id, "running")
    for name in ("a", "b"):  # running, queued
        with pytest.raises(ValueError, match="queued or running job already writes"):
            jm.submit(str(raw), name, {}, overwrite=True)
        r = c.post("/api/jobs", json={"raw_dir": str(raw), "name": name, "overwrite": True})
        assert r.status_code == 400
    jm.cancel(b.id)
    jm.cancel(a.id)
    jm.wait(a.id, timeout=10)
    jm.shutdown()


def test_a_folder_that_became_a_run_after_submit_fails_at_start(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    a = jm.submit(str(raw), "a", {})
    b = jm.submit(str(raw), "b", {})
    _wait_status(jm, a.id, "running")
    out = Path(b.out_dir)
    out.mkdir(parents=True)
    for m in ("params.json", "report.json"):
        (out / m).write_text("{}")
    jm.cancel(a.id)
    done = jm.wait(b.id, timeout=10)
    assert done.status == "failed" and "already holds a run" in done.error
    assert done.pid is None and not jm.log_path(b.id).exists()
    assert sorted(p.name for p in out.iterdir()) == ["params.json", "report.json"]
    jm.shutdown()


def test_a_second_job_runner_on_the_same_root_is_refused(tmp_path):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    with pytest.raises(RuntimeError, match="already running"):
        JobManager(tmp_path / "cache", command=_sleeper)
    jm.shutdown()
    JobManager(tmp_path / "cache", command=_sleeper).shutdown()  # free again after shutdown


def test_a_corrupt_jobs_file_is_kept_aside_and_bad_entries_skipped(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "jobs.json").write_text("[{not json")
    jm = JobManager(root, command=_sleeper)
    assert jm.list() == []
    assert len(list(root.glob("jobs.json.bad-*"))) == 1
    jm.shutdown()
    good = {"id": "j1", "raw_dir": "/r", "name": "n", "params": {}, "keep_store": False}
    (root / "jobs.json").write_text(json.dumps([good, {"id": "j2", "bogus": 1}, 7]))
    jm = JobManager(root, command=_sleeper)
    assert [j.id for j in jm.list()] == ["j1"]
    jm.shutdown()


def test_list_and_get_survive_concurrent_submits(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                jm.list()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(20):
        jm.submit(str(raw), f"n{i}", {})
    stop.set()
    t.join()
    assert errors == []
    jm.shutdown()
