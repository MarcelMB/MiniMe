import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from caanfast.serve.app import create_app
from caanfast.serve.jobs import JobManager
from caanfast.testing import synth
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
