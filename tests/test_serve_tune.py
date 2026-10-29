"""Tune inside serve: rerun and auto-tune jobs, the rerun query, and the Tune page's routes."""

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minime import window2
from minime.autotune import run as at
from minime.autotune import sweep
from minime.serve import jobs as jobs_mod
from minime.serve.app import create_app
from minime.serve.jobs import JobManager
from minime.serve.paramsmeta import defaults
from tests.conftest import wait_ok

D = 7.0
SMALL = sweep.Grids(
    seeds_diff_thres=(2, 3),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_snr=(0.5, 1.0),
    loop_merge_corr=(0.9,),
    n_confirm=2,
    max_extend=0,
)


@pytest.fixture(scope="module")
def tuned_run(tiny_run, tmp_path_factory):
    """A copy of the tiny run with a manual rerun ``r1`` and an auto-tune ``auto``."""
    run = tmp_path_factory.mktemp("tuned") / "t"
    shutil.copytree(tiny_run, run)
    window2.rerun(run, dataclasses.replace(window2.run_params(run), sparse_snr=0.5), name="r1")
    at.autotune(run, D, grids=SMALL)
    return run


def _client(tmp_path, jm=None):
    c = TestClient(create_app("tok", cache_root=tmp_path / "cache", jobs=jm))
    c.headers["X-Token"] = "tok"
    return c


def _rid(client, run):
    return client.get("/api/runs", params={"root": str(run)}).json()["runs"][0]["id"]


# --- the progress file of an auto-tune ----------------------------------------------


def test_step_log_writes_one_row_per_step_and_the_rerun(tmp_path):
    log = at.StepLog(tmp_path / "p.jsonl")
    for step in ("seeds", "seeds", "merge"):
        log.trial(sweep.Trial(step, {}, {}, None, None, 0.0))
    log.stage("seeds_init")
    log.stage("cnmf")
    log.close()
    rows = [
        json.loads(line) for line in (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(r["stage"], r["event"]) for r in rows] == [
        ("tune_seeds", "start"),
        ("tune_seeds", "end"),
        ("tune_merge", "start"),
        ("tune_merge", "end"),
        ("tune_rerun", "start"),
        ("tune_rerun", "end"),
    ]
    assert rows[1]["n"] == 2 and rows[3]["n"] == 1 and "seconds" in rows[-1]


def test_step_log_marks_a_failed_step(tmp_path):
    log = at.StepLog(tmp_path / "p.jsonl")
    log.trial(sweep.Trial("seeds", {}, {}, None, None, 0.0))
    log.close(error=True)
    rows = [
        json.loads(line) for line in (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["event"] == "error"


# --- jobs ---------------------------------------------------------------------------


def _noop(job, params_file):
    return [sys.executable, "-c", "pass"]


def test_submit_rerun_and_autotune_validate(tmp_path, tiny_run):
    jm = JobManager(tmp_path / "cache", command=_noop)
    with pytest.raises(ValueError, match="not a minime run"):
        jm.submit_rerun(str(tmp_path), "r", {})
    with pytest.raises(ValueError, match="not a window-2 parameter"):
        jm.submit_rerun(str(tiny_run), "r", {"tophat_wnd": 5})
    with pytest.raises(ValueError, match="reserved"):
        jm.submit_rerun(str(tiny_run), "auto", {})
    with pytest.raises(ValueError, match="unknown parameter"):
        jm.submit_rerun(str(tiny_run), "r", {"bogus": 1})
    with pytest.raises(ValueError, match="enter a number"):
        jm.submit_autotune(str(tiny_run), {"diameter": "seven"})
    with pytest.raises(ValueError, match="unknown auto-tune option"):
        jm.submit_autotune(str(tiny_run), {"grid": 1})
    job = jm.submit_autotune(str(tiny_run), {"diameter": "7", "tune_frames": None})
    assert job.kind == "autotune" and job.tune_args == {"diameter": 7.0}
    assert Path(job.out_dir).parts[-2:] == ("reruns", "auto")
    with pytest.raises(ValueError, match="already writes"):
        jm.submit_autotune(str(tiny_run))
    jm.shutdown()


def test_commands_by_kind(tmp_path):
    base = dict(id="j", raw_dir="/raw", name="n", params={}, keep_store=False)
    run = jobs_mod.Job(**base, tune=True)
    assert jobs_mod.default_command(run, tmp_path / "p.json")[2:4] == ["minime", "run"]
    rerun = jobs_mod.Job(**base, kind="rerun", run_dir="/run")
    cmd = jobs_mod.default_command(rerun, tmp_path / "p.json")
    assert cmd[2:5] == ["minime", "cnmf", "/run"] and "--name" in cmd
    auto = jobs_mod.Job(**base, kind="autotune", run_dir="/run", tune_args={"diameter": 7.0})
    cmd = jobs_mod.default_command(auto, tmp_path / "p.json")
    assert cmd[2:5] == ["minime", "autotune", "/run"]
    assert cmd[cmd.index("--diameter") + 1] == "7.0" and "--progress" in cmd


def test_a_run_with_tune_chains_the_autotune(tmp_path, tiny_run, monkeypatch):
    """The second command runs after the first with the job still ``running``; its progress
    file is read next to the run's."""
    raw = tiny_run.parent.parent / "raw"
    line = json.dumps({"stage": "tune_seeds", "event": "end", "t": time.time() + 1, "n": 3})
    monkeypatch.setattr(
        jobs_mod,
        "tune_command",
        lambda run_dir, args, progress: [
            sys.executable,
            "-c",
            f"open({str(progress)!r}, 'w').write({line!r} + '\\n')",
        ],
    )
    jm = JobManager(tmp_path / "cache", command=_noop)
    job = jm.submit(str(raw), "chained", {}, tune=True, tune_args={"diameter": 7})
    done = jm.wait(job.id, timeout=30)
    assert done.status == "done", jm.log_lines(job.id)
    assert [e["stage"] for e in jm.progress(done)] == ["tune_seeds"]
    jm.shutdown()


def test_a_failing_first_command_skips_the_chained_one(tmp_path, tiny_run, monkeypatch):
    raw = tiny_run.parent.parent / "raw"
    called = tmp_path / "called"
    monkeypatch.setattr(
        jobs_mod,
        "tune_command",
        lambda *a: [sys.executable, "-c", f"open({str(called)!r}, 'w')"],
    )
    jm = JobManager(tmp_path / "cache", command=lambda j, p: [sys.executable, "-c", "exit(2)"])
    job = jm.submit(str(raw), "fails", {}, tune=True)
    assert jm.wait(job.id, timeout=30).status == "failed"
    assert not called.exists()
    jm.shutdown()


def test_a_real_rerun_job(tmp_path, run_copy):
    jm = JobManager(tmp_path / "cache")
    c = _client(tmp_path, jm)
    r = c.post(
        "/api/jobs/rerun",
        json={"run_dir": str(run_copy), "name": "r1", "params": {"sparse_snr": 0.5}},
    )
    assert r.status_code == 200, r.text
    job = jm.wait(r.json()["id"], timeout=300)
    assert job.status == "done", jm.log_lines(job.id)[-20:]
    assert "cnmf" in [e["stage"] for e in jm.progress(job) if e["event"] == "end"]
    rows = c.get(f"/api/runs/{_rid(c, run_copy)}/reruns").json()["reruns"]
    assert [x["name"] for x in rows] == ["r1"]
    assert rows[0]["status"] == "done" and rows[0]["changed"]["sparse_snr"][1] == 0.5
    assert c.post("/api/jobs/autotune", json={"run_dir": "/nope"}).status_code == 400
    jm.shutdown()


# --- the Tune page's routes ---------------------------------------------------------


def test_reruns_listing_and_proposal(tmp_path, tuned_run):
    c = _client(tmp_path)
    rid = _rid(c, tuned_run)
    rows = {x["name"]: x for x in c.get(f"/api/runs/{rid}/reruns").json()["reruns"]}
    assert set(rows) == {"r1", "auto"}
    assert rows["r1"]["changed"]["sparse_snr"][1] == 0.5 and not rows["r1"]["autotune"]
    assert rows["auto"]["autotune"] and rows["auto"]["counts"]["units_final"] > 0
    assert set(rows["r1"]["params"]) == set(window2.WINDOW2_PARAMS)
    doc = c.get(f"/api/runs/{rid}/autotune").json()
    assert doc["diameter"] == D and set(doc["proposed"]) == set(sweep.SEARCHED)
    assert set(doc["curves"]) <= set(sweep.SEARCHED) and "sparse_snr" in doc["curves"]
    pt = doc["curves"]["sparse_snr"][0]
    assert set(pt) == {"value", "good", "flagged"}


def test_flags_and_pairs(tmp_path, tuned_run):
    c = _client(tmp_path)
    rid = _rid(c, tuned_run)
    r = wait_ok(c, f"/api/runs/{rid}/flags?rerun=auto")
    assert r.status_code == 200
    f = r.json()
    assert f["checks"]["diameter"] == D and set(f["reasons"]) == set(at.score.REASONS)
    assert f["good"] + f["flagged"] == len(
        c.get(f"/api/runs/{rid}/units?rerun=auto").json()["units"]
    )
    r = wait_ok(c, f"/api/runs/{rid}/flags")  # the run itself, scored with the same checks
    assert r.status_code == 200 and r.json()["good"] + r.json()["flagged"] > 0
    p = wait_ok(c, f"/api/runs/{rid}/pairs?rerun=auto").json()
    assert len(p["near_corr"]) == len(p["corr_bins"]) - 1
    assert len(p["nn_dist"]) == len(p["nn_bins"]) - 1 and p["dup_corr"] == f["checks"]["dup_corr"]


def test_flags_need_an_autotune(tmp_path, tiny_run):
    c = _client(tmp_path)
    rid = _rid(c, tiny_run)
    assert c.get(f"/api/runs/{rid}/flags").status_code == 404
    assert c.get(f"/api/runs/{rid}/autotune").status_code == 404
    assert c.get(f"/api/runs/{rid}/reruns").json() == {"reruns": []}


def test_the_rerun_query_reads_the_reruns_stages(tmp_path, tuned_run):
    c = _client(tmp_path)
    rid = _rid(c, tuned_run)
    s = c.get(f"/api/runs/{rid}/seeds?rerun=r1").json()
    assert set(s) == {"init", "pnr", "merged"} and len(s["init"]) >= len(s["merged"]) > 0
    ini = c.get(f"/api/runs/{rid}/units?rerun=r1&stage=init").json()
    fin = c.get(f"/api/runs/{rid}/units?rerun=r1").json()
    assert ini["units"] and fin["units"] and ini["units"][0]["peak"] is None
    assert c.get(f"/api/runs/{rid}/units?rerun=nope").status_code == 404
    assert c.get(f"/api/runs/{rid}/units?rerun=../t").status_code == 404
    ov = wait_ok(c, f"/api/runs/{rid}/traces/overview?rerun=r1")
    assert ov.status_code == 200
    n = len(fin["units"])
    assert c.get(f"/api/runs/{rid}/traces/{n - 1}?rerun=r1&stop=10").status_code == 200


def test_params_defaults_flag_the_window2_parameters():
    d = {p["name"]: p["window2"] for p in defaults()}
    assert d["sparse_snr"] and d["merge_corr"] and not d["tophat_wnd"] and not d["workers"]
