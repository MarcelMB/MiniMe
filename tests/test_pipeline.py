import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from typer.testing import CliRunner

from minime import cli, pipeline, report
from minime.params import Params
from minime.testing import synth


def test_pipeline_end_to_end_tiny(tiny_session):
    # default seed filters (auto PNR, no gmm_refine): with gmm_refine on, its 2-GMM on ptp
    # splits the 7 toy cells into bright and dim and drops the dim ones
    p = Params(
        tile=32,
        seeds_wnd=100,
        seeds_stp=50,
        seeds_max_wnd=8,
        template_frames=50,
        inita_wnd=8,
        dl_wnd=5,
        size_thres=(9, None),
        svd_k0=16,
        # 64 x 64 px is too small to register the toy movie in either order (3.8 px old,
        # 5.8 px band-passed; test_pass1_shifts_match_truth_in_both_orders uses 128 x 128)
        motion_before_tophat=False,
    )
    out = pipeline.run(tiny_session["raw_dir"], p, run_name="t", workers=2)
    assert out == tiny_session["raw_dir"].parent / "processed" / "t"
    assert (out / "factors" / "meta.json").exists() and not (out / "tiles").exists()
    A = sps.load_npz(out / "cnmf" / "A.npz")
    C = np.load(out / "cnmf" / "C.npy")
    rep = json.loads((out / "report.json").read_text())
    assert A.shape[1] == 64 * 64 and C.shape == (A.shape[0], 300)
    assert rep["counts"]["units_final"] == A.shape[0]
    cents = synth.true_centroids(tiny_session["setup"])
    cents = cents[(cents >= 4).all(1) & (cents < 60).all(1)]
    from minime.init import centroids

    uc = centroids(A, 64, 64)
    d = np.sqrt(((cents[:, None] - uc[None]) ** 2).sum(-1)).min(1)
    assert (d < 5).mean() >= 0.8
    # output contract
    assert np.load(out / "cnmf" / "YrA.npy").shape == C.shape
    assert np.load(out / "cnmf" / "b.npy").shape == (64, 64)
    assert np.load(out / "cnmf" / "f.npy").shape == (300,)
    units = pd.read_csv(out / "cnmf" / "units.csv")
    assert list(units.columns) == ["unit_id", "centroid_h", "centroid_w", "size_px", "peak"]
    assert len(units) == A.shape[0]
    prm = json.loads((out / "params.json").read_text())
    assert prm["tile"] == 32 and prm["raw_dir"] == str(tiny_session["raw_dir"].resolve())
    assert {"raw_hash", "minime_version"} <= set(prm)
    assert (out / "motion" / "meta.json").exists()
    assert rep["motion"]["sidecar_reused"] is False
    assert rep["bytes"]["store_peak"] > 0 and rep["bytes"]["factors"] > 0
    assert rep["workers"] == 2 and rep["counts"]["seeds_init"] >= rep["counts"]["seeds_pnr"]
    assert "cnmf" in {r["stage"] for r in rep["stages"]}
    for k in ("init", "pnr", "merged"):
        pos = np.load(out / "seeds" / f"{k}.npy")
        assert pos.dtype == np.float32 and pos.shape == (rep["counts"][f"seeds_{k}"], 2)
    A0 = sps.load_npz(out / "init" / "A.npz")
    assert A0.shape == (rep["counts"]["units_init_merge"], 64 * 64)
    events = [json.loads(s) for s in (out / "progress.jsonl").read_text().splitlines()]
    ends = [e["stage"] for e in events if e["event"] == "end"]
    assert ends == [r["stage"] for r in rep["stages"]]
    (out / "stale.txt").write_text("from a previous run")
    # second run reuses the sidecar and overwrites
    out2 = pipeline.run(tiny_session["raw_dir"], p, run_name="t", workers=1)
    rep2 = json.loads((out2 / "report.json").read_text())
    assert rep2["motion"]["sidecar_reused"] is True
    assert not (out2 / "stale.txt").exists()
    assert np.array_equal(sps.load_npz(out2 / "cnmf" / "A.npz").toarray(), A.toarray())


def test_stage_timer_progress(tmp_path):
    from minime.report import StageTimer

    timer = StageTimer()
    with timer("probe"):
        pass
    timer.attach_progress(tmp_path / "progress.jsonl")
    with timer("pass1"):
        pass
    lines = [json.loads(s) for s in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [(x["stage"], x["event"]) for x in lines] == [
        ("probe", "end"),
        ("pass1", "start"),
        ("pass1", "end"),
    ]
    assert lines[2]["seconds"] >= 0 and "seconds" not in lines[1]
    with pytest.raises(RuntimeError), timer("cnmf"):
        raise RuntimeError("boom")
    lines = [json.loads(s) for s in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [(x["stage"], x["event"]) for x in lines[3:]] == [("cnmf", "start"), ("cnmf", "error")]
    assert lines[4]["seconds"] >= 0 and lines[4]["t"] > 0


def test_pipeline_refuses_out_dir_containing_raw(tmp_path):
    raw = tmp_path / "proc" / "t" / "raw"
    raw.mkdir(parents=True)
    with pytest.raises(ValueError, match="raw"):
        pipeline.run(raw, Params(), run_name="t", out_root=tmp_path / "proc")
    assert raw.exists()


def test_out_dir_of_relative_raw_dir_lies_beside_it(tmp_path, monkeypatch):
    raw = tmp_path / "sess" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.chdir(raw)
    assert pipeline.out_dir_for(Path("."), "t") == tmp_path / "sess" / "processed" / "t"
    with pytest.raises(ValueError, match="inside the raw dir"):
        pipeline._check_out_dir(Path("."), raw / "processed" / "t")


def test_stage_timer(tmp_path):
    timer = report.StageTimer()
    with timer("a"):
        time.sleep(0.01)
    timer.counts["n"] = 3
    timer.to_json(tmp_path / "r.json", {"version": "x"})
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["stages"][0]["stage"] == "a" and rep["stages"][0]["seconds"] >= 0.01
    assert rep["counts"] == {"n": 3} and rep["version"] == "x"


def test_cli_run_wires_options(tmp_path, monkeypatch):
    seen = {}

    def fake_run(raw_dir: Path, params: Params, **kw) -> Path:
        seen.update(raw_dir=raw_dir, params=params, **kw)
        out = tmp_path / "out"
        out.mkdir()
        (out / "report.json").write_text(json.dumps({"counts": {"units_final": 7}}))
        return out

    monkeypatch.setattr(pipeline, "run", fake_run)
    Params(tile=32).to_json(tmp_path / "p.json")
    res = CliRunner().invoke(
        cli.app,
        ["run", str(tmp_path), "--params", str(tmp_path / "p.json"), "--workers", "3"]
        + ["--name", "x", "--keep-store"],
    )
    assert res.exit_code == 0, res.output
    assert seen["params"].tile == 32 and seen["workers"] == 3
    assert seen["run_name"] == "x" and seen["keep_store"] is True
    assert "units_final" in res.output and str(tmp_path / "out") in res.output


def test_cli_run_refuses_existing_run_without_overwrite(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "processed" / "x"
    out.mkdir(parents=True)
    (out / "params.json").write_text("{}")
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **kw: calls.append(1))
    res = CliRunner().invoke(cli.app, ["run", str(raw), "--name", "x"])
    assert res.exit_code == 1 and "--overwrite" in res.output and not calls


@pytest.mark.parametrize("use_gmm", [False, True])
def test_seeds_stage_runs_gmm_refine_only_when_enabled(monkeypatch, use_gmm):
    from types import SimpleNamespace

    from minime import corr, seeds

    df = pd.DataFrame({"height": [1, 2, 3], "width": [1, 2, 3]})
    called = []

    def fake_gmm(f, s):
        called.append(len(s))
        return s.assign(mask_gmm=[True, False, True])

    monkeypatch.setattr(seeds, "seeds_init", lambda *a: df)
    monkeypatch.setattr(seeds, "pnr_refine", lambda f, s, thres, pnr=None: s.assign(mask_pnr=True))
    monkeypatch.setattr(seeds, "gmm_refine", fake_gmm)
    monkeypatch.setattr(corr, "seeds_merge", lambda fc, s, mp, d, c: s.assign(mask_mrg=True))
    fac = SimpleNamespace(block_max=None, block_edges=None, T=10, max_proj=None)
    assert Params().gmm_refine is False
    timer = report.StageTimer()
    prm = Params(gmm_refine=use_gmm)
    s = pipeline.seeds_merged(fac, None, pipeline.seeds_unmerged(fac, prm, timer), prm, timer)
    assert called == ([3] if use_gmm else [])
    assert (
        len(s) == timer.counts["seeds_gmm"] == timer.counts["seeds_merged"] == (2 if use_gmm else 3)
    )
    assert ("gmm_refine" in {r["stage"] for r in timer.rows}) is use_gmm


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b"])
def test_pipeline_rejects_unsafe_run_names(tmp_path, name):
    raw = tmp_path / "raw"
    raw.mkdir()
    keep = tmp_path / "processed" / "other_run"
    keep.mkdir(parents=True)
    (keep / "params.json").write_text(json.dumps({"minime_version": "0"}))
    with pytest.raises(ValueError, match="run name"):
        pipeline.run(raw, Params(), run_name=name)
    assert (keep / "params.json").exists()


def test_reset_dir_wipes_only_empty_or_minime_dirs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    pipeline._reset_dir(empty)
    assert empty.is_dir() and not any(empty.iterdir())
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"stages": []}))
    (run / "old.npy").write_text("x")
    pipeline._reset_dir(run)
    assert run.is_dir() and not any(run.iterdir())
    old = tmp_path / "old"  # run written under the old name cafast
    old.mkdir()
    (old / "params.json").write_text(json.dumps({"cafast_version": "0"}))
    pipeline._reset_dir(old)
    assert old.is_dir() and not any(old.iterdir())
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "params.json").write_text(json.dumps({"tile": 64}))  # no minime marker key
    with pytest.raises(ValueError, match="refusing to delete"):
        pipeline._reset_dir(fake)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="refusing to delete"):  # same guard from run()
        pipeline.run(raw, Params(), run_name="fake", out_root=tmp_path)
    assert (fake / "params.json").exists()


def test_a_failed_run_removes_the_tile_store(tiny_session, tmp_path, monkeypatch):
    from minime import pass2

    def boom(*a, **kw):
        raise RuntimeError("pass 2 failed")

    monkeypatch.setattr(pass2, "run_pass2", boom)
    p = Params(tile=32, seeds_wnd=100, seeds_stp=50, template_frames=50)
    with pytest.raises(RuntimeError, match="pass 2"):
        pipeline.run(tiny_session["raw_dir"], p, run_name="f", out_root=tmp_path)
    assert (tmp_path / "f" / "motion").is_dir() and not (tmp_path / "f" / "tiles").exists()
