import json
import shutil
import time

import pytest

from minime.serve.cache import Background, RunCache
from minime.serve.runs import (
    Registry,
    RunView,
    changed_params,
    find_runs,
    fps_from_files,
    is_run,
    run_id,
)


def test_find_runs_finds_the_run_below_its_session(tiny_run):
    assert is_run(tiny_run) and not is_run(tiny_run.parent)
    assert find_runs(tiny_run.parent.parent) == [tiny_run]
    assert find_runs(tiny_run) == [tiny_run]


def test_run_id_is_stable_and_short(tiny_run):
    assert run_id(tiny_run) == run_id(tiny_run / ".." / "t")
    assert len(run_id(tiny_run)) == 12


def test_run_view_summary(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    s = rv.summary()
    assert (s["H"], s["W"], s["T"]) == (64, 64, 300)
    assert s["fps"] == 30.0
    assert len(s["ranks"]) == 4 and s["tile"] == 32
    assert s["available"] == {"raw": True, "pre": True, "seeds_saved": True, "init_saved": True}
    assert s["counts"]["units_final"] == rv.A.shape[0]
    assert "tile" in s["params_changed"] and "workers" not in s["params_changed"]
    assert s["name"] == "t" and s["id"] == rv.id


def test_run_view_reports_missing_parts(run_copy, cache_dir):
    pj = json.loads((run_copy / "params.json").read_text())
    pj["raw_dir"] = str(run_copy / "gone")
    (run_copy / "params.json").write_text(json.dumps(pj))
    for d in ("tiles", "seeds", "init"):
        shutil.rmtree(run_copy / d)
    rv = RunView(run_copy, cache_dir)
    assert rv.summary()["available"] == {
        "raw": False,
        "pre": False,
        "seeds_saved": False,
        "init_saved": False,
    }
    assert rv.fps is None
    rv.set_fps(20)
    assert RunView(run_copy, cache_dir).fps == 20.0
    with pytest.raises(ValueError):
        rv.set_fps(0)


def test_fps_from_miniscope_metadata(tmp_path):
    raw = tmp_path / "My_V4_Miniscope"
    raw.mkdir()
    assert fps_from_files(raw) is None
    (raw / "metaData.json").write_text(json.dumps({"frameRate": "20FPS"}))
    assert fps_from_files(raw) == 20.0
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30}))
    assert fps_from_files(raw) == 30.0


def test_changed_params_compares_tuples_as_lists():
    assert changed_params({"size_thres": [25, None], "tile": 32}) == {
        "tile": {"value": 32, "default": 64}
    }


def test_registry_reopens_by_id_after_restart(tiny_run, cache_dir):
    rid = Registry(cache_dir).open(tiny_run).id
    assert Registry(cache_dir).get(rid).path == tiny_run.resolve()
    with pytest.raises(KeyError):
        Registry(cache_dir).get("0" * 12)


def test_run_cache_roundtrip(cache_dir):
    import numpy as np

    c = RunCache("abc", cache_dir)
    assert not c.has("x.npy") and c.load_json("a.json", {}) == {}
    c.save_npy("x.npy", np.arange(3, dtype=np.float32))
    c.save_json("a.json", {"k": 1})
    assert c.has("x.npy") and c.load_json("a.json") == {"k": 1}
    np.testing.assert_array_equal(c.load_npy("x.npy"), [0, 1, 2])


def _until(pred, timeout=5.0):
    """Poll ``pred`` instead of sleeping a fixed time: CI runners can be slow."""
    t0 = time.time()
    while not pred():
        assert time.time() - t0 < timeout, "condition not reached"
        time.sleep(0.01)


def test_background_runs_once_and_reports_failure():
    calls = []
    bg = Background()

    def slow():
        calls.append(1)
        time.sleep(0.2)

    assert bg.ensure("a", slow)["status"] == "computing"
    assert bg.ensure("a", slow)["status"] == "computing"
    _until(lambda: "a" not in bg._running)
    assert calls == [1]

    def boom():
        raise RuntimeError("no")

    bg.ensure("b", boom)
    _until(lambda: bg.ensure("b", boom)["status"] == "failed")
    assert bg.ensure("b", boom) == {"status": "failed", "error": "RuntimeError: no"}


def test_registry_open_is_thread_safe(run_copy, cache_dir):
    from concurrent.futures import ThreadPoolExecutor

    reg = Registry(cache_dir)
    with ThreadPoolExecutor(8) as ex:
        views = list(ex.map(lambda _: reg.open(run_copy), range(8)))
    assert len({id(v) for v in views}) == 1
    assert views[0].id in json.loads((cache_dir / "runs.json").read_text())


def _rewrite_report(run):
    p = run / "report.json"
    doc = json.loads(p.read_text())
    doc["rewritten"] = True
    p.write_text(json.dumps(doc))


def test_rewritten_run_clears_its_cache_but_keeps_fps(run_copy, cache_dir):
    v = RunView(run_copy, cache_dir)
    v.cache.save_json("contrast.json", {"a": 1})
    v.set_fps(20)
    assert RunView(run_copy, cache_dir).cache.has("contrast.json")
    _rewrite_report(run_copy)
    v2 = RunView(run_copy, cache_dir)
    assert not v2.cache.has("contrast.json")
    assert v2.cache.load_json("fps.json") == {"fps": 20.0}
    assert v2.cache.has("stamp.json")


def test_registry_replaces_a_stale_view(run_copy, cache_dir):
    reg = Registry(cache_dir)
    v = reg.open(run_copy)
    assert reg.get(v.id) is v
    _rewrite_report(run_copy)
    assert reg.get(v.id) is not v
    assert reg.open(run_copy) is reg.get(v.id)


def test_find_runs_skips_library(tmp_path, tiny_run):
    lib = tmp_path / "Library" / "x"
    lib.mkdir(parents=True)
    for m in ("params.json", "report.json"):
        (lib / m).write_text("{}")
    shutil.copytree(tiny_run, tmp_path / "data" / "t")
    assert find_runs(tmp_path) == [tmp_path / "data" / "t"]


def test_only_the_two_most_recent_runs_keep_their_heavy_data(run_copy, cache_dir):
    reg = Registry(cache_dir)
    paths = [run_copy]
    for i in (1, 2):
        dst = run_copy.parent / f"t{i}"
        shutil.copytree(run_copy, dst)
        paths.append(dst)
    views = []
    for p in paths:
        v = reg.open(p)
        assert v.factors.H == v.H and v.A.shape[1] == v.H * v.W and v.sidecar is not None
        views.append(v)
    first = views[0]
    assert first._factors is None and first._A is None and first._sidecar is None
    assert all(v._factors is not None for v in views[1:])
    assert reg.get(first.id) is first  # still registered; the parts load again on use
    assert first.factors is not None
    assert views[1]._factors is None  # now the least recent of three


def test_heavy_parts_load_once_under_concurrency(run_copy, cache_dir, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from minime import factors as factors_mod

    calls = []
    real = factors_mod.Factors.load

    def slow_load(d):
        calls.append(1)
        time.sleep(0.2)
        return real(d)

    monkeypatch.setattr(factors_mod.Factors, "load", staticmethod(slow_load))
    v = RunView(run_copy, cache_dir)
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda _: v.factors, range(4)))
    assert calls == [1] and len({id(f) for f in got}) == 1
    v.release()
    assert v._factors is None


def test_background_failures_expire_and_are_retried():
    bg = Background(failed_ttl=0.1)
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("no")

    bg.ensure("k", boom)
    _until(lambda: bg.ensure("k", boom)["status"] == "failed")
    assert calls == [1]
    _until(lambda: bg.ensure("k", boom)["status"] == "computing")  # the failure expired
    _until(lambda: calls == [1, 1])
