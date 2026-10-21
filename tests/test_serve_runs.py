import json
import shutil
import time

import pytest

from caanfast.serve.cache import Background, RunCache
from caanfast.serve.runs import (
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


def test_background_runs_once_and_reports_failure():
    calls = []
    bg = Background()

    def slow():
        calls.append(1)
        time.sleep(0.2)

    assert bg.ensure("a", slow)["status"] == "computing"
    assert bg.ensure("a", slow)["status"] == "computing"
    time.sleep(0.4)
    assert calls == [1]

    def boom():
        raise RuntimeError("no")

    bg.ensure("b", boom)
    time.sleep(0.1)
    assert bg.ensure("b", boom) == {"status": "failed", "error": "RuntimeError: no"}
