import json
from pathlib import Path

from minime import window2
from minime.cli import _input_summary


def test_input_summary_lists_order_ignored_files_and_frame_rate():
    segs = [Path(f"{i}.avi") for i in range(8)]
    lines = _input_summary(segs, ["behavior.avi"], None, 20.0)
    assert lines[0] == "8 video(s), read in this order: 0.avi, 1.avi, 2.avi, ..., 6.avi, 7.avi"
    assert lines[1] == "ignored (name does not start with a number): behavior.avi"
    assert lines[2] == "frame rate: 20 fps (from the metadata file)"
    assert _input_summary(segs[:1], [], 19.5, 20.0)[-1] == "frame rate: 19.5 fps (--fps)"
    assert "pass --fps" in _input_summary(segs[:1], [], None, None)[-1]


def test_run_fps_prefers_the_given_rate_over_meta_json(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    doc = {"raw_dir": str(raw), "fps": None}
    (run / "params.json").write_text(json.dumps(doc), encoding="utf-8")
    assert window2.run_fps(run) == 30.0
    (run / "params.json").write_text(json.dumps({**doc, "fps": 19.5}), encoding="utf-8")
    assert window2.run_fps(run) == 19.5
