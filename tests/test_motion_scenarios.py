"""Small synthetic sessions with a persistent offset, a drift or sustained jumps: the default
spread template + template fill leaves (almost) no false units along the frame edges, the
first-frames template + zero fill (before Nov 2026) leaves strip-shaped ones."""

import numpy as np
import pytest

from minime import pipeline, score
from minime.params import Params
from minime.testing import motion_traces, synth

SIDE, T = 256, 600
SMALL = dict(
    template_frames=100,  # the offset (frame 100) comes after the first-frames template
    seeds_wnd=300,
    seeds_stp=150,
    inita_wnd=8,
    size_thres=(9, None),
    svd_k0=16,
)
LEGACY = dict(template_source="first", motion_fill="zero")


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("motion scenarios")
    out = {}
    for name in ("offset", "drift", "jumps"):
        sh = motion_traces.scenario_shifts(name, T, offset_at=T // 6)
        raw = root / name / "raw"
        opts = synth.SynthOptions(motion_margin=12)
        synth.write_session(raw, T / 30 / 60, 30, 300, 0, SIDE, options=opts, shifts=sh)
        out[name] = raw
    return out


def _false_edge_units(raw, params: Params, name: str) -> tuple[int, dict]:
    run = pipeline.run(raw, params, run_name=name, workers=2)
    s = score.score_pipeline_dir(run)
    return s["false_edge_units"], s


@pytest.mark.parametrize("name", ["offset", "drift", "jumps"])
def test_default_motion_leaves_no_false_edge_units(sessions, name):
    new, s = _false_edge_units(sessions[name], Params(**SMALL), "new")
    old, s_old = _false_edge_units(sessions[name], Params(**SMALL, **LEGACY), "legacy")
    assert s["alignment_ok"] and s["recall"] > 0.85, s
    assert new <= 2, (new, old)  # 0, 0, 2 on macOS arm64
    assert old >= new + 2, (new, old)  # 4, 4, 5: the scenario does show the failure
    assert s["f1"] >= s_old["f1"] - 0.01, (s["f1"], s_old["f1"])


def test_scenario_shifts_are_the_session_motion(sessions):
    """The given trace is the truth's motion."""
    truth = score.load_truth(sessions["jumps"].parent / "meta.json")
    sh = motion_traces.scenario_shifts("jumps", T)
    assert np.array_equal(truth["shifts"], sh)
