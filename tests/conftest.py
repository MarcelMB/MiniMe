import sys
from pathlib import Path

import pytest

from cafast import pipeline
from cafast.params import Params
from cafast.testing import synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tiny_session(tmp_path):
    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return {"raw_dir": raw, "setup": synth.synth_setup(64, 64, 300, 0), "T": 300, "h": 64, "w": 64}


# the tiny-session parameters of test_pipeline_end_to_end_tiny
TINY = Params(
    tile=32,
    seeds_wnd=100,
    seeds_stp=50,
    seeds_max_wnd=8,
    template_frames=50,
    inita_wnd=8,
    dl_wnd=5,
    size_thres=(9, None),
    svd_k0=16,
    motion_before_tophat=False,
)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """A full pipeline run on the tiny synthetic session (shared across test modules)."""
    raw = tmp_path_factory.mktemp("w2") / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return pipeline.run(raw, TINY, run_name="full", workers=1)
