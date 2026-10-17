import sys
from pathlib import Path

import pytest

from cafast.testing import synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tiny_session(tmp_path):
    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return {"raw_dir": raw, "setup": synth.synth_setup(64, 64, 300, 0), "T": 300, "h": 64, "w": 64}
