import shutil
import sys
import time as _time
from pathlib import Path

import pytest

from caanfast.testing import synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tiny_session(tmp_path):
    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return {"raw_dir": raw, "setup": synth.synth_setup(64, 64, 300, 0), "T": 300, "h": 64, "w": 64}


TINY_PARAMS = dict(
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
    """One finished run (tile store kept) on a 300-frame 64 x 64 synthetic session.

    Shared by the serve tests, which must not modify it; the folder name has a space, like
    ``/Volumes/Extreme SSD``. Use ``run_copy`` to change files.
    """
    from caanfast import pipeline
    from caanfast.params import Params

    root = tmp_path_factory.mktemp("serve data")
    raw = root / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return pipeline.run(raw, Params(**TINY_PARAMS), run_name="t", workers=1, keep_store=True)


@pytest.fixture
def run_copy(tiny_run, tmp_path):
    """A writable copy of ``tiny_run`` (its params.json still points at the same raw folder)."""
    dst = tmp_path / "copy run" / "t"
    shutil.copytree(tiny_run, dst)
    return dst


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def client(cache_dir, tmp_path):
    from fastapi.testclient import TestClient

    from caanfast.serve.app import create_app

    app = create_app("tok", cache_root=cache_dir, static_dir=tmp_path / "no-static")
    c = TestClient(app)
    c.headers["X-Token"] = "tok"
    return c


def wait_ok(client, url, timeout=120):
    """GET ``url`` until it stops answering 202 (a background computation)."""
    t0 = _time.time()
    while True:
        r = client.get(url)
        if r.status_code != 202:
            return r
        assert _time.time() - t0 < timeout, f"{url} still computing"
        _time.sleep(0.2)
