import os
import shutil
import tempfile
import time as _time

import pytest

from minime import pipeline
from minime.params import Params
from minime.testing import synth

# runs and serve write the known-folders list and jobs here, never into ~/.cache/minime
os.environ["MINIME_CACHE"] = tempfile.mkdtemp(prefix="minime-test-cache-")


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
# the tiny-session parameters of test_pipeline_end_to_end_tiny
TINY = Params(**TINY_PARAMS)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """One finished run (tile store kept) on a 300-frame 64 x 64 synthetic session.

    Shared by the serve tests, which must not modify it; the folder name has a space, like
    ``/Volumes/Extreme SSD``. Use ``run_copy`` to change files.
    """
    root = tmp_path_factory.mktemp("serve data")
    raw = root / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return pipeline.run(raw, Params(**TINY_PARAMS), run_name="t", workers=1, keep_store=True)


@pytest.fixture(scope="session")
def tiny_w2_run(tiny_run, tmp_path_factory):
    """A session-wide copy of ``tiny_run`` that the window-2, tune and autotune tests may
    write reruns into (``tiny_run`` itself must stay as the pipeline left it)."""
    dst = tmp_path_factory.mktemp("w2") / "full"
    shutil.copytree(tiny_run, dst)
    return dst


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

    from minime.serve.app import create_app

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
