import numpy as np
import pytest

from minime import io
from minime.params import Params
from minime.serve import preview
from minime.serve.arrays import decode
from minime.serve.frames import raw_frame


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    from minime.testing import synth

    raw = tmp_path_factory.mktemp("preview") / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return io.probe(raw)


def test_frame_panels_are_uint8_and_start_from_the_raw_frame(info):
    t = 150
    panels = preview.frame_panels(info, t, Params(tophat_wnd=5))
    assert panels.shape == (4, 64, 64) and panels.dtype == np.uint8
    np.testing.assert_array_equal(panels[0], raw_frame(info, t).astype(np.uint8))
    assert not np.array_equal(panels[0], panels[3])  # the tophat changed it


@pytest.mark.parametrize(
    ("t", "n", "t0"), [(0, 50, 0), (120, 50, 120), (280, 50, 250), (120, 500, 100)]
)
def test_clip_start_keeps_the_clip_inside_the_segment(info, t, n, t0):
    assert preview.clip_start(info, t, n) == t0


def test_clip_start_rejects_a_frame_beyond_the_session(info):
    with pytest.raises(IndexError):
        preview.clip_start(info, 300, 10)


def test_motion_check_on_a_short_clip(info):
    r = preview.motion_check(info, 280, Params(template_frames=20), n=40)
    assert (r["t0"], r["frames"], len(r["shifts"])) == (260, 40, 40)
    assert r["max_shift_px"] >= 0 and -1 <= r["quality_median"] <= 1


def test_preview_routes(client, info):
    raw = str(info.segments[0].parent)
    r = client.post("/api/raw/preview", json={"raw_dir": raw, "t": 10, "params": {}})
    assert r.status_code == 200
    assert decode(r.content, r.headers["X-Shape"], r.headers["X-Dtype"]).shape == (4, 64, 64)
    bad = {"raw_dir": raw, "t": 300, "params": {}}
    assert client.post("/api/raw/preview", json=bad).status_code == 400
    bad = {"raw_dir": raw, "t": 0, "params": {"bogus": 1}}
    assert client.post("/api/raw/preview", json=bad).status_code == 400
    r = client.post("/api/raw/motion", json={"raw_dir": raw, "t": 290, "params": {}})
    assert r.status_code == 200 and r.json()["t0"] == 200 and r.json()["frames"] == 100
    # the probe is cached per folder
    assert len(client.app.state.raw_infos) == 1


def test_params_derived(client):
    d = client.get("/api/params/derived", params={"diameter": 8}).json()
    assert d["tophat_wnd"] == 8 and d["dl_wnd"] == 8 and d["merge_min_dist"] == 3.2
    assert d["size_thres"] == [13, None]
    assert client.get("/api/params/derived", params={"diameter": 0}).status_code == 400
