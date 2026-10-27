import json
import shutil

import numpy as np
import pytest

from caanfast import io
from caanfast.serve.frames import FrameSource, raw_frame, recon_frame, store_frame
from caanfast.serve.runs import RunView


@pytest.mark.parametrize("t", [0, 99, 100, 250, 299])
def test_raw_frame_matches_sequential_decode(tiny_run, cache_dir, t):
    rv = RunView(tiny_run, cache_dir)
    info = rv.raw_info
    seg, k = divmod(t, 100)
    want = io.read_avi(info.segments[seg], max_frames=k + 1, bgr=info.bgr)[k]
    np.testing.assert_array_equal(raw_frame(info, t), want.astype(np.float32))


def test_store_and_recon_frames_have_the_frame_shape(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    pre = store_frame(tiny_run / "tiles", rv.factors, 10)
    rec = recon_frame(rv.factors, 10)
    assert pre.shape == rec.shape == (64, 64)
    # the reconstruction approximates the preprocessed frame
    assert np.corrcoef(pre.ravel(), rec.ravel())[0, 1] > 0.5


def test_frame_is_uint8_with_cached_contrast(tiny_run, cache_dir):
    fs = FrameSource(RunView(tiny_run, cache_dir))
    assert not fs.has_contrast("recon")
    f = fs.frame("recon", 299)
    assert f.dtype == np.uint8 and f.shape == (64, 64)
    doc = json.loads((cache_dir / fs.run.id / "contrast.json").read_text())
    assert doc["recon"][0] < doc["recon"][1]
    assert fs.has_contrast("recon")


def test_frame_errors(run_copy, cache_dir):
    shutil.rmtree(run_copy / "tiles")
    pj = json.loads((run_copy / "params.json").read_text())
    pj["raw_dir"] = str(run_copy / "unplugged")
    (run_copy / "params.json").write_text(json.dumps(pj))
    fs = FrameSource(RunView(run_copy, cache_dir))
    assert not fs.available("pre") and not fs.available("raw") and fs.available("recon")
    with pytest.raises(LookupError):
        fs.frame("pre", 0)
    with pytest.raises(LookupError):
        fs.frame("raw", 0)
    with pytest.raises(IndexError):
        fs.frame("recon", 300)
    with pytest.raises(IndexError):
        fs.frame("recon", -1)
    with pytest.raises(ValueError):
        fs.frame("movie", 0)


def test_frames_past_the_end_raise_index_error(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    with pytest.raises(IndexError):
        raw_frame(rv.raw_info, 300)
    with pytest.raises(IndexError):
        store_frame(tiny_run / "tiles", rv.factors, 300)


def test_concurrent_contrast_keeps_both_entries(tiny_run, cache_dir):
    import threading

    fs = FrameSource(RunView(tiny_run, cache_dir))
    ths = [threading.Thread(target=fs.contrast, args=(s,)) for s in ("recon", "pre")]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    doc = json.loads((cache_dir / fs.run.id / "contrast.json").read_text())
    assert {"recon", "pre"} <= set(doc)
