import json

import numpy as np
import pytest

from minime import io, pass1, tiles
from minime.params import Params


def test_block_edges_match_rolling_windows():
    # 300 frames, step 50 -> 7 centres 0, 49.83, ..., 299; half-width 50
    e = pass1.seed_block_edges(300, wnd_size=100, stp_size=50)
    centres = np.linspace(0, 299, 7)
    expected = {0, 300}
    for c in centres:
        expected.add(int(np.floor(c - 50).clip(0)))
        expected.add(min(int(np.ceil(c + 50)), 300))
    assert e.tolist() == sorted(expected)


def test_params_json_roundtrip(tmp_path):
    p = Params(tile=32, pnr_thres=2.5)
    assert p.size_thres == (25, None) and Params().pnr_thres == "auto"
    p.to_json(tmp_path / "p.json")
    q = Params.from_json(tmp_path / "p.json")
    assert q == p and q.size_thres == (25, None) and isinstance(q.size_thres, tuple)
    assert set(p.motion_params()) == {
        "median_ksize",
        "tophat_wnd",
        "motion_max_shift",
        "motion_down",
        "template_frames",
        "motion_before_tophat",
        "motion_bandpass",
    }
    assert p.asdict()["tile"] == 32


def test_pass1_single_and_parallel_agree(tiny_session, tmp_path):
    info = io.probe(tiny_session["raw_dir"])
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    tmpl = pass1.build_template(info, p)
    grid = tiles.tile_grid(64, 64, 32)
    outs = []
    for workers in (1, 2):
        store = tiles.TileStore(tmp_path / f"s{workers}", grid, pass1.n_parts_for(info, workers))
        outs.append((store, pass1.run_pass1(info, store, tmpl, p, workers)))
    (s1, r1), (s2, r2) = outs
    assert r1.frames == r2.frames == 300 and r1.shifts.shape == (300, 2)
    assert r1.shifts.dtype == np.float32 and r1.quality.shape == (300,)
    assert np.allclose(r1.shifts, r2.shifts) and np.array_equal(r1.max_proj, r2.max_proj)
    assert np.array_equal(r1.block_max, r2.block_max)
    for t in grid:
        assert np.array_equal(s1.read_tile(t.index), s2.read_tile(t.index))
    # block maxima equal the max over the same corrected frames
    full = np.concatenate([s1.read_tile(t.index) for t in grid[:1]])
    e = r1.block_edges
    for bi in range(len(e) - 1):
        assert np.array_equal(r2.block_max[bi, :32, :32], full[e[bi] : e[bi + 1]].max(0))
    assert np.array_equal(r1.max_proj[:32, :32], full.max(0))


@pytest.mark.parametrize("before", [True, False])
def test_pass1_shifts_match_truth_in_both_orders(tmp_path, before):
    """Motion estimated before the tophat (band-passed frames) or after it (old order).

    128 x 128 px: at 64 x 64 neither order registers the synthetic movie (3.8 / 5.8 px).
    """
    from minime.testing import synth

    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=150, seed=0, side=128)
    info = io.probe(raw)
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=200, motion_before_tophat=before)
    q = Params.from_json(_write(tmp_path / "p.json", p))
    assert q.motion_bandpass == (1.0, 6.0) and isinstance(q.motion_bandpass, tuple)
    grid = tiles.tile_grid(128, 128, 64)
    store = tiles.TileStore(tmp_path / "s", grid, pass1.n_parts_for(info, 1))
    r = pass1.run_pass1(info, store, pass1.build_template(info, p), p, 1)
    true = synth.synth_setup(128, 128, 300, 0).shifts.astype(float)
    rmse = []
    for sign in (-1, 1):
        d = r.shifts - sign * true
        rmse.append(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()))
    assert min(rmse) < 0.5, rmse
    assert r.quality.min() > 0.3


def _write(path, p: Params):
    p.to_json(path)
    return path


def test_pass1_ragged_last_segment(tmp_path):
    from minime.testing import synth

    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=250 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    info = io.probe(raw)
    assert info.segment_frames == [100, 100, 50]
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    grid = tiles.tile_grid(64, 64, 32)
    store = tiles.TileStore(tmp_path / "s", grid, pass1.n_parts_for(info, 2))
    r = pass1.run_pass1(info, store, pass1.build_template(info, p), p, workers=2)
    assert r.frames == 250 and r.block_edges[-1] == 250
    full = store.read_tile(0)
    assert full.shape[0] == 250
    e = r.block_edges
    for bi in range(len(e) - 1):
        assert np.array_equal(r.block_max[bi, :32, :32], full[e[bi] : e[bi + 1]].max(0))


def test_pass1_sub_batches_straddling_block_edges(tiny_session, tmp_path):
    info = io.probe(tiny_session["raw_dir"])
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50, sub_batch=30)
    tmpl = pass1.build_template(info, p)
    grid = tiles.tile_grid(64, 64, 32)
    store = tiles.TileStore(tmp_path / "s", grid, pass1.n_parts_for(info, 1))
    r = pass1.run_pass1(info, store, tmpl, p, workers=1)
    e = r.block_edges
    full = store.read_tile(3)  # bottom-right tile
    for bi in range(len(e) - 1):
        assert np.array_equal(r.block_max[bi, 32:, 32:], full[e[bi] : e[bi + 1]].max(0))


def test_pass1_worker_starting_mid_block(tiny_session, tmp_path):
    info = io.probe(tiny_session["raw_dir"])
    p = Params(seeds_wnd=120, seeds_stp=40, template_frames=50)
    e = pass1.seed_block_edges(info.frames, p.seeds_wnd, p.seeds_stp)
    t_starts = np.cumsum([0, *info.segment_frames])[:-1]  # one segment per worker
    assert not {100, 200} & set(e.tolist()) and t_starts.tolist() == [0, 100, 200]
    grid = tiles.tile_grid(64, 64, 32)
    store = tiles.TileStore(tmp_path / "s", grid, pass1.n_parts_for(info, 3))
    assert store.n_parts == 3
    r = pass1.run_pass1(info, store, pass1.build_template(info, p), p, workers=3)
    full = store.read_tile(1)  # top-right tile
    for bi in range(len(e) - 1):
        assert np.array_equal(r.block_max[bi, :32, 32:], full[e[bi] : e[bi + 1]].max(0))


def test_pass1_part_count_mismatch_names_helper(tiny_session, tmp_path):
    info = io.probe(tiny_session["raw_dir"])
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    assert pass1.n_parts_for(info, 8) == 3
    store = tiles.TileStore(tmp_path / "s", tiles.tile_grid(64, 64, 32), n_parts=8)
    with pytest.raises(ValueError, match="n_parts_for"):
        pass1.run_pass1(info, store, pass1.build_template(info, p), p, workers=8)


def test_pass1_apply_only_uses_given_shifts(tiny_session, tmp_path):
    info = io.probe(tiny_session["raw_dir"])
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    tmpl = pass1.build_template(info, p)
    grid = tiles.tile_grid(64, 64, 32)
    s_est = tiles.TileStore(tmp_path / "est", grid, pass1.n_parts_for(info, 2))
    r_est = pass1.run_pass1(info, s_est, tmpl, p, workers=2)
    s_app = tiles.TileStore(tmp_path / "app", grid, pass1.n_parts_for(info, 2))
    r_app = pass1.run_pass1(info, s_app, tmpl, p, workers=2, shifts=r_est.shifts)
    assert np.array_equal(r_app.shifts, r_est.shifts)
    assert np.allclose(r_app.quality, r_est.quality)
    assert np.array_equal(r_app.block_max, r_est.block_max)
    for t in grid:
        assert np.array_equal(s_app.read_tile(t.index), s_est.read_tile(t.index))
    # the given shifts are applied as they are: zero shifts leave the frames unwarped
    s0 = tiles.TileStore(tmp_path / "zero", grid, pass1.n_parts_for(info, 1))
    r0 = pass1.run_pass1(info, s0, tmpl, p, workers=1, shifts=np.zeros((300, 2), np.float32))
    assert not r0.shifts.any() and not np.array_equal(s0.read_tile(0), s_est.read_tile(0))
    with pytest.raises(ValueError, match="shifts"):
        pass1.run_pass1(info, s0, tmpl, p, workers=1, shifts=np.zeros((10, 2), np.float32))


def _truncate(path, frac=0.6):
    import os

    os.truncate(path, int(path.stat().st_size * frac))


def test_probe_decodes_truncated_last_segment(tmp_path):
    from minime.testing import synth

    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    last = io.list_segments(raw)[-1]
    _truncate(last)
    assert io.frame_count(last) == 100  # the header still claims the full segment
    n_last = io.read_avi(last).shape[0]
    assert 0 < n_last < 100
    info = io.probe(raw)
    assert info.segment_frames == [100, 100, n_last] and info.frames == 200 + n_last
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    grid = tiles.tile_grid(64, 64, 32)
    store = tiles.TileStore(tmp_path / "s", grid, pass1.n_parts_for(info, 2))
    r = pass1.run_pass1(info, store, pass1.build_template(info, p), p, workers=2)
    assert r.frames == info.frames and r.block_edges[-1] == info.frames
    assert store.read_tile(0).shape[0] == info.frames


def test_pass1_names_truncated_middle_segment(tmp_path):
    from minime.testing import synth

    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    _truncate(io.list_segments(raw)[1])
    info = io.probe(raw)
    assert info.segment_frames == [100, 100, 100]  # middle headers are trusted by probe
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    grid = tiles.tile_grid(64, 64, 32)
    store = tiles.TileStore(tmp_path / "s", grid, pass1.n_parts_for(info, 2))
    with pytest.raises(ValueError, match=r"1\.avi: probed 100, decoded \d+"):
        pass1.run_pass1(info, store, pass1.build_template(info, p), p, workers=2)


def test_build_factors_uses_decoded_frame_count():
    from types import SimpleNamespace

    from minime import pass2

    T = 40
    info = SimpleNamespace(height=8, width=8, frames=50)  # stale probe count
    p1 = pass1.Pass1Result(
        np.zeros((T, 2), np.float32),
        np.zeros(T, np.float32),
        np.zeros((8, 8), np.uint8),
        np.zeros((1, 8, 8), np.uint8),
        np.array([0, T]),
        T,
        {},
    )
    assert pass2.build_factors(info, tiles.tile_grid(8, 8, 8), p1, [], Params()).T == T


def test_pass1_worker_streams_single_segment_in_batches(tmp_path):
    from minime.testing import synth

    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=300, seed=0, side=64)
    info = io.probe(raw)
    assert info.segment_frames == [300]
    seg = info.segments[0]
    batches = list(io.iter_avi(seg, batch=100))
    assert [b.shape for b in batches] == [(100, 64, 64)] * 3
    assert np.array_equal(np.concatenate(batches), io.read_avi(seg))
    assert [b.shape[0] for b in io.iter_avi(seg, batch=70)] == [70, 70, 70, 70, 20]
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50, sub_batch=30)
    tmpl = pass1.build_template(info, p)
    grid = tiles.tile_grid(64, 64, 32)
    edges = pass1.seed_block_edges(300, p.seeds_wnd, p.seeds_stp)
    out = {}
    for rb in (1000, 100, 70):  # one shot, aligned batches, batches straddling sub-batches
        store = tiles.TileStore(tmp_path / f"s{rb}", grid, 1)
        res = pass1.pass1_worker(
            [str(seg)], 0, str(store.root), grid, 1, tmpl, p, edges, 0, read_batch=rb
        )
        out[rb] = (res, [store.read_tile(t.index) for t in grid])
    ref, ref_tiles = out[1000]
    assert ref["frames"] == 300 and ref["segment_frames"] == [300]
    for rb in (100, 70):
        res, tl = out[rb]
        assert res["segment_frames"] == [300]
        assert np.array_equal(res["shifts"], ref["shifts"])
        assert np.array_equal(res["block_max"], ref["block_max"])
        assert np.array_equal(res["max_proj"], ref["max_proj"])
        # quality is a float32 correlation: batch boundaries change the summation order
        assert np.allclose(res["quality"], ref["quality"], atol=1e-6)
        assert all(np.array_equal(a, b) for a, b in zip(tl, ref_tiles, strict=True))


def test_run_params_json_roundtrips(tmp_path):
    from minime import pipeline

    p = Params(tile=32, pnr_thres=2.5, workers=3, size_thres=(9, None))
    pipeline._write_params(tmp_path, p, tmp_path, "abc", 19.5)
    doc = json.loads((tmp_path / "params.json").read_text(encoding="utf-8"))
    assert {"raw_dir", "raw_hash", "minime_version"} <= set(doc) and doc["fps"] == 19.5
    assert Params.from_json(tmp_path / "params.json") == p  # workers kept as recorded
    old = {**doc, "cafast_version": doc.pop("minime_version")}  # run written as cafast
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    assert Params.from_json(tmp_path / "old.json") == p
    (tmp_path / "bad.json").write_text(json.dumps({"tile": 32, "tiel": 16}), encoding="utf-8")
    with pytest.raises(ValueError, match="tiel"):
        Params.from_json(tmp_path / "bad.json")


def test_block_range():
    edges = np.array([0, 50, 100, 150])
    assert pass1._block_range(edges, 0, 150) == (0, 3)
    assert pass1._block_range(edges, 60, 40) == (1, 2)
    assert pass1._block_range(edges, 40, 20) == (0, 2)
    assert pass1._block_range(edges, 100, 0) == (2, 2)  # no frames: no blocks


def test_run_pass1_with_an_empty_last_segment_matches_one_worker(tmp_path):
    import cv2

    from minime.testing import synth

    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=150, seed=0, side=64)
    # a recording that stopped right after the DAQ opened the next file
    cv2.VideoWriter(str(raw / "2.avi"), cv2.VideoWriter_fourcc(*"GREY"), 30, (64, 64), False)
    info = io.probe(raw)
    assert info.segment_frames == [150, 150, 0]
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    tmpl = pass1.build_template(info, p)
    grid = tiles.tile_grid(64, 64, 32)
    res = {}
    for w in (1, 3):
        store = tiles.TileStore(tmp_path / f"s{w}", grid, pass1.n_parts_for(info, w))
        res[w] = pass1.run_pass1(info, store, tmpl, p, w)
    assert res[3].shifts.shape == (300, 2) and res[3].frames == 300
    assert np.array_equal(res[3].shifts, res[1].shifts)
    assert np.array_equal(res[3].block_max, res[1].block_max)


def test_run_workers_raises_when_a_worker_dies():
    import os
    from concurrent.futures.process import BrokenProcessPool

    with pytest.raises(BrokenProcessPool):
        pass1.run_workers(os._exit, [(1,)])
