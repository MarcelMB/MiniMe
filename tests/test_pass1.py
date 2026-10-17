import numpy as np
import pytest

from fastminian import io, pass1, tiles
from fastminian.params import Params


def test_block_edges_cover_movie():
    e = pass1.seed_block_edges(300, wnd_size=100, stp_size=50)
    assert e[0] == 0 and e[-1] == 300 and np.all(np.diff(e) > 0)


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
    assert set(r1.seconds) == {
        "read_avi",
        "denoise",
        "tophat",
        "motion_estimate",
        "motion_apply",
        "max_proj",
        "block_max",
        "pack",
        "write_tiles",
    }
    for t in grid:
        assert np.array_equal(s1.read_tile(t.index), s2.read_tile(t.index))
    # block maxima equal the max over the same corrected frames
    full = np.concatenate([s1.read_tile(t.index) for t in grid[:1]])
    e = r1.block_edges
    for bi in range(len(e) - 1):
        assert np.array_equal(r2.block_max[bi, :32, :32], full[e[bi] : e[bi + 1]].max(0))
    assert np.array_equal(r1.max_proj[:32, :32], full.max(0))


def test_pass1_ragged_last_segment(tmp_path):
    from fastminian.testing import synth

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
    assert r_app.seconds["motion_estimate"] < 0.1 * r_est.seconds["motion_estimate"]
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
