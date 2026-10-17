import numpy as np

from cafast import io, tiles


def test_segments_and_probe(tiny_session):
    segs = io.list_segments(tiny_session["raw_dir"])
    assert [s.stem for s in segs] == ["0", "1", "2"]
    info = io.probe(tiny_session["raw_dir"])
    assert (info.height, info.width, info.frames) == (64, 64, 300)
    assert info.segment_frames == [100, 100, 100] and info.fps == 30


def test_read_avi_roundtrip(tiny_session):
    from cafast.testing import synth

    frames = io.read_avi(io.list_segments(tiny_session["raw_dir"])[0])
    assert frames.dtype == np.uint8 and frames.shape == (100, 64, 64)
    assert np.array_equal(frames, synth.synth_movie(64, 64, 300, 0)[:100])


def test_hash_changes_with_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(bytes(range(256)) * 100)
    b.write_bytes(bytes(range(256)) * 99 + bytes(255 - i for i in range(256)))
    assert io.hash_files([a]) == io.hash_files([a]) != io.hash_files([b])


def test_pack_and_store_roundtrip(tmp_path):
    grid = tiles.tile_grid(70, 100, tile=32)  # ragged edge tiles
    assert len(grid) == 3 * 4 and grid[-1].shape == (6, 4)
    rng = np.random.default_rng(0)
    parts = [
        rng.integers(0, 255, (7, 70, 100), dtype=np.uint8),
        rng.integers(0, 255, (5, 70, 100), dtype=np.uint8),
    ]
    store = tiles.TileStore(tmp_path / "tiles", grid, n_parts=2)
    for p, frames in enumerate(parts):
        packed = tiles.pack_tiles(frames, grid)
        files = store.open_part(p)
        for f, (a, b) in zip(files, tiles.tile_offsets(frames.shape[0], grid), strict=True):
            f.write(memoryview(packed[a:b]))
            f.close()
    full = np.concatenate(parts)
    for t in grid:
        assert np.array_equal(store.read_tile(t.index), full[:, t.y0 : t.y1, t.x0 : t.x1])
    assert store.nbytes() == full.size
    store.remove()
    assert not (tmp_path / "tiles").exists()


def test_read_avi_max_frames_and_wrong_frame_count(tiny_session, monkeypatch):
    seg = io.list_segments(tiny_session["raw_dir"])[0]
    full = io.read_avi(seg)
    assert np.array_equal(io.read_avi(seg, max_frames=7), full[:7])
    for wrong in (10, 500):  # container frame count too low / too high
        monkeypatch.setattr(io, "frame_count", lambda p, n=wrong: n)
        assert np.array_equal(io.read_avi(seg), full)
        assert np.array_equal(io.read_avi(seg, max_frames=30), full[:30])
