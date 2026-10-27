import numpy as np

from minime import io, tiles


def test_segments_and_probe(tiny_session):
    segs = io.list_segments(tiny_session["raw_dir"])
    assert [s.stem for s in segs] == ["0", "1", "2"]
    info = io.probe(tiny_session["raw_dir"])
    assert (info.height, info.width, info.frames) == (64, 64, 300)
    assert info.segment_frames == [100, 100, 100] and info.fps == 30


def test_read_avi_roundtrip(tiny_session):
    from minime.testing import synth

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


def _write_avi(path, frames, fourcc, color):
    import cv2

    h, w = frames.shape[1:]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 30, (w, h), color)
    if not wr.isOpened():
        return False
    for f in frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if color else f)
    wr.release()
    return True


def test_needs_bgr_decision():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    assert not io.needs_bgr(f, f.copy())
    rounding = np.where(rng.random(f.shape) < 0.1, 1, 0)  # +-1 codec rounding on 10 % of px
    near = np.clip(f.astype(int) + rounding, 0, 255).astype(np.uint8)
    assert not io.needs_bgr(f, near)
    one_px = f.copy()
    one_px[0, 0] = f[0, 0] ^ 0x80  # a single pixel off by 128
    assert io.needs_bgr(f, one_px)
    limited = (16 + f.astype(np.float32) * 219 / 255).astype(np.uint8)  # TV-range Y plane
    assert io.needs_bgr(limited, f)
    assert io.needs_bgr(f[:, :16], f)  # shape mismatch


def test_probe_uses_bgr_path_for_codec_with_wrong_raw_decode(tmp_path):
    import pytest

    from minime import pass1, tiles
    from minime.params import Params
    from minime.testing import synth

    movie = synth.synth_movie(64, 64, 120, 0)
    grey, hfyu = tmp_path / "grey" / "raw", tmp_path / "hfyu" / "raw"
    for d, cc, color in ((grey, "GREY", False), (hfyu, "HFYU", True)):
        d.mkdir(parents=True)
        for k in range(2):
            if not _write_avi(d / f"{k}.avi", movie[60 * k : 60 * (k + 1)], cc, color):
                pytest.skip(f"cv2.VideoWriter cannot write {cc} here")
    raw0 = io.read_avi(hfyu / "0.avi")  # raw decode of bgr0 HuffYUV: right shape, wrong pixels
    assert raw0.shape == (60, 64, 64) and np.abs(raw0.astype(int) - movie[:60]).mean() > 10
    gi, hi = io.probe(grey), io.probe(hfyu)
    assert gi.bgr is False and hi.bgr is True and hi.frames == 120
    assert np.array_equal(io.read_avi(hfyu / "0.avi", bgr=True), movie[:60])  # lossless
    p = Params(seeds_wnd=60, seeds_stp=30, template_frames=50)
    grid = tiles.tile_grid(64, 64, 32)
    runs = []
    for name, info in (("g", gi), ("h", hi)):
        store = tiles.TileStore(tmp_path / name, grid, pass1.n_parts_for(info, 2))
        tmpl = pass1.build_template(info, p)
        runs.append((pass1.run_pass1(info, store, tmpl, p, workers=2), store))
    (rg, sg), (rh, sh) = runs
    assert np.array_equal(rg.shifts, rh.shifts) and np.array_equal(rg.block_max, rh.block_max)
    for t in grid:
        assert np.array_equal(sg.read_tile(t.index), sh.read_tile(t.index))


def test_probe_falls_back_to_bgr_when_raw_decode_fails(tmp_path):
    """Fiji/ImageJ AVIs are palette (pal8) rawvideo; OpenCV's raw decode returns no frame."""
    import shutil
    import subprocess

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    rng = np.random.default_rng(0)
    movie = rng.integers(0, 256, (20, 32, 32), dtype=np.uint8)
    raw = tmp_path / "raw"
    raw.mkdir()
    avi = raw / "0.avi"
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "32x32"]
    cmd += ["-i", "-", "-c:v", "rawvideo", "-pix_fmt", "pal8", str(avi)]
    subprocess.run(cmd, input=movie.tobytes(), check=True)
    cmd = ["ffmpeg", "-v", "error", "-i", str(avi), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    ref = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8)
    info = io.probe(raw)
    assert info.bgr is True and info.frames == 20 and (info.height, info.width) == (32, 32)
    got = io.read_avi(avi, bgr=True).astype(int)
    assert np.abs(got - ref.reshape(20, 32, 32)).max() <= 1
