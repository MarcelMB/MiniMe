import json

import numpy as np
import scipy.sparse as sps

from caanfast import compress, pass2, tiles
from caanfast import factors as fm
from caanfast.params import Params
from caanfast.pass1 import Pass1Result
from caanfast.testing import synth


def _tile_movie(T=400, side=32, seed=3):
    return synth.synth_movie(side, side, T, seed)  # moving; fine for a compression test


def test_compress_tile_keeps_signal_and_bounds_rank():
    raw = _tile_movie()
    tf = compress.compress_tile(raw, k0=8, n_iter=1, seed=0)
    T, P = raw.shape[0], 32 * 32
    assert (
        tf.W.shape == (P, tf.rank)
        and tf.V.shape == (T, tf.rank)
        and 1 <= tf.rank <= compress.MAX_RANK
    )
    assert tf.energy.shape == (P,) and tf.energy.dtype == np.float64
    assert tf.W.dtype == tf.V.dtype == tf.mean.dtype == tf.sigma.dtype == np.float32
    Y = raw.reshape(T, -1).T.astype(np.float32)
    rec = tf.W @ tf.V.T + tf.mean[:, None]
    assert np.linalg.norm(Y - rec) / np.linalg.norm(Y - Y.mean(1, keepdims=True)) < 0.8
    assert np.allclose(tf.energy, ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1), rtol=1e-4)


def test_compress_tile_pure_noise_returns_rank_one():
    rng = np.random.default_rng(0)
    raw = np.clip(50 + 4 * rng.standard_normal((300, 16, 16)), 0, 255).astype(np.uint8)
    tf = compress.compress_tile(raw, k0=8)
    assert tf.rank == 1


def test_compress_tile_single_pixel_tile():
    rng = np.random.default_rng(0)
    raw = np.clip(50 + 4 * rng.standard_normal((200, 1, 1)), 0, 255).astype(np.uint8)
    tf = compress.compress_tile(raw, k0=8)
    assert tf.rank == 1 and tf.W.shape == (1, 1) and tf.V.shape == (200, 1)
    assert tf.mean.shape == tf.sigma.shape == tf.energy.shape == (1,)


def _minian_noise_fft(y: np.ndarray, noise_range: tuple[float, float]) -> np.ndarray:
    """Minian's ``noise_fft(..., noise_method="logmexp")`` per column, in float64."""
    T = len(y)
    nr = np.around(np.array(noise_range) * T).astype(int)
    px = np.abs(np.fft.rfft(y, axis=0)[nr[0] : nr[1]]) ** 2 / T
    return np.sqrt(np.exp(np.log(px + np.finfo(np.float32).eps).mean(axis=0)))


def test_noise_matches_minian_fft_and_is_not_quantized():
    rng = np.random.default_rng(1)
    T, sd = 2001, 0.8
    raw = np.clip(np.rint(2 + sd * rng.standard_normal((T, 300))), 0, 255).astype(np.uint8)
    sn = compress.TileMatrix(raw, chunk=512).noise(0.03)
    assert np.allclose(sn, _minian_noise_fft(raw.astype(np.float64), (0.03, 0.5)), rtol=1e-4)
    # rounded white noise: the old median |diff| took 2-3 values here, 0 for many pixels
    assert sn.min() > 0 and np.unique(sn).size == raw.shape[1]
    sd_q = np.sqrt(sd**2 + 1 / 12)  # Gaussian plus rounding noise
    assert abs(np.median(sn) / (np.exp(-np.euler_gamma / 2) * sd_q) - 1) < 0.05


def test_autocorr_keep():
    T = 1000
    rng = np.random.default_rng(0)
    smooth = np.cumsum(rng.standard_normal(T))
    white = rng.standard_normal(T)
    keep = compress.autocorr_keep(np.stack([smooth, white]), z=4.0, rho_min=0.2)
    assert keep.tolist() == [True, False]


def test_factors_save_load_and_algebra(tmp_path):
    H = W = 32
    T = 200
    grid = tiles.tile_grid(H, W, 16)
    movie = synth.synth_movie(H, W, T, 1)
    tf = [
        compress.compress_tile(np.ascontiguousarray(movie[:, t.y0 : t.y1, t.x0 : t.x1]), k0=8)
        for t in grid
    ]
    f = fm.Factors(
        H, W, T, grid, tf, movie.max(0), movie[:100].max(0)[None], np.array([0, 100, 200]), {}
    )
    f.save(tmp_path / "factors")
    g = fm.Factors.load(tmp_path / "factors")
    assert g.T == T and len(g.tiles) == 4 and np.array_equal(g.tiles[2].V, tf[2].V)
    # dense reference
    Yd = np.concatenate([(t.W @ t.V.T + t.mean[:, None]) for t in tf])  # tile-major rows
    perm = np.concatenate([g.tile_rows(b) for b in range(4)])
    Yfull = np.empty((H * W, T), np.float32)
    Yfull[perm] = Yd
    A = sps.random(5, H * W, density=0.05, format="csr", random_state=0, dtype=np.float32)
    assert np.allclose(g.AtY(A), A @ Yfull, rtol=1e-4, atol=1e-2)
    C = np.random.default_rng(0).standard_normal((5, T)).astype(np.float32)
    Cc = C - C.mean(1, keepdims=True)
    assert np.allclose(g.YCt(Cc), Yfull @ Cc.T, rtol=1e-3, atol=1e-1)
    px = np.array([0, 17, 1000])
    assert np.allclose(g.traces(px), Yfull[px], atol=1e-3)
    assert np.allclose(g.reconstruct_chunk(1, 10, 20), Yd[256:512, 10:20], atol=1e-3)


def test_factors_pixel_maps_and_flat_stats():
    H, W, T = 20, 36, 120
    grid = tiles.tile_grid(H, W, 16)  # ragged edge tiles
    movie = synth.synth_movie(H, W, T, 2)
    tf = [
        compress.compress_tile(np.ascontiguousarray(movie[:, t.y0 : t.y1, t.x0 : t.x1]), k0=6)
        for t in grid
    ]
    f = fm.Factors(H, W, T, grid, tf, movie.max(0), movie[:60].max(0)[None], np.array([0, T]), {})
    assert f.P == H * W and f.tile_of.dtype == f.loc_of.dtype == np.int32
    for b in range(len(grid)):
        rows = f.tile_rows(b)
        assert np.array_equal(f.loc_of[rows], np.arange(len(rows)))
        assert np.all(f.tile_of[rows] == b)
    p = f.pixel_index(17, 35)
    assert p == 17 * W + 35 and grid[f.tile_of[p]].x0 <= 35 < grid[f.tile_of[p]].x1
    mean = f.mean_flat()
    assert mean.shape == (H * W,) and np.allclose(mean, movie.reshape(T, -1).mean(0), atol=1e-3)
    assert f.sigma_flat().shape == f.energy_flat().shape == (H * W,)


def test_run_pass2_parallel_matches_serial_and_builds_factors(tmp_path):
    H = W = 32
    T = 120
    grid = tiles.tile_grid(H, W, 16)
    movie = synth.synth_movie(H, W, T, 4)
    store = tiles.TileStore(tmp_path / "s", grid, n_parts=1)
    packed = tiles.pack_tiles(movie, grid)
    for f, (a, b) in zip(store.open_part(0), tiles.tile_offsets(T, grid), strict=True):
        with f:
            f.write(memoryview(packed[a:b]))
    p = Params(svd_k0=6, tile=16)
    one = pass2.run_pass2(store, grid, p, workers=1)
    two = pass2.run_pass2(store, grid, p, workers=2)
    assert len(one) == len(two) == 4
    for a, b in zip(one, two, strict=True):
        assert np.allclose(a.W @ a.V.T, b.W @ b.V.T, atol=1e-3)
        assert np.array_equal(a.mean, b.mean)
    info = type("Info", (), {"height": H, "width": W, "frames": T})()
    p1 = Pass1Result(
        np.zeros((T, 2), np.float32),
        np.zeros(T, np.float32),
        movie.max(0),
        movie[None].max(1),
        np.array([0, T]),
        T,
        {"denoise": 1.0},
    )
    fac = pass2.build_factors(info, grid, p1, two, p)
    assert fac.H == H and fac.W_ == W and fac.T == T and fac.meta["params"]["tile"] == 16
    assert np.array_equal(fac.block_edges, [0, T])
    fac.save(tmp_path / "factors")
    back = fm.Factors.load(tmp_path / "factors")
    expected = json.loads(json.dumps(p.asdict()))  # tuples become lists in JSON
    assert back.meta["params"] == expected and back.meta["params"]["size_thres"] == [25, None]
    assert back.meta["pass1_seconds"] == {"denoise": 1.0} and len(back.tiles) == 4


def _truth_metrics(Y_truth, S, grid, tfs):
    """Spec section 7 metrics (prototype ``bench_opt._truth_group``), dense.

    ``rel_err = ||W V^T - (Y0 - mean Y0)|| / ||Y0 - mean Y0||`` over all tiles, with ``Y0``
    the noise-free movie through the same preprocessing and shifts; ``capture`` = fraction
    of the (per-pixel centred) true cell variance that lies in span(V) of its tile.
    """
    T = Y_truth.shape[0]
    err2 = y2 = cap = s2 = 0.0
    for t, tf in zip(grid, tfs, strict=True):
        sl = (slice(None), slice(t.y0, t.y1), slice(t.x0, t.x1))
        Yc = Y_truth[sl].reshape(T, -1).T.astype(np.float64)
        Yc -= Yc.mean(axis=1, keepdims=True)
        Sc = S[sl].reshape(T, -1).T.astype(np.float64)
        Sc -= Sc.mean(axis=1, keepdims=True)
        rec = tf.W.astype(np.float64) @ tf.V.T.astype(np.float64)
        err2 += ((rec - Yc) ** 2).sum()
        y2 += (Yc**2).sum()
        V = tf.V.astype(np.float64)
        proj = Sc @ V
        cap += np.trace(proj @ np.linalg.solve(V.T @ V, proj.T))
        s2 += (Sc**2).sum()
    return float(np.sqrt(err2 / y2)), float(cap / s2)


def test_compression_gate_against_noise_free_truth(tmp_path):
    """Spec section 7 gate on pass-1 preprocessing + pass-2 compression: rel err <= 0.30
    against the noise-free movie and cell capture >= 0.97 (prototype, 600 x 600 x 9000:
    0.145 / 0.991). Known shifts are undone on both the noisy and the noise-free movie,
    so this measures compression only. Measured here: 0.217 / 0.981."""
    from caanfast import motion, preprocess

    H = W = 128
    T = 1000
    p = Params()
    setup = synth.synth_setup(H, W, T, 0)
    undo = -setup.shifts.astype(np.float64)  # apply_shifts moves content by -(dy, dx)
    movie = synth.synth_movie(H, W, T, 0)
    Y = motion.apply_shifts(preprocess.preprocess(movie, p.median_ksize, p.tophat_wnd), undo)
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, H, W)])
    cells = np.concatenate([s for _, _, s in synth.truth_chunks(setup, H, W)])
    Y0 = motion.apply_shifts(preprocess.preprocess(clean, p.median_ksize, p.tophat_wnd), undo)
    S = motion.apply_shifts(cells, undo)
    grid = tiles.tile_grid(H, W, p.tile)
    store = tiles.TileStore(tmp_path / "s", grid, n_parts=1)
    packed = tiles.pack_tiles(Y, grid)
    for f, (a, b) in zip(store.open_part(0), tiles.tile_offsets(T, grid), strict=True):
        with f:
            f.write(memoryview(packed[a:b]))
    tfs = pass2.pass2_worker(str(store.root), grid, 1, range(len(grid)), p)
    rel_err, capture = _truth_metrics(Y0, S, grid, tfs)
    assert rel_err <= 0.30, rel_err
    assert capture >= 0.97, capture


def test_noise_frame_subset_close_to_full_and_falls_back():
    rng = np.random.default_rng(2)
    T = 12000
    raw = np.clip(np.rint(60 + 3 * rng.standard_normal((T, 200))), 0, 255).astype(np.uint8)
    Y = compress.TileMatrix(raw)
    full = Y.noise(0.06)
    sub = Y.noise(0.06, frames=4000)  # 4 segments of 1000 frames
    assert np.median(np.abs(sub / full - 1)) < 0.03
    assert np.array_equal(Y.noise(0.06, frames=T), full)  # a subset of all frames = all frames
    assert np.array_equal(Y.noise(0.06, frames=500), full)  # shorter than one segment


def test_energy_one_pass_matches_centred_sum():
    raw = _tile_movie(T=500)
    Y = compress.TileMatrix(raw, chunk=128)
    X = raw.reshape(len(raw), -1).astype(np.float64)
    assert np.allclose(Y.energy, ((X - X.mean(0)) ** 2).sum(0), rtol=1e-9)
    assert np.allclose(Y.mean, X.mean(0), rtol=1e-6)
