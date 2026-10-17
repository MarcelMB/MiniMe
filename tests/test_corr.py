import numpy as np
import pandas as pd
import scipy.sparse as sps

from cafast import compress, corr, tiles
from cafast import factors as fm
from cafast.testing import synth


def _session(H=64, W=64, T=800, seed=0):
    movie = synth.synth_movie(H, W, T, seed)
    setup = synth.synth_setup(H, W, T, seed)
    grid = tiles.tile_grid(H, W, 32)
    tf = [
        compress.compress_tile(np.ascontiguousarray(movie[:, t.y0 : t.y1, t.x0 : t.x1]), k0=16)
        for t in grid
    ]
    f = fm.Factors(H, W, T, grid, tf, movie.max(0), movie.max(0)[None], np.array([0, T]), {})
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, H, W)])
    return movie, clean, setup, f


def test_lowpass_removes_dc_and_high_band():
    T = 500
    x = np.ones((2, T))
    # 2.5 rad/sample (brief) is not periodic in T=500 and leaks ~0.02 into the passband of a
    # brick-wall FFT mask; use an on-bin tone (bin 200, far above nr = 30) instead
    x[1] = np.sin(2 * np.pi * 200 * np.arange(T) / T)
    y = corr.lowpass(x, 0.06)
    assert abs(y).max() < 1e-6


def test_pair_correlations_track_noise_free_truth():
    movie, clean, setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 2).all(1) & (cents < 62).all(1)][:40]
    src = cents[:, 0] * 64 + cents[:, 1]
    dst = (cents[:, 0] + 1) * 64 + cents[:, 1]  # neighbour pixel of the same cell: high corr
    far = ((cents[:, 0] + 20) % 64) * 64 + ((cents[:, 1] + 20) % 64)
    c_near, c_far = fc.pairs(src, dst), fc.pairs(src, far)
    Y = clean.reshape(clean.shape[0], -1).T.astype(float)
    lp = corr.lowpass(Y[np.r_[src, dst, far]], 0.06)
    n = len(src)

    def pear(a, b):
        return np.sum(a * b, 1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))

    ref_near = pear(lp[:n], lp[n : 2 * n])
    assert np.median(np.abs(c_near - ref_near)) < 0.05
    assert np.median(c_near) > 0.8 and np.median(np.abs(c_far)) < 0.3
    idx, cw = fc.window(int(src[0]), 10)
    assert cw.max() <= 1.0 and idx.size <= 317 and (cw[idx == src[0]] > 0.99).all()


def test_seeds_merge_keeps_brightest_per_group():
    movie, clean, setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 3).all(1) & (cents < 61).all(1)][:10]
    dup = cents + np.array([1, 0])
    s = pd.DataFrame(
        {"height": np.r_[cents[:, 0], dup[:, 0]], "width": np.r_[cents[:, 1], dup[:, 1]]}
    )
    out = corr.seeds_merge(fc, s, f.max_proj, thres_dist=5, thres_corr=0.6)
    assert out["mask_mrg"].sum() <= len(cents) + 1
    kept = out[out["mask_mrg"]]
    # every kept seed is at least as bright as any seed it absorbed within thres_dist
    for h, w in zip(kept.height, kept.width, strict=True):
        near = (abs(s.height - h) <= 5) & (abs(s.width - w) <= 5)
        assert f.max_proj[h, w] >= f.max_proj[s.height[near], s.width[near]].max()


# ---- extra checks beyond the brief ---------------------------------------------------


def test_window_is_clipped_disk_and_matches_pairs():
    _, _, _, f = _session(T=400)
    fc = corr.FactorCorr(f, 0.06)
    seed = 1 * 64 + 2  # (1, 2): near the corner, disk is clipped
    idx, cw = fc.window(seed, 4)
    h, w = np.divmod(idx, 64)
    assert ((h - 1) ** 2 + (w - 2) ** 2 <= 16).all()
    full = [(y, x) for y in range(64) for x in range(64) if (y - 1) ** 2 + (x - 2) ** 2 <= 16]
    assert idx.size == len(full) and seed in idx
    np.testing.assert_allclose(cw, fc.pairs(np.full(idx.size, seed), idx))


def test_traces_corr_and_merge_labels():
    rng = np.random.default_rng(0)
    T = 600
    fc = corr.FactorCorr(_session(T=T)[3], 0.06)
    base = corr.lowpass(rng.standard_normal((1, T)), 0.1)[0] * 10
    C = np.stack([base, base * 2 + 0.01 * rng.standard_normal(T), rng.standard_normal(T)])
    C = np.vstack([C, np.zeros(T)])  # zero-variance row
    R = fc.traces_corr(C)
    assert R.shape == (4, 4) and np.isfinite(R).all()
    assert R[0, 1] > 0.99 and abs(R[0, 2]) < 0.3
    assert (R[3, :3] == 0).all() and (R[:3, 3] == 0).all()
    adj = sps.csr_matrix(R > 0.9)
    lab = corr.merge_labels(adj)
    assert lab[0] == lab[1] and len({lab[0], lab[2], lab[3]}) == 3


def _reference_corr(f, noise_freq):
    """Independent reference: reconstruct the centred traces, low-pass them, and normalise
    by the low-passed norm plus the passband share of the discarded (white) energy."""
    P, T = f.P, f.T
    Yr = np.empty((P, T))
    cap = np.zeros(P)
    for b, t in enumerate(f.tiles):
        Wd = t.W.astype(np.float64)
        Yr[f.tile_rows(b)] = Wd @ t.V.T.astype(np.float64)
        cap[f.tile_rows(b)] = (Wd**2).sum(1)
    Ylp = corr.lowpass(Yr, noise_freq)
    f_lp = 2 * (int(noise_freq * T) - 1) / T
    n2 = (Ylp**2).sum(1) + f_lp * np.maximum(f.energy_flat() - cap, 0)

    def ref(s, d):
        r = np.einsum("ij,ij->i", Ylp[s], Ylp[d]) / np.sqrt(n2[s] * n2[d])
        r = np.clip(r, -1, 1)
        r[s == d] = 1.0
        return r

    return ref


def test_cross_tile_correlations_match_reconstruct_then_lowpass():
    _, _, _, f = _session(T=400)  # 64 x 64, 32-px tiles: (31, 31) is a corner of 4 tiles
    fc = corr.FactorCorr(f, 0.06)
    ref = _reference_corr(f, 0.06)
    seed = 31 * 64 + 31
    idx, cw = fc.window(seed, 6)
    assert np.unique(f.tile_of[idx]).tolist() == [0, 1, 2, 3]
    assert np.abs(cw - ref(np.full(idx.size, seed), idx)).max() < 1e-10
    rng = np.random.default_rng(1)
    s, d = rng.integers(0, f.P, 3000), rng.integers(0, f.P, 3000)
    assert np.mean(f.tile_of[s] != f.tile_of[d]) > 0.5
    c = fc.pairs(s, d)
    assert np.abs(c - ref(s, d)).max() < 1e-10
    assert np.abs(c - fc.pairs(d, s)).max() < 1e-12  # symmetric across tiles
