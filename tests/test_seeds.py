import numpy as np
import pandas as pd
import pytest

from minime import compress, tiles
from minime import factors as fm
from minime import seeds as sd
from minime.gmm import fit_gmm_1d
from minime.pass1 import seed_block_edges
from minime.testing import synth


def _session(H=64, W=64, T=600, seed=0):
    movie = synth.synth_movie(H, W, T, seed)
    grid = tiles.tile_grid(H, W, 32)
    tf = [
        compress.compress_tile(np.ascontiguousarray(movie[:, t.y0 : t.y1, t.x0 : t.x1]), k0=16)
        for t in grid
    ]
    edges = seed_block_edges(T, 300, 150)  # the same edges pass 1 would store
    bm = np.stack([movie[a:b].max(0) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    return movie, fm.Factors(H, W, T, grid, tf, movie.max(0), bm, edges, {})


def test_seeds_from_block_max_equal_full_projection():
    movie, f = _session()
    s = sd.seeds_init(
        f.block_max, f.block_edges, f.T, wnd_size=300, stp_size=150, max_wnd=10, diff_thres=2
    )
    wins = sd.rolling_windows(f.T, 300, 150)
    ref = sum(sd.local_max_roll(movie[w.start : w.stop].max(0), 2, 10, 2) for w in wins)
    hh, ww = np.nonzero(ref)
    assert set(zip(s.height, s.width, strict=True)) == set(zip(hh, ww, strict=True))
    assert len(s) > 5
    assert list(s.columns) == ["height", "width", "seeds"]
    assert (s["seeds"].to_numpy() == ref[hh, ww]).all()
    assert s.equals(s.sort_values(["height", "width"]).reset_index(drop=True))


def test_pnr_and_gmm_masks():
    movie, f = _session()
    setup = synth.synth_setup(64, 64, 600, 0)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 0).all(1) & (cents < 64).all(1)]
    rng = np.random.default_rng(0)
    noise = np.stack([rng.integers(0, 64, 30), rng.integers(0, 64, 30)], 1)
    seeds = pd.DataFrame(
        {"height": np.r_[cents[:, 0], noise[:, 0]], "width": np.r_[cents[:, 1], noise[:, 1]]}
    )
    out = sd.pnr_refine(f, seeds, thres="auto")
    assert out["mask_pnr"].iloc[: len(cents)].mean() > 0.8
    out2 = sd.pnr_refine(f, seeds, thres=2.0)
    assert out2["pnr"].iloc[: len(cents)].median() > out2["pnr"].iloc[len(cents) :].median()
    g = sd.gmm_refine(f, seeds)
    assert g["mask_gmm"].iloc[: len(cents)].mean() > 0.8


def test_pnr_raises_when_nothing_survives():
    _, f = _session()
    seeds = pd.DataFrame({"height": [1, 2], "width": [1, 2]})
    with pytest.raises(ValueError, match="no seeds survived pnr_refine"):
        sd.pnr_refine(f, seeds, thres=1e9)


def test_seed_pixels_and_ptp_q():
    _, f = _session()
    seeds = pd.DataFrame({"height": [3, 10], "width": [5, 7]})
    px = sd.seed_pixels(f, seeds)
    assert px.dtype == np.int64
    assert px.tolist() == [3 * f.W_ + 5, 10 * f.W_ + 7]
    x = np.arange(101, dtype=np.float32)[None].repeat(2, 0)
    assert np.allclose(sd.ptp_q(x, (10, 90)), [80.0, 80.0])


def test_label_medians_match_np_median():
    rng = np.random.default_rng(1)
    mask = (rng.random((40, 40)) > 0.7).astype(np.uint8)
    import cv2

    nlab, lab = cv2.connectedComponents(mask)
    r, c = sd.label_medians(lab, nlab)
    for lb in range(1, nlab):
        crd = np.where(lab == lb)
        assert (r[lb - 1], c[lb - 1]) == tuple(int(np.median(x)) for x in crd)


def test_gmm_mean_mask_uses_lowest_component_mean():
    _, f = _session()
    setup = synth.synth_setup(64, 64, 600, 0)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 0).all(1) & (cents < 64).all(1)]
    rng = np.random.default_rng(0)
    noise = np.stack([rng.integers(0, 64, 30), rng.integers(0, 64, 30)], 1)
    seeds = pd.DataFrame(
        {"height": np.r_[cents[:, 0], noise[:, 0]], "width": np.r_[cents[:, 1], noise[:, 1]]}
    )
    ptp = sd.ptp_q(f.traces(sd.seed_pixels(f, seeds)), (0.1, 99.9))
    low_mean = fit_gmm_1d(ptp, 2).means.min()
    on = sd.gmm_refine(f, seeds, mean_mask=True)["mask_gmm"].to_numpy()
    off = sd.gmm_refine(f, seeds, mean_mask=False)["mask_gmm"].to_numpy()
    assert on.any()
    assert (ptp[on] > low_mean).all()
    assert (off | ~on).all()  # mean_mask=False keeps a superset


def test_seed_ptp_batched_equals_all_at_once():
    _, f = _session()
    rng = np.random.default_rng(1)
    pix = rng.integers(0, f.P, 37)
    ref = sd.ptp_q(f.traces(pix), (0.1, 99.9))
    # float32 traces differ in the last bits with the batch composition (BLAS blocking)
    assert np.allclose(sd.seed_ptp(f, pix, (0.1, 99.9), batch=5), ref, rtol=1e-6, atol=0)
    assert np.allclose(sd.seed_ptp(f, pix, (0.1, 99.9)), ref, rtol=1e-6, atol=0)


def test_seed_ptp_threads_equal_serial():
    _, f = _session()
    pix = np.random.default_rng(2).integers(0, f.P, 50)
    one = sd.seed_ptp(f, pix, (0.1, 99.9), batch=7, threads=1)
    assert np.array_equal(sd.seed_ptp(f, pix, (0.1, 99.9), batch=7, threads=4), one)
