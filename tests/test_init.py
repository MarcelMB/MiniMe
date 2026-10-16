import numpy as np
import pandas as pd
import scipy.sparse as sps

from fastminian import compress, corr, init, tiles
from fastminian import factors as fm
from fastminian.testing import synth


def _session(H=64, W=64, T=800, seed=0):
    movie = synth.synth_movie(H, W, T, seed)
    setup = synth.synth_setup(H, W, T, seed)
    # undo the synthetic rigid motion with the known shifts (stands in for pass-1 motion
    # correction); setup.C is in the unshifted frame, so traces only match a registered movie
    movie = np.stack(
        [np.roll(fr, tuple(s), axis=(0, 1)) for fr, s in zip(movie, setup.shifts, strict=True)]
    )
    grid = tiles.tile_grid(H, W, 32)
    tf = [
        compress.compress_tile(np.ascontiguousarray(movie[:, t.y0 : t.y1, t.x0 : t.x1]), k0=16)
        for t in grid
    ]
    return setup, fm.Factors(
        H, W, T, grid, tf, movie.max(0), movie.max(0)[None], np.array([0, T]), {}
    )


def test_initA_footprints_cover_cells_and_initC_is_least_squares():
    setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 5).all(1) & (cents < 59).all(1)]
    seeds = pd.DataFrame({"height": cents[:, 0], "width": cents[:, 1]})
    A = init.initA(fc, seeds, thres_corr=0.6, wnd=10)
    assert A.shape == (len(cents), 64 * 64)
    sizes = np.diff(A.indptr)
    assert sizes.min() >= 9 and sizes.max() <= 317
    C = init.initC(f, A)
    # reference: dense lstsq on the reconstructed movie
    Y = np.concatenate([f.reconstruct_chunk(b, 0, f.T) for b in range(4)])
    perm = np.concatenate([f.tile_rows(b) for b in range(4)])
    Yf = np.empty_like(Y)
    Yf[perm] = Y
    ref = np.linalg.lstsq(A.toarray().T, Yf, rcond=None)[0]
    assert np.allclose(C, ref, atol=1e-3 * abs(ref).max())
    # traces correlate with truth
    tr = setup.C[
        (setup.ys + 7 - 8 >= 5)
        & (setup.ys + 7 - 8 < 59)
        & (setup.xs + 7 - 8 >= 5)
        & (setup.xs + 7 - 8 < 59)
    ]
    r = [np.corrcoef(C[k], tr[k])[0, 1] for k in range(len(tr))]
    assert np.median(r) > 0.8


def test_unit_merge_means_duplicates():
    A = sps.csr_matrix(np.array([[1, 1, 0, 0], [1, 0.5, 0, 0], [0, 0, 1, 0]], np.float32))
    C = np.array([[1, 2, 3, 4], [1.1, 2, 3, 3.9], [4, 3, 2, 1]], np.float32)
    A2, C2, extra = init.unit_merge(A, C, lambda X: np.corrcoef(X), thres_corr=0.9, extra=[C * 2])
    assert A2.shape == (2, 4) and np.allclose(A2.toarray()[0], [1, 0.75, 0, 0])
    assert np.allclose(C2[0], (C[0] + C[1]) / 2) and np.allclose(extra[0][1], C[2] * 2)


def test_initA_raises_without_footprints():
    import pytest

    setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    with pytest.raises(ValueError, match="no footprints"):
        init.initA(fc, pd.DataFrame({"height": [0], "width": [0]}), thres_corr=1.01, wnd=3)


def test_centroids_are_intensity_weighted():
    A = sps.csr_matrix(np.array([[0, 1, 0, 1, 0, 0], [0, 0, 3, 0, 0, 1]], np.float32))
    cents = init.centroids(A, 2, 3)
    assert cents.dtype == np.float64
    assert np.allclose(cents, [[0.5, 0.5], [0.25, 2.0]])
