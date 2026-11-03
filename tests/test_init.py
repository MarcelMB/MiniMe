import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from minime import compress, corr, init, tiles
from minime import factors as fm
from minime.testing import synth


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


def test_unit_merge_transitive_chain_means_all_three():
    # u0-u1 and u1-u2 share a pixel, u0-u2 do not; all three traces correlate; u3 is apart
    A = sps.csr_matrix(
        np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], np.float32)
    )
    base = np.array([1, 2, 3, 5, 4], np.float32)
    C = np.stack([base, base + 0.1, 2 * base, base[::-1]])
    E = np.arange(8, dtype=np.float32).reshape(4, 2)
    A2, C2, extra = init.unit_merge(A, C, lambda X: np.corrcoef(X), thres_corr=0.9, extra=[E])
    assert A2.shape == (2, 5) and C2.shape == (2, 5) and extra[0].shape == (2, 2)
    assert np.allclose(A2.toarray(), [[1 / 3, 2 / 3, 2 / 3, 1 / 3, 0], [0, 0, 0, 0, 1]])
    assert np.allclose(C2[0], C[:3].mean(0)) and np.allclose(C2[1], C[3])
    assert np.allclose(extra[0][0], E[:3].mean(0)) and np.allclose(extra[0][1], E[3])


def test_unit_merge_min_dist_merges_close_units_whatever_their_traces():
    # u0 and u1: centroids 2 px apart, uncorrelated traces; u2: 6 px away, correlated with u0
    H = W = 10
    img = np.zeros((3, H, W), np.float32)
    img[0, 5, 2] = img[1, 5, 4] = img[2, 5, 8] = 1.0
    A = sps.csr_matrix(img.reshape(3, -1))
    C = np.array([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3], [1, 2, 3, 4, 5]], np.float32)
    corr = lambda X: np.corrcoef(X)  # noqa: E731
    A_off, _, _ = init.unit_merge(A, C, corr, thres_corr=0.9)  # no shared pixel: nothing merges
    assert A_off.shape[0] == 3
    A_on, C_on, _ = init.unit_merge(A, C, corr, thres_corr=0.9, min_dist=3.0, shape=(H, W))
    assert A_on.shape[0] == 2 and np.allclose(C_on[0], C[:2].mean(0))
    with pytest.raises(ValueError, match="shape"):
        init.unit_merge(A, C, corr, thres_corr=0.9, min_dist=3.0)


def _exact_factors(H: int, W: int, Y: np.ndarray) -> fm.Factors:
    """Single-tile Factors whose reconstruction is exactly ``Y`` (P, T)."""
    mean = Y.mean(1)
    U, s, Vt = np.linalg.svd(Y - mean[:, None], full_matrices=False)
    T = Y.shape[1]
    tile = fm.TileFactors(
        (U * s).astype(np.float32),
        np.ascontiguousarray(Vt.T).astype(np.float32),
        mean.astype(np.float32),
        np.ones(H * W, np.float32),
        np.zeros(H * W),
    )
    grid = tiles.tile_grid(H, W, 32)
    mp = Y.max(1).reshape(H, W)
    return fm.Factors(H, W, T, grid, [tile], mp, mp[None], np.array([0, T]), {})


def _degenerate_case(copy_eps: float) -> tuple[fm.Factors, sps.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(1)
    H = W = 6
    A3 = np.zeros((3, H * W), np.float32)
    A3[0, 0:10], A3[1, 8:20], A3[2, 22:34] = 1.0, 0.7, 0.5
    A3[:, :] *= rng.uniform(0.5, 1.0, H * W).astype(np.float32)
    C3 = rng.uniform(0, 1, (3, 40))
    Y = A3.T.astype(np.float64) @ C3 + rng.normal(0, 0.01, (H * W, 40))
    f = _exact_factors(H, W, Y)
    dup = A3[0] * (1 + copy_eps * rng.standard_normal(H * W))
    A = sps.csr_matrix(np.vstack([A3, dup]).astype(np.float64))
    return f, A, init.initC(f, sps.csr_matrix(A3))


def test_initC_exact_duplicate_footprint_is_min_norm_lstsq():
    f, A, C_ok = _degenerate_case(0.0)
    G = (A @ A.T).toarray()
    assert np.linalg.cond(G) > init.COND_MAX  # forces the pinv path
    C = init.initC(f, A)
    Y = f.reconstruct_chunk(0, 0, f.T).astype(np.float64)
    ref = np.linalg.lstsq(A.toarray().T, Y, rcond=None)[0]
    assert np.allclose(C, ref, atol=1e-6 * abs(ref).max())
    assert np.abs(C).max() < 10 * np.abs(C_ok).max()


def test_initC_near_duplicate_footprint_stays_bounded():
    f, A, C_ok = _degenerate_case(1e-7)
    assert np.linalg.cond((A @ A.T).toarray()) > init.COND_MAX
    C = init.initC(f, A)
    assert np.isfinite(C).all() and np.abs(C).max() < 10 * np.abs(C_ok).max()
