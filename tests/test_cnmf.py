import warnings

import cv2
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from numba import njit, prange

from cafast import cnmf, compress, corr, init, tiles
from cafast import factors as fm
from cafast.params import Params
from cafast.testing import synth


def _dense(f):
    Y = np.concatenate([f.reconstruct_chunk(b, 0, f.T) for b in range(len(f.tiles))])
    perm = np.concatenate([f.tile_rows(b) for b in range(len(f.tiles))])
    out = np.empty_like(Y)
    out[perm] = Y
    return out.astype(np.float64)


def _undo_motion(movie, shifts):
    """Shift each frame back by its known (dy, dx): CNMF runs on motion-corrected data."""
    T, H, W = movie.shape
    out = np.empty_like(movie)
    for i, (dy, dx) in enumerate(shifts):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out[i] = cv2.warpAffine(movie[i], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return out


def _setup(H=48, W=48, T=600, tile=24):
    setup = synth.synth_setup(H, W, T, 2)
    movie = _undo_motion(synth.synth_movie(H, W, T, 2), setup.shifts)
    grid = tiles.tile_grid(H, W, tile)
    tf = [
        compress.compress_tile(np.ascontiguousarray(movie[:, t.y0 : t.y1, t.x0 : t.x1]), k0=16)
        for t in grid
    ]
    f = fm.Factors(H, W, T, grid, tf, movie.max(0), movie.max(0)[None], np.array([0, T]), {})
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 4).all(1) & (cents < H - 4).all(1)]
    A = init.initA(fc, pd.DataFrame({"height": cents[:, 0], "width": cents[:, 1]}), 0.6, 8)
    C = init.initC(f, A)
    return f, fc, A, C


def test_background_and_trace_match_dense_reference():
    f, fc, A, C = _setup()
    Y = _dense(f)
    Ad = A.toarray().astype(np.float64)
    R = np.clip(Y - Ad.T @ C, 0, None)
    b_ref = R.mean(1)
    f_ref = R.T @ b_ref / (b_ref @ b_ref)
    b, fb = cnmf.update_background(f, A, C, chunk=128)
    assert np.allclose(b, b_ref, atol=1e-4 * b_ref.max()) and np.allclose(fb, f_ref, rtol=1e-4)
    b2, fb2 = cnmf.update_background(f, A, C, b=b)
    assert np.array_equal(b2, b) and np.allclose(fb2, fb)
    YtA = (Y - np.outer(b_ref, f_ref)).T @ Ad.T
    An = 1.0 / (Ad**2).sum(1)
    ref = np.clip(C + ((YtA - C.T @ (Ad @ Ad.T)) * An).T, 0, None)
    assert np.allclose(cnmf.compute_trace(f, A, b, C, fb), ref, atol=1e-3 * abs(ref).max())


def test_background_uses_only_units_touching_each_tile():
    f, fc, A, C = _setup()
    Acsc = sps.csc_matrix(A)
    for t in range(len(f.tiles)):
        rows = f.tile_rows(t)
        touching = np.flatnonzero(np.abs(A.toarray()[:, rows]).sum(1) > 0)
        assert np.array_equal(cnmf.tile_units(Acsc, rows), touching)
    # a unit confined to tile 0 with a huge trace leaves the other tiles' background unchanged
    extra = sps.csr_matrix(([1.0], ([0], [f.tile_rows(0)[0]])), shape=(1, f.P), dtype=np.float32)
    A2 = sps.vstack([A, extra]).tocsr()
    C2 = np.vstack([C, np.full((1, f.T), 1e6, dtype=C.dtype)])
    b, _ = cnmf.update_background(f, A, C)
    b2, _ = cnmf.update_background(f, A2, C2)
    rest = np.concatenate([f.tile_rows(t) for t in range(1, len(f.tiles))])
    assert np.array_equal(b2[rest], b[rest]) and b2[f.tile_rows(0)[0]] == 0


@njit(parallel=True)
def _parallel_double(x):
    out = np.zeros(x.shape[0])
    for i in prange(x.shape[0]):
        out[i] = 2.0 * x[i]
    return out


def test_background_trace_correct_after_numba_parallel_region():
    """Canary, not a regression test: it cannot fail deterministically.

    Accelerate (macOS 26, M1 Max) was seen to return wrong float32 transposed-gemv results
    for output lengths ~513-575 right after a numba parallel region; the WL25 tail chunk
    (541 frames) turned f into NaN in 3 of 6 runs. The final review could not reproduce it
    (0 of 150 trials with the old ``R.T @ b`` code), so this test also passes on the old
    code. It keeps the failing shape (a 64 x 64 tile = 4096 pixels, a 541-frame chunk) in
    the suite in case the bug returns; the guards that do catch it at run time are the
    NaN checks in ``run_cnmf`` and ``f64_spot_check`` in ``update_background``.
    """
    f, fc, A, C = _setup(64, 64, 600, tile=64)
    Y = _dense(f)
    R = np.clip(Y - A.toarray().astype(np.float64).T @ C, 0, None)
    b_ref = R.mean(1)
    f_ref = R.T @ b_ref / (b_ref @ b_ref)
    for _ in range(20):  # intermittent when it occurred; passing proves nothing
        _parallel_double(np.ones(100_000))
        _, fb = cnmf.update_background(f, A, C, b=b_ref.astype(np.float32), chunk=541)
        assert np.isfinite(fb).all() and np.allclose(fb, f_ref, rtol=1e-4)


def test_dilate_footprints_disk():
    H = W = 15
    A = sps.csr_matrix(([1.0], ([0], [7 * W + 7])), shape=(1, H * W), dtype=np.float32)
    sub = cnmf.dilate_footprints(A, H, W, 3)
    yy, xx = np.mgrid[0:H, 0:W]
    disk = ((yy - 7) ** 2 + (xx - 7) ** 2 <= 9).ravel()
    assert sub.dtype == bool and sub.shape == (1, H * W)
    assert np.array_equal(sub.toarray()[0], disk)


def test_update_spatial_shrinks_outside_cells_and_keeps_units():
    f, fc, A, C = _setup()
    sn = f.sigma_flat()
    A2, mask, nf = cnmf.update_spatial(
        f, A, C, sn, dl_wnd=5, sparse_penal=0.01, size_thres=(9, None)
    )
    assert mask.sum() >= 0.8 * A.shape[0] and A2.shape[1] == f.P
    assert np.allclose(A2.max(axis=1).toarray().ravel(), 1.0)
    assert (A2.data >= 0).all()
    # support stays inside the dilated footprints of the kept units
    sub = cnmf.dilate_footprints(A, f.H, f.W_, 5)[np.flatnonzero(mask)]
    assert (A2.astype(bool) > sub).nnz == 0
    assert nf.shape == (A2.shape[0],) and (nf > 0).all()


def test_update_spatial_size_filter_is_strict():
    f, fc, A, C = _setup()
    sn = f.sigma_flat()
    A2, mask, _ = cnmf.update_spatial(f, A, C, sn, 5, 0.01, (None, None))
    nnz = np.diff(A2.indptr)
    lo = int(np.sort(nnz)[len(nnz) // 2])
    A3, mask3, _ = cnmf.update_spatial(f, A, C, sn, 5, 0.01, (lo, None))
    assert np.array_equal(mask3[mask], nnz > lo)
    A4, mask4, _ = cnmf.update_spatial(f, A, C, sn, 5, 0.01, (None, lo))
    assert np.array_equal(mask4[mask], nnz < lo)


def test_run_cnmf_two_iterations():
    f, fc, A, C = _setup()
    p = Params(dl_wnd=5, size_thres=(9, None), n_iter=2)
    res = cnmf.run_cnmf(f, A, C, p, fc.traces_corr)
    assert res.A.shape[0] == res.C.shape[0] == res.YrA.shape[0] >= 0.7 * A.shape[0]
    assert res.b.shape == (f.P,) and res.f.shape == (f.T,) and res.counts["iter"] == 2
    assert res.counts["units_after_merge_1"] == res.A.shape[0]
    assert (res.YrA >= 0).all()


def test_run_cnmf_refreshes_background_on_last_iteration():
    f, fc, A, C = _setup()
    b0, _ = cnmf.update_background(f, A, C)
    # loop_merge_corr 1.0: no merge, so res.A / res.C are the units the last background saw
    p = Params(dl_wnd=5, size_thres=(9, None), n_iter=2, loop_merge_corr=1.0)
    res = cnmf.run_cnmf(f, A, C, p, fc.traces_corr)
    Y = _dense(f)
    R = np.clip(Y - res.A.toarray().astype(np.float64).T @ res.C.astype(np.float64), 0, None)
    b_ref = R.mean(1)
    f_ref = R.T @ b_ref / (b_ref @ b_ref)
    assert np.linalg.norm(res.b - b0) > 0.05 * np.linalg.norm(b0)  # footprints changed b
    assert np.allclose(res.b, b_ref, atol=1e-4 * b_ref.max())
    assert np.allclose(res.f, f_ref, rtol=1e-4)


def test_background_f64_spot_check_warns_on_wrong_float32_trace(monkeypatch):
    f, fc, A, C = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the correct path does not warn
        b, fb = cnmf.update_background(f, A, C, chunk=128)
    true_part = cnmf._trace_part
    monkeypatch.setattr(cnmf, "_trace_part", lambda R, bt: true_part(R, bt) * 1.01)
    with pytest.warns(RuntimeWarning, match="differs from float64"):
        cnmf.update_background(f, A, C, b=b, chunk=128)
    monkeypatch.setattr(cnmf, "_trace_part", lambda R, bt: true_part(R, bt) * np.nan)
    with pytest.warns(RuntimeWarning, match="differs from float64"):
        cnmf.update_background(f, A, C, b=b, chunk=128)


@pytest.mark.parametrize(
    "target, stage",
    [
        ("update_background", "initial background"),
        ("update_spatial", "spatial update"),
        ("compute_trace", "traces"),
    ],
)
def test_run_cnmf_raises_on_non_finite_stage_output(monkeypatch, target, stage):
    f, fc, A, C = _setup()
    real = getattr(cnmf, target)

    def poisoned(*a, **kw):
        out = real(*a, **kw)
        if target == "update_spatial":
            A2, mask, nf = out
            A2 = A2.copy()
            A2.data[0] = np.nan
            return A2, mask, nf
        if target == "update_background":
            return out[0], np.full_like(out[1], np.nan)
        return np.full_like(out, np.inf)

    monkeypatch.setattr(cnmf, target, poisoned)
    p = Params(dl_wnd=5, size_thres=(9, None), n_iter=2)
    with pytest.raises(RuntimeError, match=stage):
        cnmf.run_cnmf(f, A, C, p, fc.traces_corr)
