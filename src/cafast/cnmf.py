"""CNMF stages on factors: spatial update, background, residual traces, the loop.

Every stage reads the movie only through ``Factors`` (``YCt``, ``AtY``,
``reconstruct_chunk``); nothing the size of the movie is held or written.

* ``update_spatial``: Minian ``update_spatial`` with the per-pixel LassoLars replaced by
  the Gram-shared NNQP solver of ``spatial_kernel`` (same objective, true optimum, on unit-norm
  traces as ``LassoLars(normalize=True)`` in the scikit-learn Minian pins; see
  ``column_norms``), then Minian's post-processing (strict ``size_thres``, divide by the
  footprint maximum).
* ``update_background``: by default (``Params.background_clip = False``) the rank-1
  background of the unclipped residual, ``b = clip(mean_t (Y - A C), 0)``,
  ``f_t = (Y_t - A C_t) . b / (b . b)``, computed on the factors without reconstructing the
  movie. ``background_clip = True`` gives Minian's ``b = mean_t clip(Y - A C, 0)``,
  ``f_t = clip(Y_t - A C_t, 0) . b / (b . b)``, streamed over reconstructed tile chunks
  (``bench_passes/candidates/bg_streamed.py``), one tile per thread.
* ``compute_trace``: Minian ``compute_trace`` (``trace_shared.py`` algebra).
* ``run_cnmf``: background, then ``n_iter`` x (spatial, background, traces,
  ``C := YrA``, unit merge). The background footprint ``b`` is computed in full on the
  first and on the last iteration (Minian recomputes it every iteration; in between the
  previous ``b`` is reused and only ``f`` is refreshed).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import scipy.sparse as sps

from cafast import init
from cafast.factors import Factors
from cafast.params import Params
from cafast.spatial_kernel import solve_nnqp

# ---------------------------------------------------------------------------
# Spatial update
# ---------------------------------------------------------------------------


def _disk(radius: int) -> np.ndarray:
    """(2r+1, 2r+1) uint8 disk, ``dy^2 + dx^2 <= r^2`` (``skimage.morphology.disk``)."""
    dy, dx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (dy**2 + dx**2 <= radius**2).astype(np.uint8)


def dilate_footprints(A: sps.csr_matrix, H: int, W: int, dl_wnd: int) -> sps.csr_matrix:
    """(K, P) bool CSR: each footprint's support dilated by a disk of radius ``dl_wnd``."""
    A = sps.csr_matrix(A)
    kernel = _disk(dl_wnd)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for k in range(A.shape[0]):
        img = np.zeros(H * W, dtype=np.uint8)
        lo, hi = A.indptr[k], A.indptr[k + 1]
        img[A.indices[lo:hi][A.data[lo:hi] > 0]] = 1
        pix = np.flatnonzero(cv2.dilate(img.reshape(H, W), kernel))
        cols.append(pix)
        rows.append(np.full(pix.size, k, dtype=np.int64))
    data = np.ones(sum(c.size for c in cols), dtype=bool)
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    return sps.csr_matrix((data, (r, c)), shape=(A.shape[0], H * W), dtype=bool)


def _size_mask(nnz: np.ndarray, size_thres: tuple[float | None, float | None]) -> np.ndarray:
    """Minian's strict ``low < nnz < high`` (a falsy bound is skipped); empty units dropped."""
    low, high = size_thres
    mask = nnz > 0  # an empty footprint has no maximum to normalise by
    if low:
        mask &= nnz > low
    if high:
        mask &= nnz < high
    return mask


def column_norms(Cc: np.ndarray) -> np.ndarray:
    """(K,) L2 norms of the centred traces ``Cc`` (zero norms become 1).

    Minian was tuned on scikit-learn < 1.2 (it pins 0.22.1), where ``LassoLars`` defaulted
    to ``normalize=True``: the fit ran on the columns of ``C^T`` scaled to unit norm and the
    coefficients were scaled back. The penalty on ``a_k`` is then ``||Cc_k||`` (about
    ``sqrt(T) std(C_k)``, ~350 on a 40k-frame session) times stronger than in an unscaled
    fit; without it ``sparse_penal`` is negligible and footprints fill the dilated support.
    """
    norms = np.linalg.norm(Cc, axis=1)
    norms[norms == 0] = 1.0
    return norms


def update_spatial(
    f: Factors,
    A: sps.csr_matrix,
    C: np.ndarray,
    sn: np.ndarray,
    dl_wnd: int,
    sparse_penal: float,
    size_thres: tuple[float | None, float | None],
) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """New footprints on the dilated supports of ``A``.

    Returns ``(A_new (K', P) float32 with row maxima 1, mask (K,) bool of kept units,
    norm_fac (K',) float32 row maxima before normalisation)``.
    """
    K = A.shape[0]
    sub = dilate_footprints(A, f.H, f.W_, dl_wnd)
    subT = sps.csr_matrix(sub.T)  # (P, K): per-pixel allowed units, sorted
    counts = np.diff(subT.indptr)
    active = np.flatnonzero(counts)
    indptr = np.zeros(active.size + 1, dtype=np.int64)
    np.cumsum(counts[active], out=indptr[1:])
    indices = subT.indices.astype(np.int64)  # inactive pixels hold no entries

    Cc = np.asarray(C, dtype=np.float64)
    Cc = Cc - Cc.mean(axis=1, keepdims=True)
    scale = column_norms(Cc)
    Cc /= scale[:, None]
    G = Cc @ Cc.T
    B = f.YCt(Cc)[active].astype(np.float64)
    lam = f.T * sparse_penal * np.asarray(sn, dtype=np.float64)[active]
    values = solve_nnqp(G, B, lam, indptr, indices) / scale[indices]

    pix = np.repeat(active, counts[active])
    A_new = sps.csr_matrix((values, (indices, pix)), shape=(K, f.P), dtype=np.float64)
    A_new.eliminate_zeros()
    mask = _size_mask(np.diff(A_new.indptr), size_thres)
    if not mask.any():
        raise ValueError("all units dropped by size_thres")
    A_new = A_new[np.flatnonzero(mask)]
    norm_fac = A_new.max(axis=1).toarray().ravel()
    A_new = sps.csr_matrix(sps.diags(1.0 / norm_fac) @ A_new, dtype=np.float32)
    return A_new, mask, norm_fac.astype(np.float32)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def _clipped_residual(
    f: Factors, b: int, Ab: np.ndarray, C: np.ndarray, t0: int, t1: int
) -> np.ndarray:
    """``clip(Y_b[:, t0:t1] - Ab^T C[:, t0:t1], 0)`` (P_b, Tc) float32."""
    R = f.reconstruct_chunk(b, t0, t1)
    R -= Ab.T @ C[:, t0:t1]
    np.maximum(R, 0, out=R)
    return R


F64_CHECK_RTOL = 1e-3  # warn when the float32 background trace of the spot check is off


def _trace_part(R: np.ndarray, b_tile: np.ndarray) -> np.ndarray:
    """``R^T b`` (Tc,) of one clipped-residual chunk, float32.

    einsum, not ``R.T @ b``: after a numba parallel region Accelerate's float32 transposed
    gemv returned wrong values for output lengths ~513-575 (macOS 26, M1 Max); the
    541-frame tail chunk of a 39541-frame session turned f into NaN. The cause is not
    established (it did not reproduce later), so ``update_background`` also recomputes one
    chunk in float64 and ``run_cnmf`` checks every stage's output for NaN/inf.
    """
    return np.einsum("pt,p->t", R, b_tile)


def f64_spot_check(
    f: Factors,
    t: int,
    Ab: np.ndarray,
    Cb: np.ndarray,
    t0: int,
    t1: int,
    b_tile: np.ndarray,
    part: np.ndarray,
) -> float:
    """Relative error of the float32 ``part = R^T b`` of tile ``t``, frames ``t0:t1``, vs float64.

    Warns (``RuntimeWarning``) when it exceeds ``F64_CHECK_RTOL`` or is not finite.
    """
    tf = f.tiles[t]
    Y = tf.W.astype(np.float64) @ tf.V[t0:t1].astype(np.float64).T + tf.mean[:, None]
    R = np.clip(Y - Ab.astype(np.float64).T @ Cb[:, t0:t1].astype(np.float64), 0, None)
    ref = R.T @ b_tile.astype(np.float64)
    den = float(np.linalg.norm(ref))
    diff = float(np.linalg.norm(np.asarray(part, dtype=np.float64) - ref))
    err = diff / den if den > 0 else diff
    if not np.isfinite(err) or err > F64_CHECK_RTOL:
        warnings.warn(
            f"update_background: float32 background trace of tile {t}, frames {t0}-{t1} "
            f"differs from float64 by {err:.3g} (relative); f may be wrong",
            RuntimeWarning,
            stacklevel=4,
        )
    return err


def tile_units(Acsc: sps.csc_matrix, rows: np.ndarray) -> np.ndarray:
    """Indices of the units whose footprint has a stored entry among the pixels ``rows``."""
    return np.flatnonzero(np.diff(sps.csr_matrix(Acsc[:, rows]).indptr))


def _tile_background(
    f: Factors,
    t: int,
    Acsc: sps.csc_matrix,
    C32: np.ndarray,
    b_out: np.ndarray,
    given: bool,
    chunk: int,
) -> tuple[np.ndarray, tuple]:
    """One tile of the clipped background: ``(num_t (T,) float64, spot-check arguments)``.

    Writes ``b_out[rows]`` of the tile (disjoint rows, so tiles can run in threads) unless
    ``given``. ``num_t = R^T b_tile`` over all frames; the spot-check arguments are those of
    the tile's last chunk.
    """
    rows = f.tile_rows(t)
    ks = tile_units(Acsc, rows)
    Ab = Acsc[ks][:, rows].toarray()  # (K_b, P_b) float32
    Cb = C32[ks]
    ranges = [(t0, min(f.T, t0 + chunk)) for t0 in range(0, f.T, chunk)]
    if not given:
        acc = np.zeros(rows.size, dtype=np.float64)
        for t0, t1 in ranges:
            acc += _clipped_residual(f, t, Ab, Cb, t0, t1).sum(axis=1, dtype=np.float64)
        b_out[rows] = acc / f.T
    b_tile = b_out[rows]
    num = np.zeros(f.T, dtype=np.float64)
    for t0, t1 in ranges:
        part = _trace_part(_clipped_residual(f, t, Ab, Cb, t0, t1), b_tile)
        num[t0:t1] = part
    return num, (f, t, Ab, Cb, t0, t1, b_tile, part)


def _background_clipped(
    f: Factors, A: sps.csr_matrix, C: np.ndarray, b: np.ndarray | None, chunk: int, threads: int
) -> tuple[np.ndarray, np.ndarray]:
    """Minian's clipped background, one tile per thread (numpy/BLAS release the GIL).

    Each tile's ``R^T b`` is summed into ``num`` in tile order, so the result does not depend
    on ``threads`` and equals the serial loop bit for bit.
    """
    Acsc = sps.csc_matrix(A, dtype=np.float32)
    C32 = np.ascontiguousarray(C, dtype=np.float32)
    given = b is not None
    b_out = np.asarray(b, dtype=np.float32).copy() if given else np.empty(f.P, dtype=np.float32)
    num = np.zeros(f.T, dtype=np.float64)
    check_tile = f.T % len(f.tiles)
    check: tuple = ()

    def one(t: int) -> tuple[np.ndarray, tuple]:
        return _tile_background(f, t, Acsc, C32, b_out, given, chunk)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for t, (num_t, args) in enumerate(ex.map(one, range(len(f.tiles)))):
            num += num_t
            if t == check_tile:
                check = args
    f64_spot_check(*check)
    bb = float(b_out.astype(np.float64) @ b_out.astype(np.float64))
    fb = num / bb if bb > 0 else np.zeros(f.T)
    return b_out, fb.astype(np.float32)


def _background_unclipped(
    f: Factors, A: sps.csr_matrix, C: np.ndarray, b: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 background of the unclipped residual ``Y - A^T C``, on the factors only.

    ``b = clip(mean_t Y - A^T mean_t C, 0)`` and ``f_t = (Y_t - A^T C_t) . b / (b . b)``
    with ``Y_t . b = sum_tiles V_t W^T b_tile + mean . b_tile``. No frame is reconstructed.
    """
    A64 = sps.csr_matrix(A, dtype=np.float64)
    C64 = np.asarray(C, dtype=np.float64)
    if b is None:
        ymean = np.empty(f.P, dtype=np.float64)
        for t, tf in enumerate(f.tiles):
            vm = tf.V.astype(np.float64).mean(axis=0)
            ymean[f.tile_rows(t)] = tf.W.astype(np.float64) @ vm + tf.mean
        b64 = np.clip(ymean - A64.T @ C64.mean(axis=1), 0, None)
    else:
        b64 = np.asarray(b, dtype=np.float64)
    num = -(C64.T @ (A64 @ b64))
    for t, tf in enumerate(f.tiles):
        bt = b64[f.tile_rows(t)]
        num += tf.V.astype(np.float64) @ (tf.W.astype(np.float64).T @ bt)
        num += float(tf.mean.astype(np.float64) @ bt)
    bb = float(b64 @ b64)
    fb = num / bb if bb > 0 else np.zeros(f.T)
    return b64.astype(np.float32), fb.astype(np.float32)


def update_background(
    f: Factors,
    A: sps.csr_matrix,
    C: np.ndarray,
    b: np.ndarray | None = None,
    chunk: int = 1000,
    clip: bool = True,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(b (P,) float32, f (T,) float32)``, the rank-1 background of ``Y - A^T C``.

    ``clip=True`` (Minian): on ``R = clip(Y - A^T C, 0)``, pass 1 (skipped when ``b`` is
    given) ``b = mean_t R``, pass 2 ``f_t = R_t . b / (b . b)``. Both stream ``chunk`` frames
    of one tile at a time, ``threads`` tiles at once (default: all cores); ``A^T C`` of a
    tile uses only the units that touch it. The last chunk of one tile (``T mod n_tiles``)
    is recomputed in float64 (``f64_spot_check``).

    ``clip=False``: the same on the unclipped residual, computed on the factors without
    reconstructing the movie (``_background_unclipped``); ``b`` is clipped at 0 afterwards.
    """
    if not clip:
        return _background_unclipped(f, A, C, b)
    if threads is None:
        threads = min(os.cpu_count() or 1, len(f.tiles))
    return _background_clipped(f, A, C, b, chunk, threads)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def compute_trace(
    f: Factors, A: sps.csr_matrix, b: np.ndarray, C: np.ndarray, fb: np.ndarray
) -> np.ndarray:
    """``YrA (K, T) float32 = clip(C + A_norm ((Y - b f^T)^T A - C^T A A^T)^T, 0)``."""
    A = sps.csr_matrix(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Ab = A @ np.asarray(b, dtype=np.float64)  # (K,)
    YtA = (f.AtY(A) - np.outer(Ab, np.asarray(fb, dtype=np.float64))).T  # (T, K)
    AA = (A @ A.T).toarray()
    A_norm = 1.0 / np.asarray(A.multiply(A).sum(axis=1)).ravel()
    YrA = C + ((YtA - C.T @ AA) * A_norm).T
    return np.clip(YrA, 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


@dataclass
class CnmfResult:
    # (K, P) float32 footprints. Row maxima are 1 after the last spatial update; a unit the
    # last unit_merge formed is the mean of its members' rows, so its maximum can be below 1
    A: sps.csr_matrix
    C: np.ndarray  # (K, T) float32, trace fed to the last spatial update (merged)
    YrA: np.ndarray  # (K, T) float32, final residual trace (merged)
    b: np.ndarray  # (P,) float32 background footprint
    f: np.ndarray  # (T,) float32 background trace
    counts: dict[str, int] = field(default_factory=dict)


def _check_finite(stage: str, **arrays: np.ndarray) -> None:
    """Raise ``RuntimeError`` naming ``stage`` if any array holds NaN or inf."""
    bad = [k for k, v in arrays.items() if not np.isfinite(np.asarray(v)).all()]
    if bad:
        raise RuntimeError(f"CNMF {stage}: non-finite values in {', '.join(bad)}")


def run_cnmf(
    fac: Factors,
    A: sps.csr_matrix,
    C: np.ndarray,
    params: Params,
    fc_traces: Callable[[np.ndarray], np.ndarray],
    record: Callable[[str, sps.csr_matrix], None] | None = None,
) -> CnmfResult:
    """Background, then ``params.n_iter`` x (spatial, background, traces, merge).

    ``record(name, A)``, if given, receives the footprints after each spatial update
    (``spatial_{it}``) and after each unit merge (``merge_{it}``).
    """
    sn = fac.sigma_flat()
    counts: dict[str, int] = {}
    b, fb = update_background(fac, A, C, clip=params.background_clip)
    _check_finite("initial background", b=b, f=fb)
    C_sp = np.asarray(C, dtype=np.float32)
    YrA = C_sp
    for it in range(params.n_iter):
        A, mask, nf = update_spatial(
            fac, A, C, sn, params.dl_wnd, params.sparse_penal, params.size_thres
        )
        C_sp = (np.asarray(C)[mask] * nf[:, None]).astype(np.float32)
        _check_finite(f"spatial update (iteration {it})", A_data=A.data, C=C_sp)
        counts[f"units_after_spatial_{it}"] = A.shape[0]
        if record is not None:
            record(f"spatial_{it}", A)
        last = it == params.n_iter - 1
        b, fb = update_background(fac, A, C_sp, b=None if last else b, clip=params.background_clip)
        _check_finite(f"background (iteration {it})", b=b, f=fb)
        YrA = compute_trace(fac, A, b, C_sp, fb)
        _check_finite(f"traces (iteration {it})", YrA=YrA)
        A, YrA, (C_sp,) = init.unit_merge(A, YrA, fc_traces, params.loop_merge_corr, [C_sp])
        C = YrA
        counts[f"units_after_merge_{it}"] = A.shape[0]
        if record is not None:
            record(f"merge_{it}", A)
    counts["iter"] = params.n_iter
    return CnmfResult(A, C_sp, YrA, b, fb, counts)
