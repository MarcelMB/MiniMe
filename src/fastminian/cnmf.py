"""CNMF stages on factors: spatial update, clipped background, residual traces, the loop.

Every stage reads the movie only through ``Factors`` (``YCt``, ``AtY``,
``reconstruct_chunk``); nothing the size of the movie is held or written.

* ``update_spatial``: Minian ``update_spatial`` with the per-pixel LassoLars replaced by
  the Gram-shared NNQP solver of ``spatial_kernel`` (same objective, true optimum), then
  Minian's post-processing (strict ``size_thres``, divide by the footprint maximum).
* ``update_background``: Minian's ``b = mean_t clip(Y - A C, 0)``,
  ``f_t = clip(Y_t - A C_t, 0) . b / (b . b)``, streamed over reconstructed tile chunks
  (``bench_passes/candidates/bg_streamed.py``).
* ``compute_trace``: Minian ``compute_trace`` (``trace_shared.py`` algebra).
* ``run_cnmf``: background, then ``n_iter`` x (spatial, background, traces,
  ``C := YrA``, unit merge).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import scipy.sparse as sps

from fastminian import init
from fastminian.factors import Factors
from fastminian.params import Params
from fastminian.spatial_kernel import solve_nnqp

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
    G = Cc @ Cc.T
    B = f.YCt(Cc)[active].astype(np.float64)
    lam = f.T * sparse_penal * np.asarray(sn, dtype=np.float64)[active]
    values = solve_nnqp(G, B, lam, indptr, indices)

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


def tile_units(Acsc: sps.csc_matrix, rows: np.ndarray) -> np.ndarray:
    """Indices of the units whose footprint has a stored entry among the pixels ``rows``."""
    return np.flatnonzero(np.diff(sps.csr_matrix(Acsc[:, rows]).indptr))


def update_background(
    f: Factors,
    A: sps.csr_matrix,
    C: np.ndarray,
    b: np.ndarray | None = None,
    chunk: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """``(b (P,) float32, f (T,) float32)`` from the clipped residual ``clip(Y - A^T C, 0)``.

    Pass 1 (skipped when ``b`` is given) ``b = mean_t R``; pass 2 ``f_t = R_t . b / (b . b)``.
    Both stream ``chunk`` frames of one tile at a time; ``A^T C`` of a tile uses only the
    units that touch it (all K units made this the slowest CNMF stage).
    """
    Acsc = sps.csc_matrix(A, dtype=np.float32)
    C32 = np.ascontiguousarray(C, dtype=np.float32)
    given = b is not None
    b_out = np.asarray(b, dtype=np.float32).copy() if given else np.empty(f.P, dtype=np.float32)
    num = np.zeros(f.T, dtype=np.float64)
    for t in range(len(f.tiles)):
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
        for t0, t1 in ranges:
            num[t0:t1] += _clipped_residual(f, t, Ab, Cb, t0, t1).T @ b_tile
    bb = float(b_out.astype(np.float64) @ b_out.astype(np.float64))
    fb = num / bb if bb > 0 else np.zeros(f.T)
    return b_out, fb.astype(np.float32)


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
    A: sps.csr_matrix  # (K, P) float32 footprints, row maxima 1
    C: np.ndarray  # (K, T) float32, trace fed to the last spatial update (merged)
    YrA: np.ndarray  # (K, T) float32, final residual trace (merged)
    b: np.ndarray  # (P,) float32 background footprint
    f: np.ndarray  # (T,) float32 background trace
    counts: dict[str, int] = field(default_factory=dict)


def run_cnmf(
    fac: Factors,
    A: sps.csr_matrix,
    C: np.ndarray,
    params: Params,
    fc_traces: Callable[[np.ndarray], np.ndarray],
) -> CnmfResult:
    """Background, then ``params.n_iter`` x (spatial, background, traces, merge)."""
    sn = fac.sigma_flat()
    counts: dict[str, int] = {}
    b, fb = update_background(fac, A, C)
    C_sp = np.asarray(C, dtype=np.float32)
    YrA = C_sp
    for it in range(params.n_iter):
        A, mask, nf = update_spatial(
            fac, A, C, sn, params.dl_wnd, params.sparse_penal, params.size_thres
        )
        C_sp = (np.asarray(C)[mask] * nf[:, None]).astype(np.float32)
        counts[f"units_after_spatial_{it}"] = A.shape[0]
        b, fb = update_background(fac, A, C_sp, b=b)
        YrA = compute_trace(fac, A, b, C_sp, fb)
        A, YrA, (C_sp,) = init.unit_merge(A, YrA, fc_traces, params.loop_merge_corr, [C_sp])
        C = YrA
        counts[f"units_after_merge_{it}"] = A.shape[0]
    counts["iter"] = params.n_iter
    return CnmfResult(A, C_sp, YrA, b, fb, counts)
