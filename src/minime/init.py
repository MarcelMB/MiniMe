"""CNMF initialisation on factors: footprints, traces and the merge of duplicate units.

``initA`` takes each seed's correlation window (``FactorCorr.window``) thresholded at
``thres_corr`` as its footprint (Minian ``initA``). ``initC`` solves the least-squares
problem ``min_C ||A^T C - Y||`` for all frames at once through the normal equations
``(A A^T) C = A Y``, with ``A Y`` from the factors. ``unit_merge`` averages units whose
footprints overlap and whose traces correlate (Minian ``cnmf.unit_merge``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from minime.corr import FactorCorr, merge_labels
from minime.factors import Factors

COND_MAX = 1e12  # above this, G is treated as rank-deficient and pinv is used
PINV_RCOND = 1e-12  # relative cutoff on eigenvalues of G (~1e-6 on singular values of A)


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def initA(fc: FactorCorr, seeds: pd.DataFrame, thres_corr: float, wnd: int) -> sps.csr_matrix:
    """(K, P) float32 footprints: each seed's window correlations ``>= thres_corr``.

    Seeds are taken in DataFrame order; a seed with no pixel above the threshold yields
    no row.
    """
    W = fc.f.W_
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    k = 0
    for h, w in zip(seeds["height"].to_numpy(), seeds["width"].to_numpy(), strict=True):
        idx, c = fc.window(int(h) * W + int(w), wnd)
        keep = c >= thres_corr
        if not keep.any():
            continue
        cols.append(np.asarray(idx)[keep])
        vals.append(np.asarray(c)[keep])
        rows.append(np.full(int(keep.sum()), k, dtype=np.int64))
        k += 1
    if k == 0:
        raise ValueError("initA produced no footprints")
    coo = sps.coo_matrix(
        (
            np.concatenate(vals).astype(np.float32),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(k, fc.f.P),
    )
    return coo.tocsr()


def centroids(A: sps.csr_matrix, H: int, W: int) -> np.ndarray:
    """(K, 2) float64 intensity-weighted (row, col) centre of each footprint."""
    A = sps.csr_matrix(A, dtype=np.float64)
    if A.shape[1] != H * W:
        raise ValueError(f"A has {A.shape[1]} columns, expected H * W = {H * W}")
    pix = np.arange(H * W)
    coords = np.stack([pix // W, pix % W], axis=1).astype(np.float64)  # (P, 2)
    mass = np.asarray(A.sum(axis=1)).ravel()
    return (A @ coords) / mass[:, None]


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def _solve_normal(G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``G X = B`` by Cholesky; ``pinv(G) @ B`` when ``G`` is ill-conditioned or singular."""
    if np.linalg.cond(G) <= COND_MAX:
        try:
            cf = cho_factor(G, lower=True, check_finite=False)
        except LinAlgError:
            pass
        else:
            return cho_solve(cf, B, check_finite=False)
    return np.linalg.pinv(G, rcond=PINV_RCOND, hermitian=True) @ B


def initC(f: Factors, A: sps.csr_matrix) -> np.ndarray:
    """(K, T) float32 least-squares traces of footprints ``A`` (K, P), no sign constraint."""
    A = sps.csr_matrix(A)
    A64 = A.astype(np.float64)
    G = (A64 @ A64.T).toarray()
    G = 0.5 * (G + G.T)
    return _solve_normal(G, f.AtY(A)).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit merge
# ---------------------------------------------------------------------------


def _group_mean_matrix(labels: np.ndarray) -> sps.csr_matrix:
    """(K', K) indicator with ``1/|group|`` entries, groups in order of first appearance."""
    labels = np.asarray(labels)
    _, first, inv, counts = np.unique(
        labels, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.argsort(first, kind="stable")  # unique-sorted group -> appearance rank
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    g = rank[inv]
    K = labels.size
    return sps.csr_matrix(
        (1.0 / counts[inv], (g, np.arange(K))), shape=(order.size, K), dtype=np.float64
    )


def unit_merge(
    A: sps.csr_matrix,
    C: np.ndarray,
    fc_traces: Callable[[np.ndarray], np.ndarray],
    thres_corr: float,
    extra: list[np.ndarray] | None = None,
) -> tuple[sps.csr_matrix, np.ndarray, list[np.ndarray]]:
    """Replace each transitive group of overlapping, correlated units by its mean.

    Two units are adjacent when their footprints share a pixel and
    ``fc_traces(C)[i, j] > thres_corr``. ``A``, ``C`` and every ``extra`` (K, ...) array
    are averaged over each connected component; output dtypes follow the inputs.
    """
    A = sps.csr_matrix(A)
    overlap = (A @ A.T).toarray() > 0
    np.fill_diagonal(overlap, False)
    adj = overlap & (np.asarray(fc_traces(C)) > thres_corr)
    labels = merge_labels(sps.csr_matrix(adj))
    M = _group_mean_matrix(labels)
    A2 = sps.csr_matrix(M @ A, dtype=A.dtype)
    C2 = np.asarray(M @ C).astype(C.dtype, copy=False)
    extra2 = [np.asarray(M @ e).astype(e.dtype, copy=False) for e in (extra or [])]
    return A2, C2, extra2
