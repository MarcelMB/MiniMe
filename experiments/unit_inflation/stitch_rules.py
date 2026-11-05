"""Stitch variants on the same window units: ``base`` is ``stream.stitch`` as the browser ran it
until 2026-11-05, ``mindistD`` adds the cross-window ``merge_min_dist`` rule at D px (the fix),
``attachNN`` the rejected attach step at cosine 0.NN."""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streaming"))
import stream as st  # noqa: E402

from minime import corr  # noqa: E402

Result = tuple[sps.csr_matrix, np.ndarray, np.ndarray]


def _attach_targets(
    A: sps.csr_matrix, g: np.ndarray, win: np.ndarray, n_win: int, cos: float, max_windows: int
) -> np.ndarray:
    """Target group of every group: a group of at most ``max_windows`` windows joins the
    group of more windows with the highest mean-footprint cosine ``>= cos`` that has no unit
    in its windows; others stay. Within a window CNMF keeps two cells apart when both are
    there, so a unit of few windows that resembles a recurring cell which its window did not
    find is that cell, estimated worse."""
    n_g = int(g.max()) + 1
    G = st._group_means(A, g)
    gn = np.sqrt(np.asarray(G.multiply(G).sum(axis=1)).ravel())
    gn[gn == 0] = 1.0
    Gn = sps.csr_matrix(sps.diags(1.0 / gn) @ G)
    S = sps.csr_matrix(Gn @ Gn.T)
    S.sort_indices()
    Wg = sps.csr_matrix((np.ones(g.size), (g, win)), shape=(n_g, n_win)) > 0
    nwin = np.asarray(Wg.sum(axis=1)).ravel()
    shared = sps.csr_matrix(Wg.astype(np.int32) @ Wg.T.astype(np.int32))
    target = np.arange(n_g)
    for s in np.flatnonzero(nwin <= max_windows):
        best, best_cos = -1, -np.inf
        lo, hi = S.indptr[s], S.indptr[s + 1]
        for h, v in zip(S.indices[lo:hi], S.data[lo:hi], strict=True):
            if nwin[h] > max_windows and v >= cos and v > best_cos and shared[s, h] == 0:
                best, best_cos = h, v
        if best >= 0:
            target[s] = best
    return target


def base(units: list[st.WindowUnits], cos: float) -> Result:
    return st.stitch(units, cos)


def _groups(
    units: list[st.WindowUnits], cos: float
) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """``stream.stitch``'s links and brightest-parent groups: (A, window, bright, group)."""
    A = st._row_max_one(sps.vstack([u.A for u in units]).tocsr().astype(np.float64))
    win = np.concatenate([np.full(u.A.shape[0], i) for i, u in enumerate(units)])
    bright = np.concatenate([u.bright for u in units]).astype(np.float64)
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    An = sps.csr_matrix(sps.diags(1.0 / norms) @ A)
    S = sps.triu(An @ An.T, k=1).tocoo()
    keep = (S.data > cos) & (win[S.row] != win[S.col])
    edges = np.stack([S.row[keep], S.col[keep]], axis=1).astype(np.int64)
    _, g = np.unique(st._roots(corr.brightest_parent(edges, bright)), return_inverse=True)
    return A, win, bright, g


def attach(units: list[st.WindowUnits], cos: float, attach_cos: float, windows: int = 1) -> Result:
    """Rejected variant: a group of at most ``windows`` windows joins the most similar group of
    more windows (mean-footprint cosine >= ``attach_cos``) that has no unit in its windows.
    Fine on the synthetic sessions, but on real V4 data it joins units ~18 px apart (large
    footprints give high cosines)."""
    A, win, bright, g = _groups(units, cos)
    _, g = np.unique(
        _attach_targets(A, g, win, len(units), attach_cos, windows)[g], return_inverse=True
    )
    n_g = int(g.max()) + 1
    A_st = st._group_means(A, g).astype(np.float32)
    bright_g = np.zeros(n_g)
    np.maximum.at(bright_g, g, bright)
    n_windows = np.bincount(np.unique(np.stack([g, win]), axis=1)[0], minlength=n_g)
    return A_st, bright_g, n_windows


def mindist_bp(units: list[st.WindowUnits], cos: float, d: float) -> Result:
    """``min_dist`` with brightest-parent joins instead of connected components: each group
    joins its brightest group closer than ``d`` px that is brighter than itself."""
    A, win, bright, g = _groups(units, cos)
    H = W = int(round(np.sqrt(A.shape[1])))
    n_g = int(g.max()) + 1
    bg = np.zeros(n_g)
    np.maximum.at(bg, g, bright)
    c = st.init.centroids(st._group_means(A, g), H, W)
    dist = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    i, j = np.nonzero(np.triu(dist < d, 1))
    _, comp = np.unique(
        st._roots(corr.brightest_parent(np.stack([i, j], 1), bg)), return_inverse=True
    )
    g = comp[g]
    n_g = int(g.max()) + 1
    A_st = st._group_means(A, g).astype(np.float32)
    bright_g = np.zeros(n_g)
    np.maximum.at(bright_g, g, bright)
    n_windows = np.bincount(np.unique(np.stack([g, win]), axis=1)[0], minlength=n_g)
    return A_st, bright_g, n_windows


def mindist(units: list[st.WindowUnits], cos: float, d: float) -> Result:
    H = W = int(round(np.sqrt(units[0].A.shape[1])))
    return st.stitch(units, cos, min_dist=d, shape=(H, W))


RULES: dict[str, Callable[[list[st.WindowUnits], float], Result]] = {
    "base": base,
    **{f"attach{c}": partial(attach, attach_cos=c / 100) for c in (30, 40, 50, 60)},
    "attach40w2": partial(attach, attach_cos=0.4, windows=2),
    "attach50w2": partial(attach, attach_cos=0.5, windows=2),
    **{f"mindist{d}": partial(mindist, d=float(d)) for d in (3, 4, 5, 6)},
    **{f"mindistbp{d}": partial(mindist_bp, d=float(d)) for d in (5, 6)},
}
