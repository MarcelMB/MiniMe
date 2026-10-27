"""Band-limited pixel correlations computed on the per-tile SVD factors.

The low-pass is linear, so low-passing the temporal factors gives the low-passed pixel
traces exactly: ``Y_lp,b = W_b V_lp,b^T``. With ``G_ab = V_lp,a^T V_lp,b`` (r_a, r_b)::

    num(i, j)  = W_i G_ab W_j^T
    ||i||^2    = W_i G_aa W_i^T + f_lp * resid_i     ("resid" normalisation)
    corr(i, j) = num / sqrt(||i||^2 ||j||^2)

``resid_i = max(energy_i - ||W_i||^2, 0)`` is the energy the compression discarded,
assumed white; ``f_lp = 2 (nr - 1) / T`` (``nr = int(noise_freq * T)``) is the share of
white-noise energy inside the passband. The numerator stays denoised and the denominator
matches the raw trace, so a noise-only pixel keeps a correlation near zero. A pixel paired
with itself shares its own noise too (raw ``num(i, i) = ||y_lp,i||^2 ~ ||i||^2``), so the
self pair is exactly 1.
Ported from ``minian_v2.0/scripts/bench_corr/candidates/compressed_svd.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from caanfast.factors import Factors

# ---------------------------------------------------------------------------
# Low-pass filter
# ---------------------------------------------------------------------------


def lowpass(X: np.ndarray, noise_freq: float) -> np.ndarray:
    """Brick-wall low-pass along the last axis, DC removed: float64, same shape as ``X``."""
    T = X.shape[-1]
    Xf = np.fft.rfft(np.asarray(X, dtype=np.float64), axis=-1)
    Xf[..., 0] = 0.0
    Xf[..., int(noise_freq * T) :] = 0.0
    return np.fft.irfft(Xf, T, axis=-1)


# ---------------------------------------------------------------------------
# Correlations on the factors
# ---------------------------------------------------------------------------


class FactorCorr:
    """Correlation of low-passed pixel traces as bilinear forms on the tile factors."""

    def __init__(self, f: Factors, noise_freq: float) -> None:
        self.f = f
        self.noise_freq = noise_freq
        T = f.T
        nr = int(noise_freq * T)
        f_lp = 2.0 * (nr - 1) / T  # passband share of white-noise energy
        self.Vlp: list[np.ndarray] = [lowpass(tf.V.T, noise_freq).T for tf in f.tiles]
        self.Wd: list[np.ndarray] = [tf.W.astype(np.float64) for tf in f.tiles]
        self.Gaa: list[np.ndarray] = [v.T @ v for v in self.Vlp]
        self._G: dict[tuple[int, int], np.ndarray] = {}
        self._disks: dict[int, np.ndarray] = {}
        norm2 = np.empty(f.P, dtype=np.float64)
        for b, tf in enumerate(f.tiles):
            Wd = self.Wd[b]
            lowrank = ((Wd @ self.Gaa[b]) * Wd).sum(axis=1)
            resid = np.maximum(tf.energy.astype(np.float64) - (Wd**2).sum(axis=1), 0.0)
            norm2[f.tile_rows(b)] = lowrank + f_lp * resid
        self.norm2 = norm2

    def _gab(self, a: int, b: int) -> np.ndarray:
        if a == b:
            return self.Gaa[a]
        key = (a, b)
        if key not in self._G:
            self._G[key] = self.Vlp[a].T @ self.Vlp[b]
        return self._G[key]

    def pairs(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Correlations of flat pixel pairs ``(src[k], dst[k])``: float64 in [-1, 1]."""
        f = self.f
        src = np.asarray(src, dtype=np.int64).ravel()
        dst = np.asarray(dst, dtype=np.int64).ravel()
        if src.shape != dst.shape:
            raise ValueError(f"src {src.shape} and dst {dst.shape} differ")
        bs, bd = f.tile_of[src].astype(np.int64), f.tile_of[dst].astype(np.int64)
        ls, ld = f.loc_of[src], f.loc_of[dst]
        num = np.empty(src.size, dtype=np.float64)
        key = bs * len(f.tiles) + bd
        for k in np.unique(key):
            m = np.flatnonzero(key == k)
            a, b = int(bs[m[0]]), int(bd[m[0]])
            num[m] = ((self.Wd[a][ls[m]] @ self._gab(a, b)) * self.Wd[b][ld[m]]).sum(axis=1)
        den = np.sqrt(self.norm2[src] * self.norm2[dst])
        out = np.zeros(src.size, dtype=np.float64)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        out[ok & (src == dst)] = 1.0
        return np.clip(out, -1.0, 1.0)

    def _disk(self, wnd: int) -> np.ndarray:
        """(n, 2) offsets ``(dy, dx)`` with ``dy^2 + dx^2 <= wnd^2``, cached per ``wnd``."""
        if wnd not in self._disks:
            dy, dx = np.mgrid[-wnd : wnd + 1, -wnd : wnd + 1]
            inside = dy**2 + dx**2 <= wnd**2
            self._disks[wnd] = np.stack([dy[inside], dx[inside]], axis=1)
        return self._disks[wnd]

    def window(self, seed: int, wnd: int) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices of the pixels within distance ``wnd`` of ``seed`` and their corr to it."""
        f = self.f
        h0, w0 = divmod(int(seed), f.W_)
        off = self._disk(wnd)
        h, w = h0 + off[:, 0], w0 + off[:, 1]
        inside = (h >= 0) & (h < f.H) & (w >= 0) & (w < f.W_)
        idx = (h[inside] * f.W_ + w[inside]).astype(np.int64)
        return idx, self.pairs(np.full(idx.size, int(seed), dtype=np.int64), idx)

    def traces_corr(self, C: np.ndarray) -> np.ndarray:
        """(K, K) Pearson correlation of the low-passed rows of ``C``; 0 for flat rows."""
        Clp = lowpass(np.atleast_2d(C), self.noise_freq)
        Clp -= Clp.mean(axis=1, keepdims=True)
        nrm = np.sqrt((Clp**2).sum(axis=1))
        good = nrm > 0
        Z = np.zeros_like(Clp)
        Z[good] = Clp[good] / nrm[good, None]
        R = np.clip(Z @ Z.T, -1.0, 1.0)
        np.fill_diagonal(R, 1.0)
        return R


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_labels(adj: sps.spmatrix) -> np.ndarray:
    """Connected-component label of every node of the (symmetrised) adjacency ``adj``.

    Used by ``init.unit_merge`` (a few hundred units); ``seeds_merge`` uses
    ``brightest_parent`` instead.
    """
    return connected_components(adj, directed=False)[1]


def brightest_parent(edges: np.ndarray, bright: np.ndarray) -> np.ndarray:
    """Parent of each node: its brightest neighbour in ``edges`` that is brighter than it.

    ``edges`` is (m, 2) int; a node with no brighter neighbour is its own parent (a root).
    Ties in brightness are broken by index (lower index counts as brighter).
    """
    n = bright.size
    parent = np.arange(n)
    if len(edges) == 0:
        return parent
    order = np.lexsort((np.arange(n), -np.asarray(bright, dtype=np.float64)))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    up = rank[edges[:, 0]] < rank[edges[:, 1]]  # is edges[:, 0] the brighter end?
    par = np.where(up, edges[:, 0], edges[:, 1])
    child = np.where(up, edges[:, 1], edges[:, 0])
    srt = np.lexsort((rank[par], child))  # per child, brightest parent first
    child, par = child[srt], par[srt]
    first = np.r_[True, child[1:] != child[:-1]]
    parent[child[first]] = par[first]
    return parent


def seeds_merge(
    fc: FactorCorr,
    seeds: pd.DataFrame,
    max_proj: np.ndarray,
    thres_dist: int,
    thres_corr: float,
) -> pd.DataFrame:
    """Add ``mask_mrg``: keep the brightest seed of each group of close, correlated seeds.

    Two seeds are linked when they lie within ``thres_dist`` and their traces correlate
    above ``thres_corr``. Each seed joins the group of its brightest linked neighbour that is
    brighter than itself (``brightest_parent``), so groups are trees rooted at a local
    brightness maximum, and the roots are kept. Minian takes the connected components of
    the same links instead; that merge runs through the dim pixels between two cells once
    those are seeds too, and on a 2 h session (48k seeds, 280 rolling windows) it collapsed
    32k seeds into 166 (bench/README.md, "Long recordings"). The tree rule never climbs
    from a dim seed to a brighter one, so a chain stops at the valley between two cells.
    """
    out = seeds.copy()
    h = out["height"].to_numpy().astype(np.int64)
    w = out["width"].to_numpy().astype(np.int64)
    n = h.size
    keep = np.ones(n, dtype=bool)
    if n > 1:
        pr = cKDTree(np.stack([h, w], axis=1)).query_pairs(thres_dist, output_type="ndarray")
        if len(pr):
            c = fc.pairs(h[pr[:, 0]] * fc.f.W_ + w[pr[:, 0]], h[pr[:, 1]] * fc.f.W_ + w[pr[:, 1]])
            parent = brightest_parent(pr[c > thres_corr], np.asarray(max_proj)[h, w])
            keep = parent == np.arange(n)
    out["mask_mrg"] = keep
    return out
