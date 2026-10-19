"""Gram-shared non-negative coordinate descent for the CNMF spatial update.

Per active pixel ``i`` with allowed units ``s = indices[indptr[i]:indptr[i + 1]]`` the
spatial update (Minian's per-pixel ``LassoLars(positive=True, fit_intercept=True)``,
multiplied by ``T`` and without its constant) is the quadratic program

    min_{w >= 0}  f(w) = 1/2 w^T G_s w - b_s^T w + lam_i sum(w)

with ``G = Cc Cc^T`` shared by every pixel, ``b = Cc y_i`` and ``lam_i = T * alpha_i``
(``update_spatial`` passes ``Cc`` with unit-norm rows, as ``LassoLars(normalize=True)``).
Each pixel runs cyclic coordinate descent with exact clipped 1-D minimisers and stops on
a certified duality gap (port of ``bench_spatial/candidates/numba_cd.py``).

The duality gap needs ``||y_c||^2``. The prototype took it from the movie; here only
``G`` and ``B`` are given, so the kernel uses the smallest value consistent with them,
``q = b_s^T G_s^+ b_s`` (the energy of ``y_c`` projected onto the row space of
``Cc_s``). The restricted problem only sees that projection, so the gap stays a valid
bound on ``f(w) - f(w*)``, and the relative stop ``gap <= rel_gap_tol * primal`` is at
least as strict as the prototype's (its primal is smaller by the orthogonal energy).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

REL_GAP_TOL = 1e-10  # relative duality gap at which a pixel is declared converged
MAX_SWEEPS = 100_000  # hard cap per pixel (never reached in practice)
EIG_RTOL = 1e-12  # eigenvalues of G_s below this fraction of the largest are treated as 0


@njit(cache=False, fastmath=False, nogil=True)
def _projected_energy(Gs: np.ndarray, bs: np.ndarray) -> float:
    """``bs^T Gs^+ bs`` by eigendecomposition, tiny eigenvalues treated as zero."""
    n = bs.shape[0]
    evals, evecs = np.linalg.eigh(Gs)
    cut = EIG_RTOL * evals[n - 1]
    q = 0.0
    for j in range(n):
        lj = evals[j]
        if lj > cut and lj > 0.0:
            u = 0.0
            for a in range(n):
                u += evecs[a, j] * bs[a]
            q += u * u / lj
    return q


@njit(cache=False, fastmath=False, nogil=True)
def _cd_pixel(
    Gs: np.ndarray,
    bs: np.ndarray,
    yn2: float,
    lam: float,
    w: np.ndarray,
    gw: np.ndarray,
    tol: float,
    max_sweeps: int,
) -> int:
    """Cyclic NNCD on f(w) = 1/2 w'Gs w - bs'w + lam*sum(w), w >= 0.  In-place on w."""
    n = bs.shape[0]
    for j in range(n):
        w[j] = 0.0
        gw[j] = 0.0  # gw = Gs @ w
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        max_delta = 0.0
        max_w = 0.0
        for k in range(n):
            gkk = Gs[k, k]
            old = w[k]
            if gkk <= 0.0:  # constant trace: unit cannot explain anything
                new = 0.0
            else:
                new = (bs[k] - (gw[k] - gkk * old) - lam) / gkk
                if new < 0.0:
                    new = 0.0
            d = new - old
            if d != 0.0:
                w[k] = new
                for j in range(n):
                    gw[j] += d * Gs[j, k]
                ad = abs(d)
                if ad > max_delta:
                    max_delta = ad
            if new > max_w:
                max_w = new
        # --- duality gap -------------------------------------------------------
        # residual r = yc - Cc^T w:  ||r||^2 = ||yc||^2 - 2 b'w + w'Gw ;  Cc r = b - Gw
        bw = 0.0
        wgw = 0.0
        sw = 0.0
        cmax = -1e300
        for j in range(n):
            bw += bs[j] * w[j]
            wgw += w[j] * gw[j]
            sw += w[j]
            c = bs[j] - gw[j]
            if c > cmax:
                cmax = c
        rr = yn2 - 2.0 * bw + wgw
        if rr < 0.0:
            rr = 0.0
        primal = 0.5 * rr + lam * sw
        # dual feasible point theta = s*r with max_k (Cc theta)_k <= lam
        s = 1.0
        if cmax > lam:
            s = lam / cmax
        dual = s * (yn2 - bw) - 0.5 * s * s * rr
        gap = primal - dual  # >= f(w) - f(w*) by weak duality
        scale = primal if primal > 1e-300 else 1e-300
        if gap <= tol * scale:
            break
        if max_delta <= 1e-15 * max_w:  # stalled at float resolution
            break
    return sweeps


@njit(parallel=True, cache=False, nogil=True)
def _cd_all(
    G: np.ndarray,
    B: np.ndarray,
    lam: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    tol: float,
    max_sweeps: int,
) -> None:
    """prange over active pixels; writes each pixel's solution into ``values[s:e]``."""
    for p in prange(indptr.shape[0] - 1):
        s = indptr[p]
        e = indptr[p + 1]
        n = e - s
        if n == 0:
            continue
        Gs = np.empty((n, n), dtype=np.float64)
        bs = np.empty(n, dtype=np.float64)
        w = np.empty(n, dtype=np.float64)
        gw = np.empty(n, dtype=np.float64)
        for a in range(n):
            ia = indices[s + a]
            bs[a] = B[p, ia]
            for c in range(n):
                Gs[a, c] = G[ia, indices[s + c]]
        yn2 = _projected_energy(Gs, bs)
        _cd_pixel(Gs, bs, yn2, lam[p], w, gw, tol, max_sweeps)
        for a in range(n):
            values[s + a] = w[a]


def solve_nnqp(
    G: np.ndarray,
    B: np.ndarray,
    lam: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    rel_gap_tol: float = REL_GAP_TOL,
    max_sweeps: int = MAX_SWEEPS,
) -> np.ndarray:
    """Solve every pixel's non-negative QP; ``values (nnz,) float64`` aligned with ``indices``.

    ``G (K, K)``, ``B (P, K)`` and ``lam (P,)`` are float64; ``indptr (P + 1,)`` and
    ``indices (nnz,)`` int64 list each pixel's allowed units (the CSR of ``sub`` over the
    active pixels). A pixel with no allowed unit contributes nothing.
    """
    G = np.ascontiguousarray(G, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if B.shape[0] != indptr.size - 1 or lam.size != indptr.size - 1:
        raise ValueError("B, lam and indptr disagree on the number of pixels")
    values = np.zeros(indices.size, dtype=np.float64)
    _cd_all(G, B, lam, indptr, indices, values, float(rel_gap_tol), int(max_sweeps))
    return values
