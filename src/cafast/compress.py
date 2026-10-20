"""Pass 2 per tile: streamed randomized SVD with the temporal-autocorrelation rank cut.

A tile is held as (T, P) uint8 and used in float32 chunks of ``chunk`` frames (mean
removed), so the float copy never exceeds ``chunk * P`` values. Every product with
``Y`` (P x T) is a sum over chunks of BLAS sgemm calls.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
from scipy.linalg import blas as sblas

from cafast.factors import TileFactors

N_OVERSAMPLES = 10
MAX_RANK = 300  # a 64 x 64 tile dense with cells passed 176 components at 140k frames
NOISE_PX_CHUNK = 128
ADAPT_MARGIN = 10
SVD_CHUNK = 4096


# ---------------------------------------------------------------------------
# Streamed tile matrix
# ---------------------------------------------------------------------------


def _qr(A: np.ndarray) -> np.ndarray:
    return np.asfortranarray(sla.qr(A, mode="economic", overwrite_a=True)[0])


def gemm_blas(A: np.ndarray, B: np.ndarray, trans_a: bool = False) -> np.ndarray:
    """scipy's BLAS sgemm; ``A`` should be Fortran-ordered so no copy is made."""
    return sblas.sgemm(1.0, A, B, trans_a=trans_a)


class TileMatrix:
    """``Y`` (P x T, mean removed) of one tile, held as (T, P) uint8, used in float chunks."""

    def __init__(self, raw: np.ndarray, chunk: int = SVD_CHUNK) -> None:
        self.raw = raw.reshape(raw.shape[0], -1)  # (T, P) uint8
        self.T, self.P = self.raw.shape
        self.chunk = chunk
        acc = np.zeros(self.P)
        for c0 in range(0, self.T, chunk):
            acc += self.raw[c0 : c0 + chunk].sum(axis=0, dtype=np.float64)
        self.mean = (acc / self.T).astype(np.float32)
        energy = np.zeros(self.P)
        for c0 in range(0, self.T, chunk):
            d = self.raw[c0 : c0 + chunk].astype(np.float32)
            d -= self.mean
            energy += np.einsum("tp,tp->p", d, d, dtype=np.float64)
        self.energy = energy

    def _block(self, c0: int) -> np.ndarray:
        """Frames c0.. as (P, n) F-ordered float32 = (n, P) C-ordered, mean removed."""
        x = self.raw[c0 : c0 + self.chunk].astype(np.float32)
        x -= self.mean
        return x.T

    def t_matmul(self, Q: np.ndarray) -> np.ndarray:
        """``Y^T Q``: (T, ell)."""
        out = np.empty((self.T, Q.shape[1]), dtype=np.float32, order="F")
        for c0 in range(0, self.T, self.chunk):
            Xt = self._block(c0)
            out[c0 : c0 + Xt.shape[1]] = gemm_blas(Xt, Q, True)
        return out

    def matmul(self, Z: np.ndarray) -> np.ndarray:
        """``Y Z``: (P, ell)."""
        acc = np.zeros((self.P, Z.shape[1]), dtype=np.float32, order="F")
        for c0 in range(0, self.T, self.chunk):
            Xt = self._block(c0)
            acc += gemm_blas(Xt, np.asfortranarray(Z[c0 : c0 + Xt.shape[1]]), False)
        return acc

    def noise(self, noise_freq: float = 0.06) -> np.ndarray:
        """Per-pixel noise std from the power spectrum above ``noise_freq`` (Minian's sn).

        Minian's ``get_noise_fft`` with ``noise_method="logmexp"``: the square root of the
        geometric mean of the periodogram ``|rfft(y)|^2 / T`` over bins
        ``[round(noise_freq T), round(T / 2))``. The geometric mean of white-noise periodogram
        bins (exponential) is ``exp(-gamma) sigma^2``, so this is ``0.75 sigma`` on white noise;
        kept unscaled so ``sparse_snr`` compares with Minian's ``sparse_penal``. Unlike a median of
        integer frame differences it is not quantized on uint8 frames.
        """
        lo, hi = np.around(np.array([noise_freq, 0.5]) * self.T).astype(int)
        lo = max(lo, 1)
        if hi <= lo:  # too few frames for a spectrum: fall back to the difference MAD
            d = np.abs(np.diff(self.raw.astype(np.float32), axis=0))
            return 1.4826 * np.median(d, axis=0) / np.sqrt(2.0) if len(d) else np.zeros(self.P)
        eps = np.finfo(np.float32).eps
        out = np.empty(self.P)
        for p0 in range(0, self.P, NOISE_PX_CHUNK):
            x = self.raw[:, p0 : p0 + NOISE_PX_CHUNK].astype(np.float32)
            psd = np.abs(sfft.rfft(x, axis=0)[lo:hi]) ** 2 / self.T
            out[p0 : p0 + x.shape[1]] = np.exp(0.5 * np.log(psd + eps).mean(axis=0))
        return out


# ---------------------------------------------------------------------------
# Randomized SVD and rank cut
# ---------------------------------------------------------------------------


def rsvd_streamed(
    Y: TileMatrix, k: int, n_iter: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko randomized SVD of ``Y`` with every product streamed over frame chunks."""
    ell = k + N_OVERSAMPLES
    omega = np.asfortranarray(rng.standard_normal((Y.T, ell), dtype=np.float32))
    Q = _qr(Y.matmul(omega))
    for _ in range(n_iter):
        Z = _qr(Y.t_matmul(Q))
        Q = _qr(Y.matmul(Z))
    Bt = Y.t_matmul(Q)  # (T, ell) = (Q^T Y)^T
    Ub, s, Vt = sla.svd(Bt.T, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def lag1_autocorr(Vt: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each temporal component (rows of ``Vt``)."""
    v = Vt - Vt.mean(axis=1, keepdims=True)
    num = np.einsum("it,it->i", v[:, 1:], v[:, :-1], dtype=np.float64)
    den = np.einsum("it,it->i", v, v, dtype=np.float64)
    return num / np.maximum(den, 1e-30)


def autocorr_keep(Vt: np.ndarray, z: float = 4.0, rho_min: float = 0.2) -> np.ndarray:
    """Keep components whose time course is not (mostly) white noise.

    The median denoise and the tophat are spatial filters: they make the noise spatially
    correlated (which breaks the Marchenko-Pastur edge) but leave it white in time,
    while calcium transients, background drift and motion residue are all temporally
    correlated. Under white noise the lag-1 autocorrelation is ~N(0, 1/T), so a
    component must exceed ``z / sqrt(T)`` (z = 4: ~3e-5 false keeps per test). A
    component that mixes signal (rho ~0.9 for calcium at 30 Hz) with white noise has
    rho ~ (signal fraction) x 0.9, so ``rho_min`` 0.2 also drops components that are
    less than ~1/4 signal; unlike the z-test it does not loosen as T grows.
    """
    thr = max(z / np.sqrt(Vt.shape[1]), rho_min)
    return lag1_autocorr(Vt) > thr


# ---------------------------------------------------------------------------
# One tile
# ---------------------------------------------------------------------------


def compress_tile(
    raw: np.ndarray,
    k0: int = 40,
    n_iter: int = 1,
    rho_min: float = 0.2,
    z: float = 4.0,
    seed: int = 0,
    chunk: int = SVD_CHUNK,
    noise_freq: float = 0.06,
) -> TileFactors:
    """(T, th, tw) uint8 -> ``TileFactors``.

    Start at rank ``k0``; while any of the last ``ADAPT_MARGIN`` components still passes
    the autocorrelation cut, double k (up to ``min(MAX_RANK, P - 1, T - 1)``) and redo.
    If no component passes, keep component 0 only.
    """
    Y = TileMatrix(raw, chunk)
    sigma = Y.noise(noise_freq)
    rng = np.random.default_rng(seed)
    k_cap = max(1, min(MAX_RANK, Y.P - 1, Y.T - 1))
    k = min(k0, k_cap)
    while True:
        U, s, Vt = rsvd_streamed(Y, k, n_iter, rng)
        keep = autocorr_keep(Vt, z, rho_min)
        if k >= k_cap or not keep[max(k - ADAPT_MARGIN, 0) :].any():
            break
        k = min(2 * k, k_cap)
    if not keep.any():
        keep[0] = True
    return TileFactors(
        W=np.ascontiguousarray(U[:, keep] * s[keep], dtype=np.float32),
        V=np.ascontiguousarray(Vt[keep].T, dtype=np.float32),
        mean=Y.mean,
        sigma=sigma.astype(np.float32),
        energy=Y.energy,
    )
