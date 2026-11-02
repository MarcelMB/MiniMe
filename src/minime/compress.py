"""Pass 2 per tile: streamed randomized SVD with the temporal-autocorrelation rank cut.

A tile is held as (T, P) uint8 and used in float32 chunks of ``chunk`` frames (mean
removed), so the float copy never exceeds ``chunk * P`` values. Every product with
``Y`` (P x T) is a sum over chunks of BLAS sgemm calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
from scipy.linalg import blas as sblas

from minime.factors import TileFactors

N_OVERSAMPLES = 10
MAX_RANK = 300  # a 64 x 64 tile dense with cells passed 176 components at 140k frames
NOISE_PX_CHUNK = 128
ADAPT_MARGIN = 10
SVD_CHUNK = 4096
NOISE_SEG = 1000  # frames per segment of the subset noise estimate


# ---------------------------------------------------------------------------
# Streamed tile matrix
# ---------------------------------------------------------------------------


def _qr(A: np.ndarray) -> np.ndarray:
    return np.asfortranarray(sla.qr(A, mode="economic", overwrite_a=True)[0])


def gemm_blas(A: np.ndarray, B: np.ndarray, trans_a: bool = False) -> np.ndarray:
    """scipy's BLAS sgemm; ``A`` should be Fortran-ordered so no copy is made."""
    return sblas.sgemm(1.0, A, B, trans_a=trans_a)


def tile_matrix(raw: np.ndarray, chunk: int = SVD_CHUNK, device: str = "cpu") -> TileMatrix:
    """``TileMatrix`` of ``raw``; ``device="gpu"`` runs its products on the GPU (``minime.gpu``)."""
    if device == "gpu":
        from minime.gpu import WgpuTileMatrix

        return WgpuTileMatrix(raw, chunk)
    return TileMatrix(raw, chunk)


class TileMatrix:
    """``Y`` (P x T, mean removed) of one tile, held as (T, P) uint8, used in float chunks."""

    def __init__(self, raw: np.ndarray, chunk: int = SVD_CHUNK) -> None:
        self.raw = raw.reshape(raw.shape[0], -1)  # (T, P) uint8
        self.T, self.P = self.raw.shape
        self.chunk = chunk
        # one read of the tile: sum and sum of squares of the uint8 values are exact in
        # float64 (< 2^53 up to ~1.4e11 frames), so energy = sum y^2 - (sum y)^2 / T
        acc = np.zeros(self.P)
        sq = np.zeros(self.P)
        for c0 in range(0, self.T, chunk):
            x = self.raw[c0 : c0 + chunk].astype(np.float32)
            acc += x.sum(axis=0, dtype=np.float64)
            sq += np.einsum("tp,tp->p", x, x, dtype=np.float64)
        self.sum = acc  # (P,) float64, exact
        self.mean = (acc / self.T).astype(np.float32)
        self.energy = np.maximum(sq - acc * acc / self.T, 0.0)

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

    def noise(self, noise_freq: float = 0.06, frames: int = 0) -> np.ndarray:
        """Per-pixel noise std from the power spectrum above ``noise_freq`` (Minian's sn).

        Minian's ``get_noise_fft`` with ``noise_method="logmexp"``: the square root of the
        geometric mean of the periodogram ``|rfft(y)|^2 / T`` over bins
        ``[round(noise_freq T), round(T / 2))``. The geometric mean of white-noise periodogram
        bins (exponential) is ``exp(-gamma) sigma^2``, so this is ``0.75 sigma`` on white noise;
        kept unscaled so ``sparse_snr`` compares with Minian's ``sparse_penal``. Unlike a median of
        integer frame differences it is not quantized on uint8 frames.

        ``frames > 0`` (and fewer than ``T``) estimates it from ``frames // NOISE_SEG``
        contiguous segments of ``NOISE_SEG`` frames spread evenly over the session: the same
        geometric mean over the same frequency band, pooled over the segments' periodograms.
        On white noise each bin is exponential at any segment length, so the estimate keeps
        Minian's ``0.75 sigma`` scale; it costs ``frames / T`` of the full FFT.
        """
        n_seg = frames // NOISE_SEG
        if n_seg >= 1 and n_seg * NOISE_SEG < self.T:
            return self._noise_segments(noise_freq, n_seg)
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

    def _noise_segments(self, noise_freq: float, n_seg: int) -> np.ndarray:
        L = NOISE_SEG
        lo, hi = max(int(round(noise_freq * L)), 1), int(round(0.5 * L))
        starts = np.linspace(0, self.T - L, n_seg).round().astype(int)
        eps = np.finfo(np.float32).eps
        out = np.empty(self.P)
        for p0 in range(0, self.P, NOISE_PX_CHUNK):
            acc = 0.0
            for t0 in starts:
                x = self.raw[t0 : t0 + L, p0 : p0 + NOISE_PX_CHUNK].astype(np.float32)
                psd = np.abs(sfft.rfft(x, axis=0)[lo:hi]) ** 2 / L
                acc = acc + np.log(psd + eps).mean(axis=0, dtype=np.float64)
            out[p0 : p0 + x.shape[1]] = np.exp(0.5 * acc / n_seg)
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
    noise_frames: int = 0,
    device: str = "cpu",
) -> TileFactors:
    """(T, th, tw) uint8 -> ``TileFactors``.

    ``device="gpu"``: the products of the randomized SVD run on the GPU (``minime.gpu``).

    Start at rank ``k0``; while any of the last ``ADAPT_MARGIN`` components still passes
    the autocorrelation cut, double k (up to ``min(MAX_RANK, P - 1, T - 1)``) and redo.
    If no component passes, keep component 0 only.
    """
    Y = tile_matrix(raw, chunk, device)
    sigma = Y.noise(noise_freq, noise_frames)
    rng = np.random.default_rng(seed)
    U, s, Vt, keep = _adaptive_rsvd(Y, k0, n_iter, rho_min, z, rng)
    if not keep.any():
        keep[0] = True
    return TileFactors(
        W=np.ascontiguousarray(U[:, keep] * s[keep], dtype=np.float32),
        V=np.ascontiguousarray(Vt[keep].T, dtype=np.float32),
        mean=Y.mean,
        sigma=sigma.astype(np.float32),
        energy=Y.energy,
    )


def _adaptive_rsvd(
    Y: TileMatrix, k0: int, n_iter: int, rho_min: float, z: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``rsvd_streamed`` from rank ``k0``, doubled while the last ``ADAPT_MARGIN``
    components still pass the autocorrelation cut; ``(U, s, Vt, keep)``."""
    k_cap = max(1, min(MAX_RANK, Y.P - 1, Y.T - 1))
    k = min(k0, k_cap)
    while True:
        U, s, Vt = rsvd_streamed(Y, k, n_iter, rng)
        keep = autocorr_keep(Vt, z, rho_min)
        if k >= k_cap or not keep[max(k - ADAPT_MARGIN, 0) :].any():
            return U, s, Vt, keep
        k = min(2 * k, k_cap)


# ---------------------------------------------------------------------------
# One tile in time windows
# ---------------------------------------------------------------------------


@dataclass
class _Window:
    t0: int
    t1: int
    sum: np.ndarray  # (P,) float64
    energy: np.ndarray  # (P,) float64 about the window's own mean
    log_sigma: np.ndarray  # (P,) float64
    L: np.ndarray  # (P, r) float64: kept U s of the window's rSVD


def compress_tile_windows(
    read: Callable[[int, int], np.ndarray],
    T: int,
    window: int,
    k0: int = 40,
    n_iter: int = 1,
    rho_min: float = 0.2,
    z: float = 4.0,
    seed: int = 0,
    chunk: int = SVD_CHUNK,
    noise_freq: float = 0.06,
    noise_frames: int = 0,
    device: str = "cpu",
) -> TileFactors:
    """``compress_tile`` of a tile whose ``T`` frames come from ``read(t0, t1)`` -> (n, th, tw)
    uint8, holding at most ``window`` frames at a time (``window <= 0``: the whole tile).

    ``T <= window`` is ``compress_tile`` of the whole tile, bit for bit. Otherwise the tile is
    split into equal windows of at most ``window`` frames, each read three times:

    1. compressed like a tile (adaptive rank, autocorrelation cut at its own length); an SVD
       of the stacked kept ``U s`` and window-mean shifts gives a spatial basis ``U`` for the
       whole tile (``Y - m`` restricted to window w is ``U_w s_w V_w^T + (m_w - m) 1^T``);
    2. the tile is projected on it, ``P = (Y - m)^T U`` window by window, keeping only the
       Gram and lag-1 products of ``P``: the components of the projection ``U U^T (Y - m)``
       (eigenvectors of ``P^T P``) get ``compress_tile``'s autocorrelation cut and rank
       schedule at length ``T``, on time courses that carry each window's noise as a
       whole-tile rSVD's do (the window factors alone have none, and would pass far more);
    3. the kept components' time courses are formed, window by window.

    Mean and energy are exact; sigma pools the windows' log periodograms (weights T_w).
    """
    if window <= 0 or window >= T:
        return compress_tile(
            read(0, T), k0, n_iter, rho_min, z, seed, chunk, noise_freq, noise_frames, device
        )
    n = -(-T // window)
    edges = np.linspace(0, T, n + 1).round().astype(int)
    rng = np.random.default_rng(seed)
    wins = []
    for t0, t1 in zip(edges[:-1], edges[1:], strict=True):
        Y = tile_matrix(read(t0, t1), chunk, device)
        frames = int(round(noise_frames * Y.T / T)) if noise_frames else 0
        log_sigma = np.log(np.asarray(Y.noise(noise_freq, frames), dtype=np.float64))
        U, s, _, keep = _adaptive_rsvd(Y, k0, n_iter, rho_min, z, rng)
        L = (U[:, keep] * s[keep]).astype(np.float64)
        wins.append(_Window(t0, t1, Y.sum, Y.energy, log_sigma, L))
        del Y, U
    m = sum(w.sum for w in wins) / T
    energy = sum(w.energy + (w.t1 - w.t0) * (w.sum / (w.t1 - w.t0) - m) ** 2 for w in wins)
    sigma = np.exp(sum((w.t1 - w.t0) * w.log_sigma for w in wins) / T)
    shifts = [(w.sum / (w.t1 - w.t0) - m)[:, None] * np.sqrt(w.t1 - w.t0) for w in wins]
    Ub, s, _ = sla.svd(np.hstack([w.L for w in wins] + shifts), full_matrices=False)
    P = Ub.shape[0]
    k = max(1, min(MAX_RANK, P - 1, T - 1, int((s > s[0] * 1e-6).sum()) if s[0] > 0 else 1))
    Ub = np.asfortranarray(Ub[:, :k], dtype=np.float32)

    def project(w: _Window, basis: np.ndarray) -> np.ndarray:
        """``(Y - m)^T basis`` of window ``w``: (T_w, k) float64."""
        Y = tile_matrix(read(w.t0, w.t1), chunk, device)
        out = Y.t_matmul(basis).astype(np.float64)
        out += (Y.mean.astype(np.float64) - m) @ basis.astype(np.float64)
        return out

    G = np.zeros((k, k))
    G1 = np.zeros((k, k))
    total = np.zeros(k)
    ends = []
    for w in wins:
        Pw = project(w, Ub)
        G += Pw.T @ Pw
        G1 += Pw[1:].T @ Pw[:-1]
        total += Pw.sum(axis=0)
        ends.append((Pw[0].copy(), Pw[-1].copy()))
        del Pw
    for (first, _), (_, last) in zip(ends[1:], ends[:-1], strict=True):
        G1 += np.outer(first, last)
    lam, Z = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1]
    lam, Z = np.maximum(lam[order], 0.0), Z[:, order]
    good = lam > lam[0] * 1e-10 if lam[0] > 0 else np.arange(k) < 1
    lam, Z = lam[good], Z[:, good]
    C = Z / np.sqrt(np.maximum(lam, 1e-300))  # (k, k'): unit-norm time courses v = P C
    rho = _lag1_from_products(C, G, G1, total, ends[0][0], ends[-1][1], T)
    passes = rho > max(z / np.sqrt(T), rho_min)
    kk = min(k0, passes.size)
    while kk < passes.size and passes[max(kk - ADAPT_MARGIN, 0) : kk].any():
        kk = min(2 * kk, passes.size)
    keep = passes & (np.arange(passes.size) < kk)
    if not keep.any():
        keep[0] = True
    basis = np.asfortranarray(Ub.astype(np.float64) @ C[:, keep], dtype=np.float32)
    V = np.empty((T, int(keep.sum())), dtype=np.float32)
    for w in wins:
        V[w.t0 : w.t1] = project(w, basis)
    W = Ub.astype(np.float64) @ (Z[:, keep] * np.sqrt(lam[keep]))
    return TileFactors(
        W=np.ascontiguousarray(W, dtype=np.float32),
        V=V,
        mean=m.astype(np.float32),
        sigma=sigma.astype(np.float32),
        energy=energy,
    )


def _lag1_from_products(
    C: np.ndarray,
    G: np.ndarray,
    G1: np.ndarray,
    total: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    T: int,
) -> np.ndarray:
    """``lag1_autocorr`` of the time courses ``v = P C`` (columns of C) from ``G = P^T P``,
    ``G1 = sum_t p_t p_{t-1}^T``, the column sums and the first and last rows of ``P``."""
    vv = np.einsum("ik,ij,jk->k", C, G, C)
    lag = np.einsum("ik,ij,jk->k", C, G1, C)
    tot = total @ C
    mu = tot / T
    num = lag - mu * (2 * tot - first @ C - last @ C) + (T - 1) * mu * mu
    den = vv - T * mu * mu
    return num / np.maximum(den, 1e-30)
