"""Label-free unit checks: each unit is good or flagged with its reasons; score = good - flagged.

Footprint metrics are shared with the ``minime tune`` figures (``tune/figures.py`` imports them).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.spatial import cKDTree

from minime.init import centroids

SIZE_FRAC = 0.1  # footprint size = pixels above 10 % of peak
REASONS = ("low_pnr", "in_pieces", "bad_size", "duplicate")
_PAIR_CHUNK = 256  # pairs per batch in the duplicate check (bounds memory at long T)

# ---------------------------------------------------------------------------
# Footprint and trace metrics
# ---------------------------------------------------------------------------


def _row_of_entries(A: sps.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))


def size_above(A: sps.csr_matrix, frac: float = SIZE_FRAC) -> np.ndarray:
    """(K,) pixels of each footprint above ``frac`` of its own maximum."""
    A = sps.csr_matrix(A)
    if A.shape[0] == 0:
        return np.zeros(0, np.int64)
    peak = A.max(axis=1).toarray().ravel()
    row = _row_of_entries(A)
    keep = A.data > frac * peak[row]
    return np.bincount(row[keep], minlength=A.shape[0])


def footprint_image(A: sps.csr_matrix, k: int, H: int, W: int) -> np.ndarray:
    return np.asarray(A[k].toarray(), np.float32).reshape(H, W)


def peak_mask(img: np.ndarray, frac: float) -> np.ndarray:
    peak = img.max()
    return ((img > frac * peak) & (peak > 0)).astype(np.uint8)


def n_parts(A: sps.csr_matrix, H: int, W: int, frac: float = SIZE_FRAC) -> np.ndarray:
    """(K,) connected pieces (8-connected) of each footprint above ``frac`` of its peak."""
    A = sps.csr_matrix(A)
    out = np.zeros(A.shape[0], np.int64)
    for k in range(A.shape[0]):
        mask = peak_mask(footprint_image(A, k, H, W), frac)
        out[k] = cv2.connectedComponents(mask, connectivity=8)[0] - 1
    return out


def trace_pnr(C: np.ndarray) -> np.ndarray:
    """(K,) peak-to-noise of traces: (99.9th - 0.1th percentile) / noise std from diffs.

    Diffs between two frames that are both 0 are left out of the noise: ``YrA`` is clipped
    at 0, and a trace at 0 most of the time would otherwise get noise 0 and score inf.
    A flat trace scores 0; a trace with range but no diff noise scores inf.
    """
    if C.shape[0] == 0:
        return np.zeros(0)
    lo, hi = np.percentile(C, [0.1, 99.9], axis=1)
    d = np.abs(np.diff(C, axis=1)).astype(np.float64)
    d[(C[:, 1:] == 0) & (C[:, :-1] == 0)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan row: no noise, as flat
        noise = np.nan_to_num(np.nanmedian(d, axis=1)) / 0.6745 / np.sqrt(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(noise > 0, (hi - lo) / noise, np.where(hi > lo, np.inf, 0.0))


def main_piece_frac(A: sps.csr_matrix, H: int, W: int, frac: float = SIZE_FRAC) -> np.ndarray:
    """(K,) share of the weight above ``frac`` of peak in the largest 8-connected piece.

    0 for a unit with no pixels.
    """
    A = sps.csr_matrix(A)
    out = np.zeros(A.shape[0])
    for k in range(A.shape[0]):
        img = footprint_image(A, k, H, W)
        n, lab = cv2.connectedComponents(peak_mask(img, frac), connectivity=8)
        if n <= 1:
            continue
        w = np.bincount(lab.ravel(), weights=img.ravel(), minlength=n)[1:]
        out[k] = w.max() / w.sum()
    return out


def _pair_corr(Z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Pearson corr of rows ``i`` and ``j`` of the centred, unit-norm ``Z``, in chunks."""
    out = np.empty(len(i))
    for a in range(0, len(i), _PAIR_CHUNK):
        b = a + _PAIR_CHUNK
        out[a:b] = np.einsum("ij,ij->i", Z[i[a:b]], Z[j[a:b]])
    return out


def duplicates(
    cent: np.ndarray, traces: np.ndarray, pnr: np.ndarray, radius: float, thres: float
) -> np.ndarray:
    """(K,) True where a unit closer than ``radius`` with trace corr >= ``thres`` has a higher PNR.

    Equal PNR: the higher index is flagged. Units with a NaN centroid are never paired.
    """
    K = len(cent)
    dup = np.zeros(K, bool)
    ok = np.flatnonzero(np.isfinite(cent).all(axis=1))
    if len(ok) < 2:
        return dup
    pairs = cKDTree(cent[ok]).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return dup
    i, j = ok[pairs[:, 0]], ok[pairs[:, 1]]
    X = np.asarray(traces, np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Z = np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0)  # flat rows -> corr 0
    hit = _pair_corr(Z, i, j) >= thres
    i, j = i[hit], j[hit]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    dup[np.where(pnr[lo] >= pnr[hi], hi, lo)] = True
    return dup


# ---------------------------------------------------------------------------
# Checks and score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checks:
    """Thresholds of the four unit checks; sizes scale with the cell ``diameter`` (px)."""

    diameter: float
    pnr_min: float = 12.0  # trace_pnr scale: pure white noise scores ~6.2
    piece_frac: float = 0.9
    size_lo: float = 0.3  # x cell area
    size_hi: float = 3.0
    dup_corr: float = 0.8

    def __post_init__(self) -> None:
        if not (isinstance(self.diameter, int | float) and self.diameter > 0):
            raise ValueError(f"diameter must be a number of px > 0, got {self.diameter!r}")

    @property
    def cell_area(self) -> float:
        return math.pi * (self.diameter / 2) ** 2


def unit_flags(
    A: sps.csr_matrix, traces: np.ndarray, H: int, W: int, checks: Checks
) -> pd.DataFrame:
    """One row per unit: metrics, one bool column per reason in ``REASONS``, and ``good``."""
    A = sps.csr_matrix(A)
    K = A.shape[0]
    traces = np.asarray(traces)
    if traces.shape[0] != K:
        raise ValueError(f"{traces.shape[0]} traces for {K} footprints")
    pnr = trace_pnr(traces)
    size10 = size_above(A)
    with np.errstate(invalid="ignore", divide="ignore"):
        cent = centroids(A, H, W) if K else np.zeros((0, 2))
    a = checks.cell_area
    df = pd.DataFrame(
        {
            "unit_id": np.arange(K),
            "pnr": pnr,
            "size10": size10,
            "main_piece": main_piece_frac(A, H, W),
        }
    )
    df["low_pnr"] = pnr < checks.pnr_min
    df["in_pieces"] = df["main_piece"] < checks.piece_frac
    df["bad_size"] = (size10 < checks.size_lo * a) | (size10 > checks.size_hi * a)
    df["duplicate"] = duplicates(cent, traces, pnr, checks.diameter, checks.dup_corr)
    df["good"] = ~df[list(REASONS)].any(axis=1)
    return df


def summarize(flags: pd.DataFrame) -> dict[str, int]:
    """Counts of good and flagged units, the score (good - flagged) and each reason."""
    good = int(flags["good"].sum())
    flagged = len(flags) - good
    return {
        "good": good,
        "flagged": flagged,
        "score": good - flagged,
        **{r: int(flags[r].sum()) for r in REASONS},
    }
