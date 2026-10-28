"""Over-complete seed set from stored block maxima, plus PNR and GMM seed filters on factors.

``seeds_init`` reproduces Minian's rolling-window ``seeds_init`` exactly: each window's max
projection is the elementwise max over the stored per-block maxima that tile it, then
``local_max_roll`` (cv2 dilate/erode local maxima over disk kernels ``2..max_wnd-1``, union,
connected components, one pixel per component at the per-axis median) and a per-pixel count
over windows. ``pnr_refine`` and ``gmm_refine`` work on denoised traces reconstructed from the
factors.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from minime.factors import Factors

# ---------------------------------------------------------------------------
# Windows and blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """Half-open frame range ``[start, stop)``."""

    start: int
    stop: int


def rolling_windows(T: int, wnd_size: int, stp_size: int) -> list[Window]:
    """Minian's rolling windows as [start, stop) frame ranges (stop clipped to T)."""
    nstp = int(np.ceil(T / stp_size) + 1)
    centers = np.linspace(0, T - 1, nstp)
    hwnd = np.ceil(wnd_size / 2)
    return [
        Window(int(np.floor(c - hwnd).clip(0)), min(int(np.ceil(c + hwnd)), T)) for c in centers
    ]


def window_block_ranges(edges: np.ndarray, windows: list[Window]) -> list[tuple[int, int]]:
    """[i0, i1) block index range of each window (blocks i0..i1-1 tile the window exactly)."""
    out: list[tuple[int, int]] = []
    for w in windows:
        i0 = int(np.searchsorted(edges, w.start))
        i1 = int(np.searchsorted(edges, w.stop))
        if i0 >= len(edges) or i1 >= len(edges) or edges[i0] != w.start or edges[i1] != w.stop:
            raise ValueError("window edge missing from the stored block edges")
        out.append((i0, i1))
    return out


def window_projection(block_max: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Max projection of one window = elementwise max over its blocks."""
    if i1 - i0 == 1:
        return block_max[i0]
    return np.maximum.reduce(block_max[i0:i1], axis=0)


# ---------------------------------------------------------------------------
# Local maxima
# ---------------------------------------------------------------------------


def disk(radius: int) -> np.ndarray:
    """Binary disk structuring element of the given radius (uint8, (2r+1, 2r+1))."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def disk_kernels(k0: int, k1: int) -> list[np.ndarray]:
    return [disk(k) for k in range(k0, k1)]


def local_max_union(fm: np.ndarray, kernels: list[np.ndarray], diff: float) -> np.ndarray:
    """Union over kernels of Minian's ``local_extreme(fm, k, "max", diff)``, as uint8 0/1."""
    union = np.zeros(fm.shape, dtype=bool)
    for k in kernels:
        fm_max = cv2.dilate(fm, k)
        fm_min = cv2.erode(fm, k)
        union |= (fm == fm_max) & ((fm_max - fm_min) > diff)
    return union.astype(np.uint8)


def label_medians(lab: np.ndarray, nlab: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-label (rows, cols) medians for labels 1..nlab-1, exactly ``int(np.median(coords))``.

    Odd count: middle element of the sorted coordinates. Even count: floor of the mean of the
    two middle elements (coordinates are non-negative ints). Rows and columns are independent.
    """
    W = lab.shape[1]
    flat = lab.ravel()
    nz = np.flatnonzero(flat)
    lbl = flat[nz]
    rows = nz // W
    cols = nz % W
    counts = np.bincount(lbl, minlength=nlab)[1:]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    rows_sorted = rows[np.argsort(lbl, kind="stable")]
    cols_sorted = cols[np.lexsort((cols, lbl))]
    med_r = (rows_sorted[lo] + rows_sorted[hi]) // 2
    med_c = (cols_sorted[lo] + cols_sorted[hi]) // 2
    return med_r, med_c


def _local_max_roll(fm: np.ndarray, kernels: list[np.ndarray], diff: float) -> np.ndarray:
    lmax = local_max_union(fm, kernels, diff)
    nlab, lab = cv2.connectedComponents(lmax)
    out = np.zeros(fm.shape, dtype=np.uint8)
    if nlab <= 1:
        return out
    r, c = label_medians(lab, nlab)
    out[r, c] = 1
    return out


def local_max_roll(fm: np.ndarray, k0: int, k1: int, diff: float) -> np.ndarray:
    """Minian's ``local_max_roll``: uint8 image, 1 at one seed pixel per local-max component."""
    return _local_max_roll(fm, disk_kernels(k0, k1), diff)


def count_local_maxima(
    block_max: np.ndarray,
    ranges: list[tuple[int, int]],
    kernels: list[np.ndarray],
    diff: float,
) -> np.ndarray:
    """Per-pixel count of windows in which the pixel is a seed."""
    count = np.zeros(block_max.shape[1:], dtype=np.int64)
    for i0, i1 in ranges:
        count += _local_max_roll(window_projection(block_max, i0, i1), kernels, diff)
    return count


def seeds_from_count(count: np.ndarray) -> np.ndarray:
    """(n, 3) int64 (height, width, count) for pixels with count > 0, row-major order."""
    h, w = np.nonzero(count)
    return np.stack([h, w, count[h, w]], axis=1).astype(np.int64)


def seeds_init(
    block_max: np.ndarray,
    block_edges: np.ndarray,
    T: int,
    wnd_size: int,
    stp_size: int,
    max_wnd: int,
    diff_thres: int,
) -> pd.DataFrame:
    """Over-complete seeds (``height``, ``width``, ``seeds`` count) from stored block maxima."""
    if block_max.dtype == np.float64:  # cv2 morphology has no float64 path
        block_max = block_max.astype(np.float32)
    windows = rolling_windows(T, wnd_size, stp_size)
    ranges = window_block_ranges(block_edges, windows)
    count = count_local_maxima(block_max, ranges, disk_kernels(2, max_wnd), diff_thres)
    arr = seeds_from_count(count)
    return pd.DataFrame(arr, columns=["height", "width", "seeds"])


# ---------------------------------------------------------------------------
# Seed filters on factor traces
# ---------------------------------------------------------------------------


def ptp_q(X: np.ndarray, q: tuple[float, float]) -> np.ndarray:
    """Row-wise peak-to-peak value from percentiles ``q[0]`` and ``q[1]``."""
    lo, hi = np.percentile(X, [q[0], q[1]], axis=-1)
    return hi - lo


SEED_BATCH = 1024  # seeds whose (T,) traces are held at once (~160 MB at 40k frames)
PTP_MEM_BYTES = 2 * 1024**3  # traces + percentile copy held by all threads together


def seed_ptp(
    f: Factors,
    pixels: np.ndarray,
    q: tuple[float, float],
    batch: int = SEED_BATCH,
    threads: int | None = None,
) -> np.ndarray:
    """``ptp_q`` of the denoised traces of flat ``pixels``, ``batch`` traces at a time.

    All traces at once cost 4 * n * T bytes plus percentile copies (5.4 GB for 11k seeds
    of a 39541-frame session). Batches run on ``threads`` threads (default: all cores; the
    sgemm and the percentile partition release the GIL) and write disjoint slices of the
    output. The batch shrinks so that all threads together hold at most ``PTP_MEM_BYTES`` (a
    1024-seed batch is 0.9 GB of traces at 216k frames); like any change of batch size this
    moves the float32 traces by BLAS rounding only (``rtol`` 1e-6). The cost is seeds x
    frames, and the number of rolling-window seeds itself grows with the frames.
    """
    pixels = np.asarray(pixels, dtype=np.int64)
    out = np.empty(pixels.size, dtype=np.float64)
    if threads is None:
        threads = os.cpu_count() or 1
    threads = max(1, threads)
    batch = max(1, min(batch, PTP_MEM_BYTES // (threads * f.T * 8)))

    def _one(i: int) -> None:
        out[i : i + batch] = ptp_q(f.traces(pixels[i : i + batch]), q)

    starts = range(0, pixels.size, batch)
    if threads <= 1 or len(starts) <= 1:
        for i in starts:
            _one(i)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_one, starts))
    return out


def seed_pixels(f: Factors, seeds: pd.DataFrame) -> np.ndarray:
    """Flat pixel indices (int64) of the seeds."""
    return seeds["height"].to_numpy(np.int64) * f.W_ + seeds["width"].to_numpy(np.int64)


def seed_pnr(f: Factors, seeds: pd.DataFrame, q: tuple[float, float] = (0.1, 99.9)) -> np.ndarray:
    """(n,) peak-to-noise of each seed's denoised trace: ``ptp_q`` over ``6.18 sigma``.

    A pixel's value depends on the pixel only, so callers that try several seed thresholds
    can compute it once per pixel and pass it to ``pnr_refine``.
    """
    pixels = seed_pixels(f, seeds)
    sigma = f.sigma_flat()[pixels].astype(np.float64)
    bad = sigma <= 0
    if bad.any():
        sigma[bad] = np.median(f.sigma_flat())
    return seed_ptp(f, pixels, q) / (6.18 * sigma)


def pnr_refine(
    f: Factors,
    seeds: pd.DataFrame,
    thres: float | str = "auto",
    q: tuple[float, float] = (0.1, 99.9),
    pnr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Add ``pnr`` and ``mask_pnr``; ``thres="auto"`` keeps the upper component of a 2-GMM.

    ``pnr`` (n,) skips the trace read when the caller already has ``seed_pnr`` of ``seeds``.
    """
    if pnr is None:
        pnr = seed_pnr(f, seeds, q)
    elif len(pnr) != len(seeds):
        raise ValueError(f"{len(pnr)} pnr values for {len(seeds)} seeds")
    if isinstance(thres, str):
        if thres != "auto":
            raise ValueError(f"unknown pnr threshold {thres!r}")
        dat = np.log(np.maximum(pnr, np.finfo(np.float64).tiny))[:, None]
        gmm = GaussianMixture(n_components=2, random_state=0).fit(dat)
        mask = gmm.predict(dat) == np.argmax(gmm.means_.ravel())
    else:
        mask = pnr > thres
    if not mask.any():
        raise ValueError("no seeds survived pnr_refine")
    out = seeds.copy()
    out["pnr"] = pnr
    out["mask_pnr"] = mask
    return out


def gmm_refine(
    f: Factors,
    seeds: pd.DataFrame,
    q: tuple[float, float] = (0.1, 99.9),
    n_components: int = 2,
    valid_components: int = 1,
    mean_mask: bool = True,
) -> pd.DataFrame:
    """Add ``mask_gmm``: keep seeds in the top ``valid_components`` GMM components of the ptp.

    ``mean_mask`` additionally requires ptp above the mean of the lowest component. Minian writes
    ``dat > np.sort(gmm.means_)[0]``, which sorts a length-1 axis of the (n, 1) ``means_`` and so
    compares against whichever component sklearn labelled first; we use the true lowest mean,
    which is the documented intent and independent of component labelling.
    """
    ptp = seed_ptp(f, seed_pixels(f, seeds), q)
    dat = ptp.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(dat)
    idg = np.argsort(gmm.means_.ravel())[-valid_components:]
    valid = np.isin(gmm.predict(dat), idg)
    if mean_mask:
        valid &= ptp > np.sort(gmm.means_.ravel())[0]
    if not valid.any():
        raise ValueError("no seeds survived gmm_refine")
    out = seeds.copy()
    out["mask_gmm"] = valid
    return out
