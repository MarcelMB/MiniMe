"""Footprint outlines as polygons for the browser to draw."""

from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sps


def unit_polygons(
    A: sps.csr_matrix, H: int, W: int, rel: float = 0.2
) -> list[list[list[list[float]]]]:
    """Per unit, the outer outlines of its pixels >= ``rel`` x its peak, as [x, y] points.

    Points are pixel centres (x = column, y = row); a one-pixel island is a one-point
    polygon. The 0.2 threshold matches the stage viewer's outlines.
    """
    A = sps.csr_matrix(A)
    out: list[list[list[list[float]]]] = []
    for k in range(A.shape[0]):
        lo, hi = A.indptr[k], A.indptr[k + 1]
        idx, val = A.indices[lo:hi], A.data[lo:hi]
        if idx.size == 0 or val.max() <= 0:
            out.append([])
            continue
        m = np.zeros(H * W, np.uint8)
        m[idx[val >= rel * val.max()]] = 1
        cs, _ = cv2.findContours(m.reshape(H, W), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.append([c[:, 0, :].astype(float).tolist() for c in cs])
    return out
