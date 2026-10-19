"""Frame preprocessing: median denoise and square white tophat (background removal)."""

from __future__ import annotations

import cv2
import numpy as np


def denoise(frames: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Per-frame median blur (uint8 input); ``ksize <= 1`` returns the frames unchanged."""
    if ksize <= 1:
        return frames
    return np.stack([cv2.medianBlur(f, ksize) for f in frames])


def tophat(frames: np.ndarray, wnd: int = 15) -> np.ndarray:
    """Per-frame white tophat with a (2*wnd+1) square kernel; ``wnd <= 0`` skips it."""
    if wnd <= 0:
        return frames
    kernel = np.ones((2 * wnd + 1,) * 2, np.uint8)
    return np.stack([cv2.morphologyEx(f, cv2.MORPH_TOPHAT, kernel) for f in frames])


def preprocess(frames: np.ndarray, ksize: int = 7, wnd: int = 15) -> np.ndarray:
    """Median denoise followed by tophat."""
    return tophat(denoise(frames, ksize), wnd)


def bandpass(frames: np.ndarray, lo: float = 1.0, hi: float = 6.0) -> np.ndarray:
    """Per-frame difference of Gaussians (sigma ``lo`` minus sigma ``hi`` px), float32.

    The registration image when motion is estimated before the tophat: it keeps static
    structure (vessels) that the tophat would remove and drops the smooth vignetting that
    would otherwise pull every shift toward zero. The wide blur runs at half resolution
    and is upsampled (3.4x faster at 600 x 600, correlation 0.99 with the exact DoG).
    """
    h, w = frames.shape[1:]
    out = np.empty(frames.shape, np.float32)
    for i, x in enumerate(frames):
        if x.dtype != np.uint8:
            x = x.astype(np.float32)
        half = cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        wide = cv2.GaussianBlur(half.astype(np.float32), (0, 0), hi / 2)
        out[i] = cv2.GaussianBlur(x, (0, 0), lo).astype(np.float32)
        out[i] -= cv2.resize(wide, (w, h), interpolation=cv2.INTER_LINEAR)
    return out
