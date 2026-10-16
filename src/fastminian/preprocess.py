"""Frame preprocessing: median denoise and square white tophat (background removal)."""

from __future__ import annotations

import cv2
import numpy as np


def denoise(frames: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Per-frame median blur (uint8 input)."""
    return np.stack([cv2.medianBlur(f, ksize) for f in frames])


def tophat(frames: np.ndarray, wnd: int = 15) -> np.ndarray:
    """Per-frame white tophat with a (2*wnd+1) square kernel (separable, fast)."""
    kernel = np.ones((2 * wnd + 1,) * 2, np.uint8)
    return np.stack([cv2.morphologyEx(f, cv2.MORPH_TOPHAT, kernel) for f in frames])


def preprocess(frames: np.ndarray, ksize: int = 7, wnd: int = 15) -> np.ndarray:
    """Median denoise followed by tophat."""
    return tophat(denoise(frames, ksize), wnd)
