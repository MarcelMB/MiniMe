"""Pass-1 preview of a raw session for the launcher: any frame through the preprocessing
steps, and a motion check on a short clip around it. Nothing is cached: one frame takes
milliseconds, a clip a few seconds."""

from __future__ import annotations

import cv2
import numpy as np

from minime import io, motion, preprocess
from minime.params import Params
from minime.pass1 import registration_image
from minime.serve.frames import raw_frame

PANELS = ("raw", "denoised", "registration", "tophat")
MOTION_FRAMES = 150


def _u8(a: np.ndarray) -> np.ndarray:
    """``a`` scaled to uint8 between its 1st and 99.7th percentile (display only)."""
    lo, hi = np.percentile(a, [1, 99.7])
    return np.clip((a - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)


def frame_panels(info: io.RawInfo, t: int, p: Params) -> np.ndarray:
    """(4, H, W) uint8: frame ``t`` raw, denoised, as the registration image and after the
    tophat (``PANELS``), with the pass-1 parameters of ``p``."""
    f = raw_frame(info, t).astype(np.uint8)[None]
    den = preprocess.denoise(f, p.median_ksize)
    reg = registration_image(den, p)
    top = preprocess.tophat(den, p.tophat_wnd)
    return np.stack([f[0], den[0], _u8(reg[0]), top[0]])


def clip_start(info: io.RawInfo, t: int, n: int) -> int:
    """First frame of an ``n``-frame clip around ``t`` that stays inside ``t``'s segment
    (a shorter clip when the segment has fewer than ``n`` frames)."""
    edges = np.cumsum([0, *info.segment_frames])
    if not 0 <= t < edges[-1]:
        raise IndexError(f"frame {t} is beyond the {edges[-1]} raw frames")
    s = int(np.searchsorted(edges, t, side="right") - 1)
    return int(max(edges[s], min(t, edges[s + 1] - n)))


def read_clip(info: io.RawInfo, t0: int, n: int) -> np.ndarray:
    """Up to ``n`` consecutive frames from ``t0`` within its segment, (m, H, W) uint8."""
    edges = np.cumsum([0, *info.segment_frames])
    s = int(np.searchsorted(edges, t0, side="right") - 1)
    cap = io._open(info.segments[s], info.bgr)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t0 - int(edges[s]))
        out = []
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            out.append(io._gray(f, info.bgr).copy())
    finally:
        cap.release()
    if not out:
        raise OSError(f"cannot read frame {t0} of {info.segments[s]}")
    return np.stack(out)


def motion_check(info: io.RawInfo, t: int, p: Params, n: int = MOTION_FRAMES) -> dict:
    """Rigid shifts of a clip around ``t`` against a template built from the clip itself."""
    t0 = clip_start(info, t, n)
    frames = read_clip(info, t0, n)
    reg = registration_image(preprocess.denoise(frames, p.median_ksize), p)
    tmpl = motion.make_template(reg, p.motion_down, p.motion_max_shift)
    sh = motion.estimate(reg, tmpl, p.motion_max_shift)
    return {
        "t0": t0,
        "frames": int(frames.shape[0]),
        "max_shift_px": float(np.abs(sh).max()),
        "quality_median": float(np.median(motion.frame_quality(reg, tmpl))),
        "shifts": np.round(sh, 2).tolist(),
    }
