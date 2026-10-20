"""Pass-1 preview on a short clip: raw, denoised, registration image and tophat frames.

``median_ksize``, ``tophat_wnd`` and ``motion_bandpass`` shape the factors, so a change needs a
full ``cafast run``; this shows their effect on the first ``n_frames`` frames in about a second.
The motion template is built from the clip itself (the full run uses ``template_frames``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from cafast import io, motion, preprocess
from cafast.params import Params
from cafast.pass1 import registration_image
from cafast.tune.figures import CAFAST, INK, MINIAN, _new, _save, _style, _title

N_FRAMES = 300


def read_clip(raw_dir: Path, n_frames: int = N_FRAMES) -> np.ndarray:
    """The first ``n_frames`` frames of the first segment, (n, H, W) uint8."""
    seg = io.list_segments(Path(raw_dir))[0]
    via_bgr = io.first_frame(seg, True)
    try:
        bgr = io.needs_bgr(io.first_frame(seg, False), via_bgr)
    except OSError:
        bgr = True
    return io.read_avi(seg, max_frames=n_frames, bgr=bgr)


def pass1_clip(frames: np.ndarray, p: Params) -> dict[str, np.ndarray]:
    """Each pass-1 step on ``frames``: raw, denoised, registration image, tophat, shifts."""
    den = preprocess.denoise(frames, p.median_ksize)
    reg = registration_image(den, p)
    tmpl = motion.make_template(reg[: p.template_frames], p.motion_down, p.motion_max_shift)
    sh = motion.estimate(reg, tmpl, p.motion_max_shift)
    if p.motion_before_tophat:
        top = preprocess.tophat(motion.apply_shifts(den, sh), p.tophat_wnd)
    else:
        top = motion.apply_shifts(reg, sh)
    return {"raw": frames, "denoised": den, "registration": reg, "tophat": top, "shifts": sh}


def _show(ax: Axes, img: np.ndarray, title: str) -> None:
    lo, hi = np.percentile(img, [1, 99.7])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    _title(ax, title)


def make_preview(raw_dir: Path, p: Params, out: Path, n_frames: int = N_FRAMES) -> Path:
    """Write the preview figure for parameters ``p`` to ``out``; returns ``out``."""
    st = pass1_clip(read_clip(raw_dir, n_frames), p)
    n = st["raw"].shape[0]
    mid = n // 2
    reg_name = (
        f"band-pass {p.motion_bandpass[0]:g}-{p.motion_bandpass[1]:g} px"
        if p.motion_before_tophat
        else f"tophat {p.tophat_wnd}"
    )
    titles = {
        "raw": "raw",
        "denoised": f"median {p.median_ksize}",
        "registration": f"registration image ({reg_name})",
        "tophat": f"motion-corrected, tophat {p.tophat_wnd}",
    }
    fig = _new(16, 9.5)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.45])
    for c, key in enumerate(titles):
        _show(fig.add_subplot(gs[0, c]), st[key][mid], f"frame {mid}: {titles[key]}")
        _show(fig.add_subplot(gs[1, c]), st[key].max(axis=0), f"max over {n} frames")
    ax = fig.add_subplot(gs[2, :])
    ax.plot(st["shifts"][:, 0], color=CAFAST, lw=1, label="y")
    ax.plot(st["shifts"][:, 1], color=MINIAN, lw=1, label="x")
    ax.set_xlabel("frame")
    ax.set_ylabel("shift (px)")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    _title(ax, "shifts against a template from this clip")
    fig.suptitle(
        f"Pass-1 preview, first {n} frames of {Path(raw_dir).name}", x=0.01, ha="left", color=INK
    )
    _save(fig, Path(out))
    return Path(out)
