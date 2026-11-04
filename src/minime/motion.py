"""Rigid motion: template cross-correlation estimate, warp, quality score, sidecar file.

Shift convention: ``shifts[t] = (dy, dx)`` as returned by ``estimate``;
``apply_shifts`` warps with ``M = [[1, 0, -dx], [0, 1, -dy]]``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import minime

FORMAT_VERSION = 1
META_FILE = "meta.json"
ARRAY_FILE = "arrays.npz"


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def _down2(frames: np.ndarray) -> np.ndarray:
    return np.stack(
        [cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for f in frames]
    )


def _hann(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _prep(frames: np.ndarray, down: int) -> np.ndarray:
    """Optionally 2x area-downsample, remove the per-frame mean, Hann-window (float32)."""
    f = _down2(frames) if down == 2 else frames
    f = f.astype(np.float32)
    f -= f.mean(axis=(1, 2), keepdims=True)
    return f * _hann(*f.shape[1:])[None]


def _parabolic(c_m: np.ndarray, c_0: np.ndarray, c_p: np.ndarray) -> np.ndarray:
    """Sub-pixel peak offset from three samples, clipped to +-0.5."""
    den = c_m - 2 * c_0 + c_p
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-12, 0.5 * (c_m - c_p) / den, 0.0)
    return np.clip(off, -0.5, 0.5)


@dataclass(frozen=True)
class Template:
    fft: np.ndarray
    down: int
    shape: tuple[int, int]  # full-resolution (h, w) of the frames it was built from
    # (h, w) of the prepared (downsampled) template image; cv2.resize rounds h / 2 up
    prep_shape: tuple[int, int]
    # (h, w) float32 mean tophat image of the template frames in template coordinates: what
    # ``motion_fill = "template"`` puts where a shifted frame has no data
    fill: np.ndarray | None = None


def _peaks(corr: np.ndarray, max_shift: int, down: int) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel (dy, dx) of each (h, w) circular correlation's peak within ``max_shift``
    (full-resolution px), and the peak value."""
    n, h, w = corr.shape
    m = max_shift // down
    fy, fx = np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w
    outside = (np.abs(fy)[:, None] > m) | (np.abs(fx)[None] > m)
    corr[:, outside] = -np.inf
    iy, ix = np.unravel_index(corr.reshape(n, -1).argmax(axis=1), (h, w))
    corr[:, outside] = 0.0
    i = np.arange(n)
    sy = _parabolic(corr[i, (iy - 1) % h, ix], corr[i, iy, ix], corr[i, (iy + 1) % h, ix])
    sx = _parabolic(corr[i, iy, (ix - 1) % w], corr[i, iy, ix], corr[i, iy, (ix + 1) % w])
    dy = np.where(iy > h // 2, iy - h, iy) + sy
    dx = np.where(ix > w // 2, ix - w, ix) + sx
    return down * np.stack([dy, dx], axis=1), corr[i, iy, ix]


def estimate_peak(
    frames: np.ndarray, tmpl: Template, max_shift: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """``estimate`` plus each frame's normalised correlation peak (n,), a reliability score:
    about the frame's quality at the estimated shift, near 0 for a black or corrupted frame."""
    f = _prep(frames, tmpl.down)
    n, h, w = f.shape
    corr = np.fft.irfft2(np.fft.rfft2(f) * np.conj(tmpl.fft)[None], s=(h, w))
    sh, peak = _peaks(corr, max_shift, tmpl.down)
    den = np.linalg.norm(f.reshape(n, -1), axis=1) * np.linalg.norm(template_image(tmpl))
    with np.errstate(divide="ignore", invalid="ignore"):
        return sh, np.where(den > 0, peak / den, 0.0)


def estimate(frames: np.ndarray, tmpl: Template, max_shift: int = 20) -> np.ndarray:
    """Rigid (dy, dx) per frame vs the template, as (n, 2) float64."""
    f = _prep(frames, tmpl.down)
    n, h, w = f.shape
    corr = np.fft.irfft2(np.fft.rfft2(f) * np.conj(tmpl.fft)[None], s=(h, w))
    return _peaks(corr, max_shift, tmpl.down)[0]


def template_offset(a: Template, b: Template, max_shift: int = 20) -> np.ndarray:
    """(dy, dx) that ``estimate`` would give a frame looking like template ``a`` vs ``b``."""
    h, w = b.prep_shape
    corr = np.fft.irfft2(a.fft * np.conj(b.fft), s=(h, w))[None]
    return _peaks(corr, max_shift, b.down)[0][0]


def refine(reg: np.ndarray, shifts: np.ndarray, tmpl: Template, max_shift: int = 20) -> np.ndarray:
    """Shifts of the batch ``reg`` re-estimated against the batch's own mean (its frames warped
    by ``shifts``), which matches their current look (focus, bleaching, activity) better than
    a template from another part of the session; the batch template is tied back to ``tmpl``
    by its own offset, so the result stays in ``tmpl``'s coordinates."""
    b = _mean_template(apply_shifts(reg, shifts), tmpl.down)
    return estimate(reg, b, max_shift) + template_offset(b, tmpl, max_shift)


FILLS = {"zero": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE}


def _valid_rects(shifts: np.ndarray, h: int, w: int) -> tuple[np.ndarray, ...]:
    """Per frame, rows ``[y0, y1)`` and columns ``[x0, x1)`` that ``apply_shifts`` reads from
    inside the frame (source ``(y + dy, x + dx)`` within ``[0, h - 1] x [0, w - 1]``)."""
    s = np.asarray(shifts, np.float64).reshape(-1, 2)
    y0 = np.clip(np.ceil(-s[:, 0]), 0, h).astype(np.intp)
    y1 = np.clip(np.floor(h - 1 - s[:, 0]) + 1, 0, h).astype(np.intp)
    x0 = np.clip(np.ceil(-s[:, 1]), 0, w).astype(np.intp)
    x1 = np.clip(np.floor(w - 1 - s[:, 1]) + 1, 0, w).astype(np.intp)
    return y0, y1, x0, x1


def fill_outside(frames: np.ndarray, shifts: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """``frames`` (warped by ``shifts``) with the pixels read from outside the frame set to
    ``fill`` (h, w), in place."""
    n, h, w = frames.shape
    f = fill.astype(frames.dtype) if frames.dtype == np.uint8 else fill
    for i, (y0, y1, x0, x1) in enumerate(zip(*_valid_rects(shifts, h, w), strict=True)):
        frames[i, :y0] = f[:y0]
        frames[i, y1:] = f[y1:]
        frames[i, :, :x0] = f[:, :x0]
        frames[i, :, x1:] = f[:, x1:]
    return frames


def fill_image(frames: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """(h, w) float32 mean of ``frames`` warped by ``shifts`` over the frames that hold data
    at each pixel (the edge-replicated mean where none does)."""
    n, h, w = frames.shape
    warped = apply_shifts(frames.astype(np.float32), shifts, "replicate")
    inside = np.zeros((n, h, w), bool)
    for i, (y0, y1, x0, x1) in enumerate(zip(*_valid_rects(shifts, h, w), strict=True)):
        inside[i, y0:y1, x0:x1] = True
    cnt = inside.sum(axis=0)
    mean = np.where(inside, warped, 0).sum(axis=0) / np.maximum(cnt, 1)
    return np.where(cnt > 0, mean, warped.mean(axis=0)).astype(np.float32)


def apply_shifts(frames: np.ndarray, shifts: np.ndarray, fill: str = "zero") -> np.ndarray:
    """Warp each frame by its (dy, dx); works for uint8 and float32 input. ``fill``: what the
    pixels read from outside the frame get, 0 or the nearest edge pixel ("replicate")."""
    n, h, w = frames.shape
    out = np.empty_like(frames)
    border = FILLS[fill]
    for i in range(n):
        m = np.array([[1, 0, -shifts[i, 1]], [0, 1, -shifts[i, 0]]], dtype=np.float32)
        out[i] = cv2.warpAffine(frames[i], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)
    return out


def coverage(shifts: np.ndarray, h: int, w: int) -> np.ndarray:
    """(h, w) float32: fraction of frames whose ``apply_shifts`` reads the pixel from inside
    the frame (source position ``(y + dy, x + dx)`` within ``[0, h - 1] x [0, w - 1]``); the
    rest of the frames hold the warp's zero fill there."""
    s = np.asarray(shifts, np.float64).reshape(-1, 2)
    if not len(s):
        return np.ones((h, w), np.float32)
    y0, y1, x0, x1 = _valid_rects(s, h, w)  # [y0, y1) x [x0, x1)
    keep = (y1 > y0) & (x1 > x0)
    y0, y1, x0, x1 = y0[keep], y1[keep], x0[keep], x1[keep]
    d = np.zeros((h + 1, w + 1), np.int64)  # 2-D difference array of the valid rectangles
    np.add.at(d, (y0, x0), 1)
    np.add.at(d, (y0, x1), -1)
    np.add.at(d, (y1, x0), -1)
    np.add.at(d, (y1, x1), 1)
    return (d.cumsum(0).cumsum(1)[:h, :w] / len(s)).astype(np.float32)


def _mean_template(pre_frames: np.ndarray, down: int) -> Template:
    shape = (int(pre_frames.shape[1]), int(pre_frames.shape[2]))
    img = _prep(pre_frames, down).mean(axis=0)
    return Template(np.fft.rfft2(img), down, shape, (int(img.shape[0]), int(img.shape[1])))


def _best_chunk(pre_frames: np.ndarray, chunks: int, down: int) -> np.ndarray:
    """Frames of the chunk (of ``chunks`` equal consecutive runs) whose mean correlates best
    with the other chunks' means: a sharp start for frames sampled across a session."""
    parts = np.array_split(pre_frames, chunks)
    means = np.stack([_prep(p, down).mean(axis=0).ravel() for p in parts])
    means -= means.mean(axis=1, keepdims=True)
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-12
    c = means @ means.T
    return parts[int(np.argmax(c.sum(axis=1) - np.diag(c)))]


def make_template(
    pre_frames: np.ndarray,
    down: int = 2,
    max_shift: int = 20,
    iters: int = 1,
    chunks: int = 1,
    centre: str = "mean",
) -> Template:
    """Mean template, re-centred: frames are aligned to the blurred mean, then re-averaged.

    ``iters``: align-and-average rounds. ``chunks > 1`` (frames sampled in that many runs
    across a session): start from the run that agrees best with the others, not from the mean
    of all, which blurs when the runs sit at different positions. ``centre="median"``: put
    the template at the frames' median position (shifts minus their median) instead of the
    position the alignment lands on. The defaults are the first-frames template.
    """
    t = _mean_template(_best_chunk(pre_frames, chunks, down) if chunks > 1 else pre_frames, down)
    for _ in range(iters):
        s = estimate(pre_frames, t, max_shift)
        if centre == "median":
            s = s - np.median(s, axis=0)
        t = _mean_template(apply_shifts(pre_frames, s), down)
    return t


def recentre(
    pre_frames: np.ndarray, tmpl: Template, probe: np.ndarray, max_shift: int = 20
) -> tuple[Template, np.ndarray]:
    """``tmpl`` (built from ``pre_frames``) moved to the median position of the ``probe``
    frames (sampled across the session), and that median shift ``(dy, dx)``.

    Frames estimated against the result get ``shift - median`` (up to the rebuilt template's
    noise): the warp's zero fill is then split between opposite borders instead of piling up
    on the side the session drifted away from."""
    m = np.median(estimate(probe, tmpl, max_shift), axis=0)
    s = estimate(pre_frames, tmpl, max_shift) - m
    return _mean_template(apply_shifts(pre_frames, s), tmpl.down), m


def fix_outliers(
    shifts: np.ndarray, peak: np.ndarray, max_dev: float, rel_peak: float = 0.7
) -> np.ndarray:
    """``shifts`` with each unreliable estimate replaced by the median of the batch's
    reliable ones within 5 frames: a frame is unreliable when it is more than ``max_dev`` px
    from that local median and its correlation ``peak`` (``estimate_peak``) is below
    ``rel_peak`` times the batch median (a black or corrupted frame, whose peak is noise).
    Fast real motion keeps a normal peak and is left alone. ``max_dev <= 0``: unchanged."""
    s = np.asarray(shifts, np.float64)
    n = s.shape[0]
    if max_dev <= 0 or n < 3:
        return s
    low = peak < rel_peak * np.median(peak)
    out = s.copy()
    for i in range(n):
        lo, hi = max(0, i - 5), min(n, i + 6)
        ref = np.median(s[lo:hi], axis=0)
        if low[i] and np.hypot(*(s[i] - ref)) > max_dev:
            good = np.arange(lo, hi)[~low[lo:hi]]
            out[i] = np.median(s[good], axis=0) if good.size else ref
    return out


def template_image(tmpl: Template) -> np.ndarray:
    """The prepared template in image space, ``tmpl.prep_shape`` (about h/down, w/down) float32."""
    return np.fft.irfft2(tmpl.fft, s=tmpl.prep_shape).astype(np.float32)


def frame_quality(frames: np.ndarray, tmpl: Template) -> np.ndarray:
    """Pearson correlation over pixels between each prepared frame and the template, (n,)."""
    f = _prep(frames, tmpl.down)
    t = np.fft.irfft2(tmpl.fft, s=f.shape[1:])
    n = f.shape[0]
    fc = f.reshape(n, -1) - f.reshape(n, -1).mean(axis=1, keepdims=True)
    tc = t.ravel() - t.mean()
    den = np.linalg.norm(fc, axis=1) * np.linalg.norm(tc)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(den > 0, fc @ tc / den, 0.0)
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


def canonical_params(params: dict[str, Any]) -> str:
    """JSON with sorted keys so dict ordering / numpy scalars do not break equality."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set | tuple):
            return list(o)
        return str(o)

    return json.dumps(params, sort_keys=True, default=_default, separators=(",", ":"))


@dataclass
class MotionSidecar:
    shifts: np.ndarray
    template: np.ndarray
    quality: np.ndarray
    params: dict[str, Any]
    raw_hash: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.shifts.shape[0])


def save_sidecar(path: str | os.PathLike, sc: MotionSidecar) -> None:
    """Write ``meta.json`` + ``arrays.npz`` into directory ``path`` (atomic per file)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    shifts = np.ascontiguousarray(sc.shifts, dtype=np.float32)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError(f"shifts must be (T, 2), got {shifts.shape}")
    template = np.ascontiguousarray(sc.template, dtype=np.float32)
    quality = np.ascontiguousarray(sc.quality, dtype=np.float32)
    reserved = {"format_version", "raw_hash", "params"}
    meta = {
        **{k: v for k, v in sc.meta.items() if k not in reserved},
        "format_version": FORMAT_VERSION,
        "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "minime_version": minime.__version__,
        "n_frames": int(shifts.shape[0]),
        "template_shape": [int(x) for x in template.shape],
        "raw_hash": sc.raw_hash,
        "params": sc.params,
    }
    tmp_npz = path / (ARRAY_FILE + ".tmp")
    with open(tmp_npz, "wb") as f:
        np.savez(f, shifts=shifts, template=template, quality=quality)
    os.replace(tmp_npz, path / ARRAY_FILE)
    tmp_json = path / (META_FILE + ".tmp")
    tmp_json.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    os.replace(tmp_json, path / META_FILE)


def load_sidecar(path: str | os.PathLike) -> MotionSidecar:
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    if int(meta.get("format_version", -1)) > FORMAT_VERSION:
        raise ValueError(
            f"sidecar format {meta.get('format_version')} newer than supported {FORMAT_VERSION}"
        )
    with np.load(path / ARRAY_FILE) as z:
        shifts, template, quality = z["shifts"], z["template"], z["quality"]
    return MotionSidecar(
        shifts=shifts,
        template=template,
        quality=quality,
        params=dict(meta.get("params", {})),
        raw_hash=str(meta.get("raw_hash", "")),
        meta=meta,
    )


def validate(sc: MotionSidecar, raw_hash: str, params: dict[str, Any], n_frames: int) -> bool:
    """True if the stored shifts apply to this raw data, parameters and frame count."""
    return (
        int(sc.meta.get("format_version", -1)) == FORMAT_VERSION
        and sc.raw_hash == raw_hash
        and canonical_params(sc.params) == canonical_params(params)
        and sc.n_frames == n_frames
    )


def load_if_valid(
    path: str | os.PathLike, raw_hash: str, params: dict[str, Any], n_frames: int
) -> MotionSidecar | None:
    """The sidecar on a hit; ``None`` if missing, unreadable or stale."""
    path = Path(path)
    if not (path / META_FILE).exists():
        return None
    try:
        sc = load_sidecar(path)
    except Exception:  # corrupt / unreadable -> re-estimate
        return None
    return sc if validate(sc, raw_hash, params, n_frames) else None
