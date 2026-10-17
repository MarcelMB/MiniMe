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

import cafast

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


def estimate(frames: np.ndarray, tmpl: Template, max_shift: int = 20) -> np.ndarray:
    """Rigid (dy, dx) per frame vs the template, as (n, 2) float64."""
    f = _prep(frames, tmpl.down)
    n, h, w = f.shape
    corr = np.fft.irfft2(np.fft.rfft2(f) * np.conj(tmpl.fft)[None], s=(h, w))
    m = max_shift // tmpl.down
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
    return tmpl.down * np.stack([dy, dx], axis=1)


def apply_shifts(frames: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Warp each frame by its (dy, dx); works for uint8 and float32 input."""
    n, h, w = frames.shape
    out = np.empty_like(frames)
    for i in range(n):
        m = np.array([[1, 0, -shifts[i, 1]], [0, 1, -shifts[i, 0]]], dtype=np.float32)
        out[i] = cv2.warpAffine(frames[i], m, (w, h), flags=cv2.INTER_LINEAR)
    return out


def _mean_template(pre_frames: np.ndarray, down: int) -> Template:
    shape = (int(pre_frames.shape[1]), int(pre_frames.shape[2]))
    return Template(np.fft.rfft2(_prep(pre_frames, down).mean(axis=0)), down, shape)


def make_template(pre_frames: np.ndarray, down: int = 2, max_shift: int = 20) -> Template:
    """Mean template, re-centred: frames are aligned once to the blurred mean, then re-averaged."""
    t0 = _mean_template(pre_frames, down)
    aligned = apply_shifts(pre_frames, estimate(pre_frames, t0, max_shift))
    return _mean_template(aligned, down)


def template_image(tmpl: Template) -> np.ndarray:
    """The template in image space, (h/down, w/down) float32."""
    h, w = tmpl.shape
    s = (h // 2, w // 2) if tmpl.down == 2 else (h, w)
    return np.fft.irfft2(tmpl.fft, s=s).astype(np.float32)


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
        "cafast_version": cafast.__version__,
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
    tmp_json.write_text(json.dumps(meta, indent=1, default=str))
    os.replace(tmp_json, path / META_FILE)


def load_sidecar(path: str | os.PathLike) -> MotionSidecar:
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text())
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
