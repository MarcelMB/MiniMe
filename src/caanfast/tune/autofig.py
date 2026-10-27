"""Figures for auto-tune: a small sweep curve per searched parameter and a thumbnail per unit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from caanfast.autotune.score import footprint_image, peak_mask
from caanfast.autotune.sweep import SEARCHED, STEP_OF
from caanfast.init import centroids
from caanfast.tune.figures import CAFAST, FRAG, INK, MINIAN, MUTED, OUTLINE_LEVEL, _style

GOOD, FLAGGED, CHOSEN = CAFAST, FRAG, MINIAN
DPI = 110
THUMB_SCALE = 4


def _is_num(v: Any) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def _order(values: pd.Series) -> list[Any]:
    """Grid order: ``auto`` first, then numbers ascending."""
    uniq = list(dict.fromkeys(values.tolist()))
    words = [v for v in uniq if not _is_num(v)]
    nums = sorted((v for v in uniq if _is_num(v)), key=float)
    return words + nums


def best_per_value(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per value of ``key`` in its step: the best-scoring successful trial, in grid order."""
    d = df[(df["step"] == STEP_OF[key]) & df["score"].notna()]
    if d.empty:
        return d
    top = d.sort_values("score", ascending=False, kind="stable").drop_duplicates(key)
    order = {v: i for i, v in enumerate(_order(top[key]))}
    return top.sort_values(key, key=lambda s: s.map(order)).reset_index(drop=True)


def same_value(a: Any, b: Any) -> bool:
    """``2`` and ``2.0`` and ``"2.0"`` are one value; ``auto`` only equals ``auto``."""
    if _is_num(a) and _is_num(b):
        return float(a) == float(b)
    return str(a) == str(b)


def _curve(df: pd.DataFrame, key: str, chosen: Any, path: Path) -> None:
    b = best_per_value(df, key)
    fig = Figure(figsize=(2.8, 1.5), dpi=DPI, facecolor="white", layout="constrained")
    ax = fig.add_subplot()
    x = np.arange(len(b))
    ax.plot(x, b["good"], "-o", color=GOOD, ms=3, lw=1.2, label="good")
    ax.plot(x, b["flagged"], "-o", color=FLAGGED, ms=3, lw=1.2, label="flagged")
    hit = [i for i, v in enumerate(b[key]) if same_value(v, chosen)]
    if hit:
        ax.axvline(hit[0], color=CHOSEN, lw=1, ls="--")
    ax.set_xticks(x, [str(v) for v in b[key]], fontsize=6, color=INK)
    ax.tick_params(axis="y", labelsize=6, colors=MUTED)
    _style(ax)
    ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=DPI)


def sweep_curves(sweep_csv: Path, doc: dict[str, Any], out_dir: Path) -> list[str]:
    """Write ``sweep_<key>.png`` for every searched key that has rows; returns the file names."""
    df = pd.read_csv(sweep_csv)
    names = []
    for key in SEARCHED:
        if key not in df or best_per_value(df, key).empty:
            continue
        name = f"sweep_{key}.png"
        _curve(df, key, doc["proposed"].get(key), Path(out_dir) / name)
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Unit thumbnails
# ---------------------------------------------------------------------------


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def _gray8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def thumb_png(run: Any, k: int, pad: int = 12) -> bytes:
    """Max-projection crop (2*pad+1 px, edge-padded) around unit ``k`` with its outline, 4x."""
    cy, cx = (int(round(v)) for v in centroids(run.A[k], run.H, run.W)[0])
    big = np.pad(_gray8(run.max_proj), pad, mode="constant")
    fp = np.pad(peak_mask(footprint_image(run.A, k, run.H, run.W), OUTLINE_LEVEL), pad)
    sl = np.s_[cy : cy + 2 * pad + 1, cx : cx + 2 * pad + 1]
    s = THUMB_SCALE
    crop = cv2.resize(big[sl], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(fp[sl], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cs, -1, _bgr(CAFAST), 1)
    return cv2.imencode(".png", rgb)[1].tobytes()


# ---------------------------------------------------------------------------
# Max projection for measuring
# ---------------------------------------------------------------------------


def max_proj_png(max_proj: np.ndarray) -> bytes:
    """The max projection as an 8-bit PNG, one image pixel per factor pixel."""
    return cv2.imencode(".png", _gray8(np.asarray(max_proj, np.float32)))[1].tobytes()
