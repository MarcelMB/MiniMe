"""Figures for auto-tune: one small sweep curve per searched parameter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cafast.autotune.sweep import SEARCHED, STEP_OF
from cafast.tune.figures import CAFAST, FRAG, INK, MINIAN, MUTED, _style

GOOD, FLAGGED, CHOSEN = CAFAST, FRAG, MINIAN
DPI = 110


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


def _same(a: Any, b: Any) -> bool:
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
    hit = [i for i, v in enumerate(b[key]) if _same(v, chosen)]
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
