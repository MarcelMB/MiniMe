"""Per-stage figures of a window-2 rerun (``caanfast tune``), as PNG files.

Shared by every rerun of one run (``SHARED_FIGURES``, from the factors and the motion
sidecar): max projection, motion trace, per-pixel noise and mean. Per rerun
(``RUN_FIGURES``): seeds after init / PNR / merge, initA and init merge footprints,
footprints after each spatial update and merge, footprint sizes, example traces, and with a
Minian result the final overlay (``minian.png``). Plus ``unit_png`` (one unit, on demand),
``unit_table`` and the ``summary.json`` the run list shows.

Matplotlib is used through ``Figure`` objects only (no pyplot), so figures can be drawn from
the server's worker and request threads.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sps
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from caanfast import window2
from caanfast.autotune.run import load_doc
from caanfast.autotune.score import (
    REASONS,
    SIZE_FRAC,  # noqa: F401 - kept importable from here
    Checks,
    n_parts,
    size_above,
    summarize,
    trace_pnr,
    unit_flags,
)
from caanfast.autotune.score import footprint_image as _image
from caanfast.autotune.score import peak_mask as _mask
from caanfast.factors import Factors
from caanfast.init import centroids
from caanfast.motion import ARRAY_FILE
from caanfast.params import Params
from caanfast.tune.minian import MinianOverlay

SHARED_FIGURES = ("max_proj.png", "motion.png", "noise.png")
RUN_FIGURES = ("seeds.png", "init.png", "cnmf.png", "sizes.png", "traces.png")
CAFAST, MINIAN, FRAG = "#2a78d6", "#eb6834", "#e0245e"  # blue, orange, red
INK, MUTED = "#1f1f1e", "#6b6a64"
SEED_COLORS = {"kept": "#4cc9f0", "dropped": "#f72585"}
OUTLINE_LEVEL = 0.3  # outline at 30 % of each footprint's peak
DPI = 110

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    """Everything the figures read from one rerun directory."""

    name: str
    dir: Path
    H: int
    W: int
    T: int
    max_proj: np.ndarray
    params: Params
    report: dict[str, Any]
    seeds: dict[str, pd.DataFrame]
    stages: dict[str, sps.csr_matrix]
    A: sps.csr_matrix
    YrA: np.ndarray
    fps: float | None


def _fps(run_dir: Path) -> float | None:
    raw = window2.raw_dir(run_dir)
    meta = raw.parent / "meta.json" if raw is not None else None
    if meta is None or not meta.exists():
        return None
    return float(json.loads(meta.read_text()).get("fps") or 0) or None


def load_run(d: Path) -> RunData:
    d = Path(d)
    meta = json.loads((d / "factors" / "meta.json").read_text())
    st = d / "stages"
    return RunData(
        name=d.name,
        dir=d,
        H=int(meta["H"]),
        W=int(meta["W"]),
        T=int(meta["T"]),
        max_proj=np.load(d / "factors" / "max_proj.npy").astype(np.float32),
        params=Params.from_json(d / "params.json"),
        report=json.loads((d / "report.json").read_text()),
        seeds={p.stem: pd.read_csv(p) for p in sorted(st.glob("seeds_*.csv"))},
        stages=window2.footprint_stages(d),
        A=sps.load_npz(d / "cnmf" / "A.npz").tocsr(),
        YrA=np.load(d / "cnmf" / "YrA.npy"),
        fps=_fps(d),
    )


# ---------------------------------------------------------------------------
# Footprint metrics
# ---------------------------------------------------------------------------


def outlines(
    A: sps.csr_matrix, H: int, W: int, level: float = OUTLINE_LEVEL
) -> list[list[np.ndarray]]:
    """Per unit, the closed (n, 2) ``(x, y)`` outlines of its pieces above ``level`` of peak."""
    A = sps.csr_matrix(A)
    out: list[list[np.ndarray]] = []
    for k in range(A.shape[0]):
        cs, _ = cv2.findContours(
            _mask(_image(A, k, H, W), level), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        out.append([np.vstack([c[:, 0], c[:1, 0]]).astype(float) for c in cs])
    return out


def shift_rows(A: sps.csr_matrix, H: int, W: int, off: np.ndarray) -> sps.csr_matrix:
    """Footprints moved by the integer-rounded ``off`` (dy, dx); pixels pushed out are dropped."""
    A = sps.coo_matrix(A)
    dy, dx = (int(round(v)) for v in off)
    r, c = A.col // W + dy, A.col % W + dx
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    return sps.csr_matrix((A.data[ok], (A.row[ok], (r * W + c)[ok])), shape=A.shape)


def unit_table(
    run: RunData, mn: MinianOverlay | None, checks: Checks | None = None
) -> pd.DataFrame:
    """One row per final unit: position, sizes, pieces, trace PNR, check flags, Minian match."""
    A = run.A
    cent = centroids(A, run.H, run.W) if A.shape[0] else np.zeros((0, 2))
    df = pd.DataFrame(
        {
            "unit_id": np.arange(A.shape[0]),
            "centroid_h": cent[:, 0].round(1),
            "centroid_w": cent[:, 1].round(1),
            "size_px": np.diff(A.indptr),
            "size10": size_above(A),
            "parts": n_parts(A, run.H, run.W),
            "peak": run.YrA.max(axis=1).astype(float).round(2) if A.shape[0] else [],
            "pnr": trace_pnr(run.YrA).round(1),
        }
    )
    if checks is not None:
        f = unit_flags(run.A, run.YrA, run.H, run.W, checks)
        df["flag"] = [", ".join(r for r in REASONS if row[r]) for _, row in f.iterrows()]
    if mn is not None:
        df["minian_unit"] = pd.array([pd.NA] * len(df), dtype="Int64")
        df["match_dist"] = np.nan
        df["trace_corr"] = np.nan
        df.loc[mn.ui, "minian_unit"] = mn.mi
        df.loc[mn.ui, "match_dist"] = mn.dist.round(2)
        df.loc[mn.ui, "trace_corr"] = mn.corr.round(3)
    return df


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _new(w: float, h: float) -> Figure:
    return Figure(figsize=(w, h), dpi=DPI, facecolor="white", layout="constrained")


def _save(fig: Figure, path: Path | None = None) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    data = buf.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, fontsize=9, color=INK, loc="left")


def _show_proj(ax: Axes, img: np.ndarray, extent: tuple | None = None) -> None:
    lo, hi = np.percentile(img, [1, 99.7])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw(
    ax: Axes,
    lines: list[list[np.ndarray]],
    color: str | list[str],
    lw: float = 0.9,
    off: tuple[float, float] = (0.0, 0.0),
    ls: str = "-",
) -> None:
    colors = [color] * len(lines) if isinstance(color, str) else color
    segs, cols = [], []
    for parts, c in zip(lines, colors, strict=True):
        for p in parts:
            segs.append(p + np.array([off[1], off[0]]))
            cols.append(c)
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, linestyles=ls))


def _footprint_panel(
    ax: Axes, run: RunData, A: sps.csr_matrix, title: str, parts: np.ndarray | None = None
) -> None:
    _show_proj(ax, run.max_proj)
    colors: str | list[str] = CAFAST
    if parts is not None:
        colors = [FRAG if n > 1 else CAFAST for n in parts]
    _draw(ax, outlines(A, run.H, run.W), colors)
    _title(ax, title)


def _legend(ax: Axes, items: list[tuple[str, str, str]], loc: str = "lower left") -> None:
    handles = [Line2D([], [], color=c, lw=2, ls=ls, label=lab) for lab, c, ls in items]
    ax.legend(handles=handles, loc=loc, fontsize=7, framealpha=0.85)


# ---------------------------------------------------------------------------
# Shared figures (factors, motion)
# ---------------------------------------------------------------------------


def fig_max_proj(fac: Factors, path: Path) -> None:
    fig = _new(6.5, 5.6)
    ax = fig.add_subplot()
    lo, hi = np.percentile(fac.max_proj, [1, 99.7])
    im = ax.imshow(fac.max_proj, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="max over frames (after tophat)")
    _title(ax, f"Max projection ({fac.H} x {fac.W_} px, {fac.T} frames)")
    _save(fig, path)


def fig_motion(motion_dir: Path, fps: float | None, path: Path) -> None:
    fig = _new(10, 4.2)
    axs = fig.subplots(2, 1, sharex=True)
    f = motion_dir / ARRAY_FILE
    if not f.exists():
        axs[0].text(0.5, 0.5, "no motion sidecar", ha="center", transform=axs[0].transAxes)
        _save(fig, path)
        return
    with np.load(f) as z:
        sh, q = z["shifts"], z["quality"]
    t = np.arange(len(sh)) / (fps * 60) if fps else np.arange(len(sh))
    axs[0].plot(t, sh[:, 0], lw=0.6, color=CAFAST, label="y (rows)")
    axs[0].plot(t, sh[:, 1], lw=0.6, color=MINIAN, label="x (cols)")
    axs[0].set_ylabel("shift (px)")
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    jumps = int((np.abs(np.diff(sh, axis=0)).max(axis=1) >= 3).sum())
    _title(
        axs[0],
        f"Rigid shifts: max |shift| {np.abs(sh).max():.1f} px, frame-to-frame jumps >= 3 px: "
        f"{jumps}",
    )
    axs[1].plot(t, q, lw=0.6, color=INK)
    axs[1].set_ylabel("template corr")
    axs[1].set_xlabel("time (min)" if fps else "frame")
    _title(axs[1], f"Registration quality (median {np.median(q):.2f})")
    for ax in axs:
        _style(ax)
    _save(fig, path)


def fig_noise(fac: Factors, path: Path) -> None:
    """Per-pixel noise ``sigma`` (it scales the sparse penalty) and per-pixel mean."""
    fig = _new(11, 4.6)
    axs = fig.subplots(1, 2)
    sn = fac.sigma_flat().reshape(fac.H, fac.W_)
    im = axs[0].imshow(sn, cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=axs[0], shrink=0.85)
    zero = float((sn <= 0).mean())
    _title(
        axs[0],
        f"Noise sigma per pixel (sparse penalty = sqrt(T) * sparse_snr * sigma); "
        f"{100 * zero:.0f} % of pixels at 0",
    )
    mean = fac.mean_flat().reshape(fac.H, fac.W_)
    im = axs[1].imshow(mean, cmap="gray", interpolation="nearest")
    fig.colorbar(im, ax=axs[1], shrink=0.85)
    _title(axs[1], "Mean per pixel (after tophat)")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, path)


def make_shared_figures(run_dir: Path, out: Path, fac: Factors | None = None) -> None:
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fac = window2.load_factors(run_dir) if fac is None else fac
    fig_max_proj(fac, out / "max_proj.png")
    fig_motion(run_dir / "motion", _fps(run_dir), out / "motion.png")
    fig_noise(fac, out / "noise.png")


# ---------------------------------------------------------------------------
# Per-rerun figures
# ---------------------------------------------------------------------------


def _scatter(ax: Axes, df: pd.DataFrame, color: str, s: float = 5, marker: str = "o") -> None:
    ax.scatter(df["width"], df["height"], s=s, c=color, marker=marker, linewidths=0.6)


def fig_seeds(run: RunData, path: Path) -> None:
    p = run.params
    fig = _new(17, 4.6)
    axs = fig.subplots(1, 4)
    init, pnr, merged = (run.seeds[k] for k in ("seeds_init", "seeds_pnr", "seeds_merged"))
    _show_proj(axs[0], run.max_proj)
    _scatter(axs[0], init, SEED_COLORS["kept"], s=3)
    _title(
        axs[0],
        f"seeds_init: {len(init)}\n(max_wnd {p.seeds_max_wnd}, diff_thres {p.seeds_diff_thres})",
    )
    keep = pnr["mask_pnr"].to_numpy(bool)
    _show_proj(axs[1], run.max_proj)
    _scatter(axs[1], pnr[~keep], SEED_COLORS["dropped"], s=6, marker="x")
    _scatter(axs[1], pnr[keep], SEED_COLORS["kept"], s=4)
    _title(axs[1], f"PNR: {keep.sum()} kept, {(~keep).sum()} dropped\n(pnr_thres {p.pnr_thres})")
    _show_proj(axs[2], run.max_proj)
    _scatter(axs[2], merged, SEED_COLORS["kept"], s=10)
    _title(
        axs[2],
        f"seeds_merge: {len(merged)}\n(merge_dist {p.merge_dist}, merge_corr {p.merge_corr})",
    )
    ax = axs[3]
    v = np.log10(np.maximum(pnr["pnr"].to_numpy(float), 1e-3))
    bins = np.linspace(v.min(), v.max(), 50)
    ax.hist([v[keep], v[~keep]], bins, stacked=True, color=list(SEED_COLORS.values()))
    if not isinstance(p.pnr_thres, str):
        ax.axvline(np.log10(p.pnr_thres), color=INK, lw=1, ls="--")
    ax.set_xlabel("log10 seed PNR")
    ax.set_ylabel("seeds")
    _title(ax, "Seed PNR (kept / dropped)")
    _style(ax)
    _save(fig, path)


def fig_init(run: RunData, path: Path) -> None:
    p = run.params
    fig = _new(11, 5.4)
    axs = fig.subplots(1, 2)
    a, m = run.stages["initA"], run.stages["init_merge"]
    _footprint_panel(
        axs[0],
        run,
        a,
        f"initA: {a.shape[0]} units (inita_corr {p.inita_corr}, inita_wnd {p.inita_wnd})",
    )
    _footprint_panel(
        axs[1], run, m, f"init unit merge: {m.shape[0]} units (init_merge_corr {p.init_merge_corr})"
    )
    _save(fig, path)


def _stage_title(name: str, A: sps.csr_matrix, parts: np.ndarray, p: Params) -> str:
    kind, it = name.rsplit("_", 1)
    head = (
        f"spatial update {it} (dl_wnd {p.dl_wnd}, sparse_snr {p.sparse_snr}, "
        f"size_thres {p.size_thres})"
        if kind == "spatial"
        else f"unit merge {it} (loop_merge_corr {p.loop_merge_corr})"
    )
    med = np.median(size_above(A)) if A.shape[0] else 0
    return f"{head}\n{A.shape[0]} units, median {med:.0f} px > 10 %, {(parts > 1).sum()} in pieces"


def fig_cnmf(run: RunData, path: Path) -> None:
    names = [n for n in run.stages if n.startswith(("spatial_", "merge_"))]
    cols = min(4, max(1, len(names)))
    rows = int(np.ceil(len(names) / cols))
    fig = _new(5.2 * cols, 5.6 * rows)
    axs = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax, name in zip(axs, names, strict=False):
        A = run.stages[name]
        parts = n_parts(A, run.H, run.W)
        _footprint_panel(ax, run, A, _stage_title(name, A, parts, run.params), parts)
    for ax in axs[len(names) :]:
        ax.set_axis_off()
    _legend(axs[0], [("footprint", CAFAST, "-"), ("in >1 pieces", FRAG, "-")])
    _save(fig, path)


def fig_sizes(run: RunData, mn: MinianOverlay | None, path: Path) -> None:
    fig = _new(12, 4.2)
    axs = fig.subplots(1, 2)
    sizes = {n: size_above(A) for n, A in run.stages.items() if not n.startswith("merge_")}
    sizes["final"] = size_above(run.A)
    hi = max(float(np.percentile(s, 99)) for s in sizes.values() if len(s))
    if mn is not None:
        hi = max(hi, float(np.percentile(mn.size10, 99)))
    bins = np.linspace(0, hi, 40)
    shades = np.linspace(0.25, 1.0, len(sizes))
    for (name, s), a in zip(sizes.items(), shades, strict=True):
        lab = f"{name} (median {np.median(s):.0f})" if len(s) else name
        axs[0].hist(
            s,
            bins,
            histtype="step",
            lw=2 if name == "final" else 1.2,
            color=CAFAST,
            alpha=float(a),
            label=lab,
        )
    if mn is not None:
        axs[0].hist(
            mn.size10,
            bins,
            histtype="step",
            lw=2,
            color=MINIAN,
            ls="--",
            label=f"Minian (median {np.median(mn.size10):.0f})",
        )
    axs[0].set_xlabel("pixels above 10 % of peak")
    axs[0].set_ylabel("units")
    axs[0].legend(fontsize=7, frameon=False)
    _title(axs[0], "Footprint size by stage")
    parts = n_parts(run.A, run.H, run.W)
    k = np.arange(1, max(4, int(parts.max(initial=1))) + 1)
    axs[1].bar(k - 0.2, [(parts == i).sum() for i in k], 0.4, color=CAFAST, label="CaFast")
    if mn is not None:
        mparts = n_parts(mn.A, run.H, run.W)
        axs[1].bar(k + 0.2, [(mparts == i).sum() for i in k], 0.4, color=MINIAN, label="Minian")
    axs[1].set_xticks(k)
    axs[1].set_xlabel("pieces per final footprint (above 10 % of peak)")
    axs[1].set_ylabel("units")
    axs[1].legend(fontsize=7, frameon=False)
    _title(axs[1], "Fragmentation")
    for ax in axs:
        _style(ax)
    _save(fig, path)


def _example_units(size10: np.ndarray, n: int = 8) -> list[int]:
    """Units at evenly spaced quantiles of footprint size (smallest to largest)."""
    if len(size10) == 0:
        return []
    order = np.argsort(size10, kind="stable")
    picks = order[np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int)]
    return list(dict.fromkeys(int(i) for i in picks))


def _time(run: RunData, n: int) -> tuple[np.ndarray, str]:
    if run.fps:
        return np.arange(n) / (run.fps * 60), "time (min)"
    return np.arange(n), "frame"


def fig_traces(run: RunData, mn: MinianOverlay | None, path: Path) -> None:
    size10 = size_above(run.A)
    units = _example_units(size10)
    fig = _new(14, 1.0 + 0.95 * max(1, len(units)))
    ax = fig.add_subplot()
    t, xlabel = _time(run, run.YrA.shape[1])
    match = dict(zip(mn.ui.tolist(), mn.mi.tolist(), strict=True)) if mn is not None else {}
    for row, k in enumerate(units):
        y = run.YrA[k]
        ax.plot(t, (y - np.median(y)) / (np.ptp(y) or 1) - 1.3 * row, color=CAFAST, lw=0.5)
        label = f"unit {k}: {size10[k]} px"
        if k in match:
            m = mn.C[match[k]]
            T = min(len(m), len(t))
            ax.plot(
                t[:T],
                (m[:T] - np.median(m)) / (np.ptp(m) or 1) - 1.3 * row,
                color=MINIAN,
                lw=0.5,
                alpha=0.8,
            )
            label += f", Minian {match[k]}"
        ax.text(t[0], -1.3 * row + 0.8, label, fontsize=7, color=MUTED)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    _style(ax)
    ax.spines["left"].set_visible(False)
    _title(ax, "Example final traces (YrA, scaled per trace), smallest to largest footprint")
    if match:
        _legend(ax, [("CaFast", CAFAST, "-"), ("Minian", MINIAN, "-")], loc="upper right")
    _save(fig, path)


def fig_minian(run: RunData, mn: MinianOverlay, path: Path) -> None:
    fig = _new(7.5, 7.2)
    ax = fig.add_subplot()
    _show_proj(ax, run.max_proj)
    ui, mi = set(mn.ui.tolist()), set(mn.mi.tolist())
    c_lines = outlines(run.A, run.H, run.W)
    m_lines = outlines(mn.A, run.H, run.W)
    off = (float(mn.offset[0]), float(mn.offset[1]))
    _draw(ax, [c for k, c in enumerate(c_lines) if k in ui], CAFAST, lw=0.7)
    _draw(ax, [c for k, c in enumerate(m_lines) if k in mi], MINIAN, lw=0.7, off=off)
    _draw(ax, [c for k, c in enumerate(c_lines) if k not in ui], CAFAST, lw=1.8)
    _draw(ax, [c for k, c in enumerate(m_lines) if k not in mi], MINIAN, lw=1.8, off=off)
    corr = float(np.nanmedian(mn.corr)) if len(mn.corr) else float("nan")
    _title(
        ax,
        f"Final vs Minian: {len(ui)} matched of {run.A.shape[0]} CaFast / {mn.A.shape[0]} "
        f"Minian, trace corr median {corr:.2f}\nthick = unmatched",
    )
    _legend(ax, [("CaFast", CAFAST, "-"), ("Minian", MINIAN, "-")])
    _save(fig, path)


def unit_png(run: RunData, k: int, mn: MinianOverlay | None, pad: int = 18) -> bytes:
    """One final unit: footprint on the max projection (crop), its values, and its trace."""
    img = _image(run.A, k, run.H, run.W)
    cy, cx = centroids(run.A[k], run.H, run.W)[0]
    y0, y1 = max(0, int(cy) - pad), min(run.H, int(cy) + pad + 1)
    x0, x1 = max(0, int(cx) - pad), min(run.W, int(cx) + pad + 1)
    ext = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
    fig = _new(11, 6.2)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])
    ax = fig.add_subplot(gs[0, 0])
    _show_proj(ax, run.max_proj[y0:y1, x0:x1], extent=ext)
    lines = outlines(run.A, run.H, run.W)
    _draw(ax, [c for j, c in enumerate(lines) if j != k], CAFAST, lw=0.6, ls=":")
    _draw(ax, [lines[k]], CAFAST, lw=1.6)
    if mn is not None:
        off = (float(mn.offset[0]), float(mn.offset[1]))
        _draw(ax, outlines(mn.A, run.H, run.W), MINIAN, lw=1.0, off=off)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _title(ax, f"unit {k} at ({cy:.0f}, {cx:.0f}) on max projection")
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(img[y0:y1, x0:x1], cmap="viridis", extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    n10, pieces = size_above(run.A[k])[0], n_parts(run.A[k], run.H, run.W)[0]
    _title(ax, f"footprint: {n10} px > 10 %, {pieces} piece(s)")
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(img[img > 0], 30, color=CAFAST)
    ax.set_xlabel("footprint value")
    _style(ax)
    _title(ax, f"{int((img > 0).sum())} nonzero pixels")
    ax = fig.add_subplot(gs[1, :])
    t, xlabel = _time(run, run.YrA.shape[1])
    ax.plot(t, run.YrA[k], color=CAFAST, lw=0.5)
    title = "YrA"
    if mn is not None and k in mn.ui:
        j = int(np.flatnonzero(mn.ui == k)[0])
        m = mn.C[mn.mi[j]][: len(t)]
        y = run.YrA[k]
        ax.plot(t[: len(m)], m / (np.ptp(m) or 1) * np.ptp(y), color=MINIAN, lw=0.5, alpha=0.8)
        title += f" (blue) and matched Minian unit {mn.mi[j]} C, scaled (r = {mn.corr[j]:.2f})"
    ax.set_xlabel(xlabel)
    _style(ax)
    _title(ax, title)
    return _save(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def summary(run: RunData, mn: MinianOverlay | None, checks: Checks | None = None) -> dict[str, Any]:
    c = run.report["counts"]
    size10 = size_above(run.A)
    out: dict[str, Any] = {
        "name": run.name,
        **{k: c.get(k) for k in ("seeds_init", "seeds_pnr", "seeds_merged", "units_initA")},
        "units_final": int(run.A.shape[0]),
        "size10_median": float(np.median(size10)) if len(size10) else None,
        "in_pieces": int((n_parts(run.A, run.H, run.W) > 1).sum()),
        "wall_s": round(float(run.report.get("wall_total", 0.0)), 1),
    }
    if mn is not None:
        out |= {
            "minian_units": int(mn.A.shape[0]),
            "minian_matched": int(len(mn.ui)),
            "trace_corr_median": float(np.nanmedian(mn.corr)) if len(mn.corr) else None,
            "minian_size10_median": float(np.median(mn.size10)),
        }
    if checks is not None:
        s = summarize(unit_flags(run.A, run.YrA, run.H, run.W, checks))
        out |= {"good": s["good"], "flagged": s["flagged"]}
    return out


def make_run_figures(
    run: RunData, mn: MinianOverlay | None, checks: Checks | None = None
) -> dict[str, Any]:
    """Write ``RUN_FIGURES`` (and ``minian.png``, the sweep curves) to ``run.dir/figures``.

    Returns the summary, with ``good`` / ``flagged`` when ``checks`` is given.
    """
    from caanfast.tune import autofig  # here: autofig reads the palette from this module

    out = run.dir / "figures"
    out.mkdir(exist_ok=True)
    fig_seeds(run, out / "seeds.png")
    fig_init(run, out / "init.png")
    fig_cnmf(run, out / "cnmf.png")
    fig_sizes(run, mn, out / "sizes.png")
    fig_traces(run, mn, out / "traces.png")
    if mn is not None:
        fig_minian(run, mn, out / "minian.png")
    sweep_csv, doc = run.dir / "sweep.csv", load_doc(run.dir)
    if sweep_csv.exists() and doc is not None:
        autofig.sweep_curves(sweep_csv, doc, out)
    s = summary(run, mn, checks)
    (run.dir / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def write_summary(run: RunData, mn: MinianOverlay | None, checks: Checks | None) -> None:
    """Rewrite ``summary.json`` only (after new checks), without drawing figures."""
    (run.dir / "summary.json").write_text(json.dumps(summary(run, mn, checks), indent=1))
