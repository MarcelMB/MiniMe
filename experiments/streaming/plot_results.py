"""One figure of every streaming test against the batch baseline.

Reads whatever ``t1_discovery.py`` and ``t2_stream.py`` wrote under
``processed/stream/`` (missing runs are skipped) and writes
``processed/stream/figures/streaming_results.png``:

* row 1: recall, precision and F1 per run (batch in blue, streaming runs in orange);
* row 2: recall per truth category, per-unit trace correlation with the truth, and a map of
  which true cells the batch run and the streaming pipeline found;
* row 3: example traces, footprint stability per segment, and memory (raw movie, batch
  factors, streaming state, peak RSS).

    python experiments/streaming/plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sps  # noqa: E402
import stream as st  # noqa: E402

from minime import score as msc  # noqa: E402

# reference palette (dataviz skill), light mode; slots 1-3 validated all-pairs
BATCH, STREAM, BOTH = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
CATS = ("dim", "late", "overlap", "small", "near_out_of_focus")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def runs(root: Path, baseline: dict) -> list[tuple[str, dict, str]]:
    """``(label, brief score, colour)`` of every finished run, batch first."""
    out = [("Batch MiniMe", baseline, BATCH)]
    for mode in ("subset", "fresh"):
        summ = _json(root / f"t1_{mode}" / "summary.json") or {"runs": {}}
        for length, r in summ["runs"].items():
            if r["windows"] and r["units_per_window_median"] > 0:
                out.append((f"Windows {mode} {int(length) / 20 / 60:g} min", r["stitched"], STREAM))
    labels = {
        "t2_baseline_final_u0": "Batch footprints, streamed traces",
        "t2_baseline_init_u1": "Batch init + 1 streamed update",
        "t2_baseline_init_u2": "Batch init + streamed updates",
        "t2_stitched_u1": "Full streaming pipeline",
        "t2_stitched_u1_lp": "Full streaming, low-pass merge",
        "t2_stitched_u2_lp": "Full streaming, 2 updates, low-pass merge",
    }
    for name, label in labels.items():
        res = _json(root / name / "results.json")
        if res:
            out.append((label, res["passes"][-1]["score"], STREAM))
    return out


def latest_full(root: Path) -> tuple[Path | None, int]:
    """The newest finished full-pipeline run (most updates, low-pass merge first) and its
    last pass."""
    for name in ("t2_stitched_u2_lp", "t2_stitched_u1_lp", "t2_stitched_u1"):
        res = _json(root / name / "results.json")
        if res:
            last = res["passes"][-1]["pass"]
            if (root / name / f"pass{last}_C.npy").is_file():
                return root / name, last
    return None, 0


def matched(s: st.Session, A: sps.csr_matrix, shifts: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(ui, ti, offset)``: units matched to true cells as ``score_run`` matches them."""
    truth = msc.load_truth(s.meta_path)
    H, W, _ = s.shape()
    c, _, sign = msc.motion_fit(np.asarray(shifts, float), truth["shifts"].astype(float))
    unit_c = msc.footprint_centroids(A, H, W)
    best = min(
        (msc._median_dist(unit_c, truth["centroids"] + m * c) + (m,) for m in (sign, -sign)),
        key=lambda b: b[0],
    )
    return best[1], best[2], best[3] * c


def unit_corr(truth: dict, C: np.ndarray, ui: np.ndarray, ti: np.ndarray) -> np.ndarray:
    ok = truth["frame_ok"]
    return msc.row_corr(C[ui][:, ok].astype(np.float64), truth["traces"][ti][:, ok])


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def metric_panel(ax: plt.Axes, rs: list, key: str, title: str, show_labels: bool) -> None:
    y = np.arange(len(rs))[::-1]
    vals = [r[1].get(key) or 0.0 for r in rs]
    ax.barh(y, vals, height=0.6, color=[r[2] for r in rs], edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(vals[0], color=BATCH, lw=1, ls=":")
    ax.set_yticks(y, [r[0] for r in rs] if show_labels else [""] * len(rs), fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, title)


def category_panel(ax: plt.Axes, base: dict, best: tuple[str, dict, str]) -> None:
    x = np.arange(len(CATS))
    for dx, (sc, col, lab) in zip(
        (-0.2, 0.2), ((base, BATCH, "Batch MiniMe"), (best[1], STREAM, best[0])), strict=True
    ):
        v = [sc["recall_by_category"].get(c, 0) for c in CATS]
        ax.bar(x + dx, v, width=0.38, color=col, label=lab, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, [c.replace("near_out_of_focus", "near OOF") for c in CATS], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, "Recall by cell category")


def ecdf_panel(ax: plt.Axes, series: list[tuple[str, np.ndarray, str]]) -> None:
    for lab, r, col in series:
        r = np.sort(r[np.isfinite(r)])
        ax.step(r, np.arange(1, r.size + 1) / r.size, where="post", color=col, lw=2, label=lab)
        ax.axvline(np.median(r), color=col, lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("correlation with true trace (per matched unit)", fontsize=8, color=INK2)
    ax.set_ylabel("fraction of units", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.8)
    _style(ax, "Trace quality (dotted: median)")


def map_panel(
    ax: plt.Axes, img: np.ndarray, true_c: np.ndarray, det: np.ndarray, t_batch, t_stream
) -> None:
    ax.imshow(img, cmap="gray", vmax=np.percentile(img, 99.5))
    b, s_ = np.isin(np.arange(len(true_c)), t_batch), np.isin(np.arange(len(true_c)), t_stream)
    groups = (
        (b & s_, BOTH, "o", "found by both"),
        (b & ~s_, BATCH, "s", "batch only"),
        (~b & s_, STREAM, "D", "streaming only"),
        (~b & ~s_ & det, MUTED, "x", "missed (detectable)"),
    )
    for m, col, mk, lab in groups:
        ax.scatter(
            true_c[m, 1], true_c[m, 0], s=14, c=col, marker=mk, lw=1.2, label=f"{lab} ({m.sum()})"
        )
    ax.legend(fontsize=7, frameon=True, loc="lower left", markerscale=1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    _style(ax, "True cells found (on the max projection)")


def traces_panel(ax: plt.Axes, truth: dict, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    """Truth, batch and streamed traces of a few units, z-scored, 2 min around a peak."""
    fps, span = 20, 2400
    for k, (ti, cb, cs) in enumerate(rows):
        tr = truth["traces"][ti]
        t0 = int(np.clip(np.argmax(tr) - span // 3, 0, tr.size - span))
        sl = slice(t0, t0 + span)
        t = np.arange(span) / fps
        for x, col, lw, lab in (
            (tr, INK, 1.2, "truth"),
            (cb, BATCH, 1, "batch"),
            (cs, STREAM, 1, "streamed"),
        ):
            z = (x[sl] - x[sl].mean()) / (x[sl].std() + 1e-9)
            ax.plot(t, z / 6 - 1.4 * k, color=col, lw=lw, label=lab if k == 0 else None)
    ax.set_yticks([])
    ax.set_xlabel("seconds", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=3)
    _style(ax, "Example traces (batch footprints)")


def stability_panel(ax: plt.Axes, stab: dict, seg_min: float) -> None:
    cos = stab["cos"]
    x = (np.arange(cos.shape[0]) + 0.5) * seg_min
    q = np.nanpercentile(cos, [25, 50, 75], axis=1)
    ax.fill_between(x, q[0], q[2], color=STREAM, alpha=0.2, lw=0, label="25-75 %")
    ax.plot(x, q[1], color=STREAM, lw=2, marker="o", ms=8, label="median")
    ax.set_ylim(min(0.85, float(np.nanmin(q[0])) - 0.02), 1.0)
    ax.set_xlabel("minutes into session (segment centre)", fontsize=8, color=INK2)
    ax.set_ylabel("cosine to whole-session footprint", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.grid(color=GRID, lw=0.8)
    _style(ax, "Footprints fitted per segment")


def memory_panel(ax: plt.Axes, items: list[tuple[str, float, str]]) -> None:
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v, _ in items], height=0.6, color=[c for _, _, c in items])
    for yi, (_, v, _) in zip(y, items, strict=True):
        ax.text(v * 1.15, yi, f"{v:,.0f} MB", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(10, max(v for _, v, _ in items) * 8)
    ax.set_yticks(y, [n for n, _, _ in items], fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, "Memory (log scale)")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    a = ap.parse_args()

    s = st.Session(a.session)
    root = s.root / "processed" / "stream"
    base = st.brief(s.baseline_score())
    shifts = s.sidecar().shifts
    truth = msc.load_truth(s.meta_path)
    rs = runs(root, base)

    fig, axes = plt.subplots(3, 3, figsize=(17, 14), facecolor=SURFACE)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.05, hspace=0.38, wspace=0.28)
    fig.suptitle(
        "Streaming MiniMe vs batch, 30-min synthetic GRIN session",
        x=0.02,
        ha="left",
        fontsize=14,
        color=INK,
    )
    for k, (key, title) in enumerate(
        (
            ("recall_detectable", "Recall (detectable cells)"),
            ("precision", "Precision"),
            ("f1", "F1"),
        )
    ):
        metric_panel(axes[0, k], rs, key, title, show_labels=k == 0)

    best = next((r for r in reversed(rs) if r[0].startswith("Full streaming")), None)
    best = best or next((r for r in rs if r[0].startswith("Windows")), rs[-1])
    category_panel(axes[1, 0], base, best)

    A_b, C_b = s.baseline_units("cnmf")
    ui_b, ti_b, offset = matched(s, A_b, shifts)
    series = [("Batch MiniMe", unit_corr(truth, C_b, ui_b, ti_b), BATCH)]
    t3 = root / "t2_baseline_final_u0"
    C_t3 = np.load(t3 / "pass0_C.npy") if (t3 / "pass0_C.npy").is_file() else None
    if C_t3 is not None:
        series.append(
            ("Batch footprints, streamed traces", unit_corr(truth, C_t3, ui_b, ti_b), STREAM)
        )
    full, last = latest_full(root)
    stream_A = None
    if full is not None:
        stream_A = full / f"pass{last}_A.npz"
        A_f = sps.csr_matrix(sps.load_npz(stream_A))
        ui_f, ti_f, _ = matched(s, A_f, shifts)
        C_f = np.load(full / f"pass{last}_C.npy")
        series.append((best[0], unit_corr(truth, C_f, ui_f, ti_f), BOTH))
    ecdf_panel(axes[1, 1], series)

    if stream_A is None:
        for mode in ("fresh", "subset"):
            p = root / f"t1_{mode}" / "w3000" / "stitched_A.npz"
            if p.is_file():
                stream_A = p
                break
    img = np.load(s.run_dir / "factors" / "max_proj.npy")
    true_c = truth["centroids"] + offset
    t_stream = matched(s, sps.csr_matrix(sps.load_npz(stream_A)), shifts)[1] if stream_A else []
    map_panel(axes[1, 2], img, true_c, truth["cat_detectable"], ti_b, t_stream)
    if stream_A is not None:
        axes[1, 2].set_title(
            "True cells found: batch vs full streaming"
            if stream_A.parent.name.startswith("t2_stitched")
            else f"True cells found: batch vs windows {stream_A.parent.name}",
            loc="left",
            fontsize=11,
            color=INK,
        )

    if C_t3 is not None:
        r = unit_corr(truth, C_b, ui_b, ti_b)
        pick = np.argsort(r)[np.linspace(len(r) * 0.3, len(r) * 0.9, 3).astype(int)]
        traces_panel(axes[2, 0], truth, [(ti_b[i], C_b[ui_b[i]], C_t3[ui_b[i]]) for i in pick])
    else:
        axes[2, 0].set_axis_off()

    stab_dirs = sorted(root.glob("t2_*_u[1-9]*/stability.npz"))
    if stab_dirs:
        with np.load(stab_dirs[-1]) as z:
            stab = {k: z[k] for k in z.files}
        res = _json(stab_dirs[-1].parent / "results.json")
        seg_min = int(res["args"]["segment_frames"]) / 20 / 60
        stability_panel(axes[2, 1], stab, seg_min)
    else:
        axes[2, 1].set_axis_off()

    rep = _json(s.run_dir / "report.json") or {}
    t2s = [_json(p) for p in sorted(root.glob("t2_*/results.json"))]
    passes = [p for r in t2s if r for p in r["passes"]]
    items = [
        ("raw movie", rep.get("bytes", {}).get("raw", 0) / 2**20, MUTED),
        ("batch factors on disk", rep.get("bytes", {}).get("factors", 0) / 2**20, BATCH),
        ("batch peak RSS", (rep.get("peak_rss") or 0) / 2**20, BATCH),
        ("streaming state", max((p["state_mb"] for p in passes), default=0), STREAM),
        (
            "streaming script peak RSS",
            max((p["peak_rss_mb"] or 0 for p in passes), default=0),
            STREAM,
        ),
    ]
    memory_panel(axes[2, 2], [i for i in items if i[1] > 0])

    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "streaming_results.png", dpi=130, facecolor=SURFACE)
    print(out / "streaming_results.png")


if __name__ == "__main__":
    main()
