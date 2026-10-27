"""Static per-stage viewer of one cafast run (figures + a clickable unit browser).

Run under the cafast venv::

    .venv/bin/python bench/stage_viewer.py RUN_DIR [--frames 0,72000,144000,215999]
    python3 -m http.server -d RUN_DIR/viewer 8765

Stages shown: raw vs preprocessed (needs the run's kept tile store, ``--keep-store``)
vs factor reconstruction, motion shifts and quality, per-block residual shift, factors
(max projection, rank per tile, noise), seeds (init -> PNR -> merged; recomputed from the
saved factors, ~2.5 min on 3 h), initial units vs final units, and traces. Writes
``RUN_DIR/viewer/`` (PNGs, ``units.json``, ``index.html``); the run itself is read-only.
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
from scipy.ndimage import binary_erosion  # noqa: E402

from cafast import corr, io, motion, pipeline  # noqa: E402
from cafast.factors import Factors  # noqa: E402
from cafast.params import Params  # noqa: E402
from cafast.report import StageTimer  # noqa: E402

DPI = 110

# ---------------------------------------------------------------------------
# Frame access
# ---------------------------------------------------------------------------


def raw_frame(info: io.RawInfo, t: int) -> np.ndarray:
    edges = np.cumsum([0, *info.segment_frames])
    s = int(np.searchsorted(edges, t, side="right") - 1)
    k = t - int(edges[s])
    return io.read_avi(info.segments[s], max_frames=k + 1, bgr=info.bgr)[k].astype(np.float32)


def store_frame(store: Path, fac: Factors, t: int) -> np.ndarray | None:
    """Frame ``t`` of the preprocessed uint8 tile store, or None if the store is gone."""
    parts = sorted(store.glob("tile_000.part*"))
    if not parts:
        return None
    n_per = [p.stat().st_size // fac.grid[0].pixels for p in parts]
    edges = np.cumsum([0, *n_per])
    p = int(np.searchsorted(edges, t, side="right") - 1)
    k = t - int(edges[p])
    out = np.zeros((fac.H, fac.W_), np.float32)
    for tl in fac.grid:
        mm = np.memmap(store / f"tile_{tl.index:03d}.part{p:02d}", np.uint8, "r")
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = mm.reshape(-1, *tl.shape)[k]
    return out


def recon_frame(fac: Factors, t: int) -> np.ndarray:
    out = np.zeros((fac.H, fac.W_), np.float32)
    for b, tl in enumerate(fac.grid):
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = fac.reconstruct_chunk(b, t, t + 1)[:, 0].reshape(
            tl.shape
        )
    return out


def tile_map(fac: Factors, values: list[float]) -> np.ndarray:
    out = np.zeros((fac.H, fac.W_), np.float32)
    for tl, v in zip(fac.grid, values, strict=True):
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = v
    return out


def flat_img(fac: Factors, flat: np.ndarray) -> np.ndarray:
    return np.asarray(flat, np.float32).reshape(fac.H, fac.W_)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _show(ax: plt.Axes, img: np.ndarray, title: str, cmap: str = "gray", **kw) -> None:
    lo, hi = np.percentile(img, [0.5, 99.8]) if "vmin" not in kw else (kw.pop("vmin"), None)
    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=kw.pop("vmax", hi), **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


STAGE_LABELS = {
    "pass1": "preprocessing + motion correction (pass 1)",
    "pass2": "compression (pass 2)",
}


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


def fig_timings(report: dict, path: Path) -> None:
    rows = [r for r in report["stages"] if r["seconds"] >= 0.5]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1))
    names = [stage_label(r["stage"]) for r in rows][::-1]
    secs = [r["seconds"] for r in rows][::-1]
    ax.barh(names, secs, color="#4c72b0")
    for i, s in enumerate(secs):
        ax.text(s, i, f" {s:.0f} s", va="center", fontsize=8)
    ax.set_xlabel("wall seconds")
    ax.set_title(f"wall total {report['wall_total']:.0f} s, {report['workers']} workers")
    _save(fig, path)


def fig_frames(info: io.RawInfo, fac: Factors, store: Path, ts: list[int], fps: float, path: Path):
    have_store = store_frame(store, fac, 0) is not None
    rows = (
        ["raw", "preprocessed (store)", "factor reconstruction"]
        if have_store
        else ["raw", "factor reconstruction"]
    )
    fig, axes = plt.subplots(len(rows), len(ts), figsize=(3.6 * len(ts), 3.2 * len(rows)))
    for j, t in enumerate(ts):
        imgs = [raw_frame(info, t)]
        if have_store:
            imgs.append(store_frame(store, fac, t))
        imgs.append(recon_frame(fac, t))
        for i, (name, img) in enumerate(zip(rows, imgs, strict=True)):
            _show(axes[i, j], img, f"{name}\nframe {t} ({t / fps / 3600:.2f} h)")
    _save(fig, path)


def fig_motion(sc: motion.MotionSidecar, fps: float, path: Path) -> None:
    h = np.arange(sc.n_frames) / fps / 3600
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for ax, k, name in zip(axes[:2], (0, 1), ("dy", "dx"), strict=True):
        ax.plot(h, sc.shifts[:, k], lw=0.3)
        ax.set_ylabel(f"{name} (px)")
    axes[2].plot(h, sc.quality, lw=0.3, color="#55a868")
    axes[2].set_ylabel("quality")
    axes[2].set_xlabel("hours")
    axes[0].set_title(
        f"rigid shifts, max |shift| {np.abs(sc.shifts).max():.1f} px, "
        f"median quality {np.median(sc.quality):.2f}"
    )
    _save(fig, path)


def block_residual(fac: Factors, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Residual (dy, dx) and correlation of each corrected ``block_max`` vs the whole-run max.

    A consistent registration gives ~0 px for every block; a block locked to a slightly
    wrong offset shows up as a step here even when its per-frame quality looks normal.
    """
    ref = motion._mean_template(fac.max_proj[None].astype(np.float32), down=1)
    shifts, corr = [], []
    for i in range(0, len(fac.block_max), chunk):
        blk = fac.block_max[i : i + chunk].astype(np.float32)
        shifts.append(motion.estimate(blk, ref))
        corr.append(motion.frame_quality(blk, ref))
    return np.concatenate(shifts), np.concatenate(corr)


def fig_drift(fac: Factors, fps: float, path: Path) -> None:
    shifts, corr = block_residual(fac)
    e = fac.block_edges
    h = (e[:-1] + e[1:]) / 2 / fps / 3600
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    axes[0].plot(h, shifts[:, 0], ".-", ms=2, lw=0.5, label="dy")
    axes[0].plot(h, shifts[:, 1], ".-", ms=2, lw=0.5, label="dx")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("residual shift (px)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(
        f"per-block max projection vs whole-run max: {len(h)} blocks, "
        f"max |residual| {np.abs(shifts).max():.2f} px, median corr {np.median(corr):.2f}"
    )
    axes[1].plot(h, corr, ".-", ms=2, lw=0.5, color="#55a868")
    axes[1].set_ylabel("correlation")
    axes[1].set_xlabel("hours")
    _save(fig, path)


def fig_factors(fac: Factors, sc: motion.MotionSidecar, path: Path) -> None:
    ranks = [tf.rank for tf in fac.tiles]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    _show(axes[0, 0], fac.max_proj, "max projection (preprocessed)")
    _show(axes[0, 1], sc.template, "motion template")
    rm = tile_map(fac, ranks)
    _show(
        axes[1, 0],
        rm,
        f"rank per tile (sum {sum(ranks)}, max {max(ranks)})",
        "viridis",
        vmin=0,
        vmax=max(ranks),
    )
    for tl, r in zip(fac.grid, ranks, strict=True):
        axes[1, 0].text(
            (tl.x0 + tl.x1) / 2,
            (tl.y0 + tl.y1) / 2,
            str(r),
            ha="center",
            va="center",
            fontsize=6,
            color="w",
        )
    _show(axes[1, 1], flat_img(fac, fac.sigma_flat()), "noise sigma per pixel", "magma")
    _save(fig, path)


def fig_seeds(fac: Factors, stages: list[tuple[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(stages), figsize=(5.2 * len(stages), 5.2))
    lo, hi = np.percentile(fac.max_proj, [0.5, 99.8])
    for ax, (name, hw) in zip(axes, stages, strict=True):
        ax.imshow(fac.max_proj, cmap="gray", vmin=lo, vmax=hi)
        ax.scatter(hw[:, 1], hw[:, 0], s=1 if len(hw) > 5000 else 5, c="#ff5a36", lw=0)
        ax.set_title(f"{name}: {len(hw)}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, path)


def outline(A: sps.csr_matrix, H: int, W: int, rel: float = 0.2) -> np.ndarray:
    """(H, W) int map: unit index + 1 on each unit's footprint boundary, 0 elsewhere."""
    out = np.zeros((H, W), np.int32)
    for k in range(A.shape[0]):
        row = A.getrow(k)
        m = np.zeros(H * W, bool)
        m[row.indices[row.data >= rel * row.data.max()]] = True
        m = m.reshape(H, W)
        out[m & ~binary_erosion(m)] = k + 1
    return out


def fig_units(
    fac: Factors, A0: sps.csr_matrix, A: sps.csr_matrix, b: np.ndarray, path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    lo, hi = np.percentile(fac.max_proj, [0.5, 99.8])
    rng = np.random.default_rng(0)
    for ax, M, name in (
        (axes[0], A0, "initial units (after unit merge)"),
        (axes[1], A, "final units (CNMF)"),
    ):
        ax.imshow(fac.max_proj, cmap="gray", vmin=lo, vmax=hi)
        o = outline(sps.csr_matrix(M), fac.H, fac.W_)
        col = rng.uniform(0.3, 1, (M.shape[0] + 1, 4))
        col[:, 3] = 1
        rgba = col[o]
        rgba[o == 0] = 0
        ax.imshow(rgba)
        ax.set_title(f"{name}: {M.shape[0]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    _show(axes[2], b, "background footprint b", "magma")
    _save(fig, path)


def fig_traces(C: np.ndarray, peak: np.ndarray, fps: float, path: Path, n: int = 25) -> np.ndarray:
    """Overview of the ``n`` brightest units; returns their indices (brightest first)."""
    order = np.argsort(peak)[::-1][:n]
    T = C.shape[1]
    nb = 3000
    h = (np.arange(nb) + 0.5) * T / nb / fps / 3600
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [3, 1]})
    z = slice(T // 2, T // 2 + int(120 * fps))
    for i, k in enumerate(order):
        c = C[k] / max(C[k].max(), 1e-6)
        a1.plot(h, c[: T // nb * nb].reshape(nb, -1).max(1) + i, lw=0.4)
        a2.plot(np.arange(z.stop - z.start) / fps, c[z] + i, lw=0.5)
        a1.text(0, i + 0.3, str(k), fontsize=6)
    a1.set_xlabel("hours (max per bin)")
    a1.set_title(f"C of the {n} brightest units, 3 h")
    a2.set_xlabel("seconds")
    a2.set_title("2 min zoom at the middle")
    a1.set_yticks([])
    a2.set_yticks([])
    _save(fig, path)
    return order


def fig_zoom(
    C: np.ndarray,
    units: np.ndarray,
    fps: float,
    path: Path,
    wnd_min: float = 5.0,
    per_hour: int = 3,
) -> None:
    """The same units in ``wnd_min``-minute windows, ``per_hour`` per hour, over the session.

    One panel per window (rows = hours, columns = windows within the hour), traces stacked
    and each normalised by the unit's session maximum, so amplitudes compare across panels.
    """
    T = C.shape[1]
    hours = int(np.ceil(T / fps / 3600))
    step = int(3600 / per_hour * fps)
    wnd = int(wnd_min * 60 * fps)
    fig, axes = plt.subplots(
        hours, per_hour, figsize=(5 * per_hour, 2.2 * len(units) * 0.6 * hours + 1), squeeze=False
    )
    for h in range(hours):
        for j in range(per_hour):
            ax = axes[h, j]
            t0 = h * int(3600 * fps) + j * step
            t1 = min(t0 + wnd, T)
            if t0 >= T:
                ax.axis("off")
                continue
            tt = np.arange(t0, t1) / fps / 60
            for i, k in enumerate(units):
                ax.plot(tt, C[k, t0:t1] / max(C[k].max(), 1e-6) + i, lw=0.5)
                ax.text(tt[0], i + 0.3, str(k), fontsize=6)
            ax.set_title(f"{t0 / fps / 60:.0f}-{t1 / fps / 60:.0f} min", fontsize=9)
            ax.set_yticks([])
            if h == hours - 1:
                ax.set_xlabel("minutes")
    fig.suptitle(f"{len(units)} example units, {wnd_min:.0f}-min zooms {per_hour}x per hour")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Unit browser data + page
# ---------------------------------------------------------------------------


def units_json(
    A: sps.csr_matrix, C: np.ndarray, H: int, W: int, fps: float, nb: int = 2000
) -> dict:
    T = C.shape[1]
    Cb = C[:, : T // nb * nb].reshape(C.shape[0], nb, -1).max(2)
    units = []
    for k in range(A.shape[0]):
        row = A.getrow(k)
        keep = row.data >= 0.2 * row.data.max()
        units.append(
            {
                "px": row.indices[keep].tolist(),
                "trace": np.round(Cb[k] / max(Cb[k].max(), 1e-6), 3).tolist(),
            }
        )
    return {"H": H, "W": W, "hours": T / fps / 3600, "units": units}


PAGE = Path(__file__).with_name("stage_viewer.html")  # template: __TITLE__, __TABLE__


def timing_table(report: dict, ref: dict | None) -> str:
    head = "<tr><th>stage</th><th>s</th>" + ("<th>ref s</th>" if ref else "") + "</tr>"
    refs = {r["stage"]: r["seconds"] for r in ref["stages"]} if ref else {}
    rows = [head]
    for r in report["stages"]:
        cell = (
            f"<td>{refs[r['stage']]:.1f}</td>"
            if r["stage"] in refs
            else ("<td></td>" if ref else "")
        )
        rows.append(f"<tr><td>{stage_label(r['stage'])}</td><td>{r['seconds']:.1f}</td>{cell}</tr>")
    tot = f"<td>{ref['wall_total']:.0f}</td>" if ref else ""
    rows.append(f"<tr><th>wall total</th><th>{report['wall_total']:.0f}</th>{tot}</tr>")
    c = report["counts"]
    rows.append(
        f"<tr><td colspan=3 class=mut>seeds {c['seeds_init']} &rarr; PNR {c['seeds_pnr']} &rarr; "
        f"merged {c['seeds_merged']} &rarr; units {c['units_final']}; peak RSS "
        f"{report['peak_rss'] / 1e9:.1f} / {report['peak_rss_children'] / 1e9:.1f} GB</td></tr>"
    )
    return "<table>" + "".join(rows) + "</table>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--frames", default=None, help="comma-separated frame indices")
    ap.add_argument("--ref", type=Path, default=None, help="another run's report.json to compare")
    a = ap.parse_args()
    run: Path = a.run_dir
    out = run / "viewer"
    out.mkdir(exist_ok=True)

    report = json.loads((run / "report.json").read_text())
    ref = json.loads(a.ref.read_text()) if a.ref else None
    pj = json.loads((run / "params.json").read_text())
    params = Params(**{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    for k in ("size_thres", "motion_bandpass"):
        setattr(params, k, tuple(getattr(params, k)))
    info = io.probe(Path(pj["raw_dir"]))
    fps = info.fps or 20.0
    fac = Factors.load(run / "factors")
    sc = motion.load_sidecar(run / "motion")
    ts = (
        [int(x) for x in a.frames.split(",")]
        if a.frames
        else [0, fac.T // 3, 2 * fac.T // 3, fac.T - 1]
    )
    print("figures: timings, frames, motion, drift, factors", flush=True)
    fig_timings(report, out / "timings.png")
    fig_frames(info, fac, run / "tiles", ts, fps, out / "frames.png")
    fig_motion(sc, fps, out / "motion.png")
    fig_drift(fac, fps, out / "drift.png")
    fig_factors(fac, sc, out / "factors.png")
    lo, hi = np.percentile(fac.max_proj, [0.5, 99.8])
    plt.imsave(out / "maxproj.png", fac.max_proj, cmap="gray", vmin=lo, vmax=hi)

    print("recomputing seeds + init from factors", flush=True)
    import cafast.seeds as seeds_mod

    timer = StageTimer()
    fc = corr.FactorCorr(fac, params.noise_freq)
    s0 = seeds_mod.seeds_init(
        fac.block_max,
        fac.block_edges,
        fac.T,
        params.seeds_wnd,
        params.seeds_stp,
        params.seeds_max_wnd,
        params.seeds_diff_thres,
    )
    s1 = pipeline._filter(seeds_mod.pnr_refine(fac, s0, thres=params.pnr_thres), "mask_pnr")
    s2 = pipeline._filter(
        corr.seeds_merge(fc, s1, fac.max_proj, params.merge_dist, params.merge_corr), "mask_mrg"
    )
    hw = lambda s: s[["height", "width"]].to_numpy()  # noqa: E731
    fig_seeds(
        fac,
        [("seeds_init", hw(s0)), ("after PNR", hw(s1)), ("after merge", hw(s2))],
        out / "seeds.png",
    )
    A0, _ = pipeline._init_units(fac, fc, s2, params, timer)

    print("figures: units, traces, unit browser", flush=True)
    A = sps.load_npz(run / "cnmf" / "A.npz").tocsr()
    C = np.load(run / "cnmf" / "C.npy", mmap_mode="r")
    b = np.load(run / "cnmf" / "b.npy")
    fig_units(fac, A0, A, b, out / "units.png")
    order = fig_traces(np.asarray(C), np.asarray(C).max(1), fps, out / "traces.png")
    fig_zoom(np.asarray(C), order[:10], fps, out / "zoom.png")
    (out / "units.json").write_text(json.dumps(units_json(A, np.asarray(C), fac.H, fac.W_, fps)))
    title = f"{run.name}: {fac.H} x {fac.W_}, {fac.T} frames ({fac.T / fps / 3600:.2f} h)"
    (out / "index.html").write_text(
        PAGE.read_text().replace("__TITLE__", title).replace("__TABLE__", timing_table(report, ref))
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
