"""Compare a browser run with a Python run of the same session: agreement, scores, speed.

    python compare.py --python RUN --browser RUN --out DIR [--raw RAW]
        [--mem-python mem.json] [--mem-browser mem.json] [--job-browser job.json]

Run it with main's Python (the main checkout's .venv, no PYTHONPATH), so the scores are main's
``minime.score``. With ``--raw`` (a synthetic session: ``meta.json`` beside it) both runs are
scored against the truth. ``mem.json`` files come from ``measure.py``, ``job.json`` from
``run_browser.mjs``. Writes ``metrics.json``, ``stages.csv``, ``figure.png`` and ``README.md``
into ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from minime.score import (
    MAX_DIST_PX,
    footprint_centroids,
    match_units,
    row_corr,
    score_pipeline_dir,
    score_run,
)

PY_COLOR, BR_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated)
COS_AGREE = 0.7  # the stitch's footprint rule: same unit when the cosine is at least this

# Python stages that correspond to a browser stage; everything else is listed on its own.
STAGE_MAP = {
    "motion": ["motion"],
    "pass1": ["pass1"],
    "pass2": ["pass2"],
    "seeds": ["corr_setup", "seeds_init", "pnr_refine", "seeds_merge"],
    "init": ["initA", "initC", "unit_merge"],
    "cnmf": ["cnmf"],
}
FRAME_STAGES = {"motion", "pass1", "pass2", "stream"}  # stages that read every frame
PARAMS_IGNORED = {"raw_dir", "fps", "source", "workers"}


# --- loading ----------------------------------------------------------------------------


def load_run(d: Path) -> dict:
    """A run folder's footprints, traces, motion, images and records."""
    meta = json.loads((d / "factors" / "meta.json").read_text())
    H, W = int(meta["H"]), int(meta["W"])
    A = sparse.csr_matrix(sparse.load_npz(d / "cnmf" / "A.npz"), dtype=np.float64)
    return {
        "dir": d,
        "H": H,
        "W": W,
        "A": A,
        "C": np.load(d / "cnmf" / "C.npy", mmap_mode="r"),
        "shifts": np.load(d / "motion" / "arrays.npz")["shifts"].astype(np.float64),
        "max_proj": np.load(d / "factors" / "max_proj.npy"),
        "report": json.loads((d / "report.json").read_text()),
        "params": json.loads((d / "params.json").read_text()),
    }


def read_json(p: Path | None) -> dict | None:
    return json.loads(p.read_text()) if p else None


# --- agreement --------------------------------------------------------------------------


def unit_norm(A: sparse.csr_matrix) -> sparse.csr_matrix:
    n = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sparse.diags(1.0 / n) @ A


def match_footprints(py: dict, br: dict) -> dict:
    """One-to-one matching by centroid (``minime score``'s rule) and best cosine each way."""
    cp = footprint_centroids(py["A"], py["H"], py["W"])
    cb = footprint_centroids(br["A"], br["H"], br["W"])
    ip, ib = match_units(cp, cb, MAX_DIST_PX)
    Ap, Ab = unit_norm(py["A"]), unit_norm(br["A"])
    cos = np.asarray(Ap[ip].multiply(Ab[ib]).sum(axis=1)).ravel()
    dist = np.hypot(*(cp[ip] - cb[ib]).T)
    cross = (Ap @ Ab.T).tocsr()  # (Kp, Kb) cosines, sparse: footprints rarely overlap
    best_p = cross.max(axis=1).toarray().ravel() if cross.shape[1] else np.zeros(cross.shape[0])
    best_b = cross.max(axis=0).toarray().ravel() if cross.shape[0] else np.zeros(cross.shape[1])
    Kp, Kb = py["A"].shape[0], br["A"].shape[0]
    return {
        "ip": ip,
        "ib": ib,
        "cos": cos,
        "metrics": {
            "units_python": int(Kp),
            "units_browser": int(Kb),
            "matched_one_to_one": len(ip),
            "match_rule": f"greedy one-to-one by centroid distance <= {MAX_DIST_PX} px",
            "matched_fraction_of_python": len(ip) / Kp if Kp else None,
            "matched_fraction_of_browser": len(ip) / Kb if Kb else None,
            "centroid_dist_px": quantiles(dist),
            "footprint_cosine": quantiles(cos),
            f"matched_with_cosine_ge_{COS_AGREE}": int((cos >= COS_AGREE).sum()),
            f"python_units_with_a_browser_unit_cos_ge_{COS_AGREE}": float(
                (best_p >= COS_AGREE).mean()
            )
            if Kp
            else None,
            f"browser_units_with_a_python_unit_cos_ge_{COS_AGREE}": float(
                (best_b >= COS_AGREE).mean()
            )
            if Kb
            else None,
            "centroid_offset_px_median": [float(v) for v in np.median(cp[ip] - cb[ib], axis=0)]
            if len(ip)
            else None,
        },
    }


def quantiles(x: np.ndarray) -> dict | None:
    if not len(x):
        return None
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "mean": float(x.mean()),
        **{k: float(v) for k, v in zip(["q10", "q25", "median", "q75", "q90"], q, strict=True)},
    }


def trace_agreement(py: dict, br: dict, ip: np.ndarray, ib: np.ndarray) -> tuple[np.ndarray, dict]:
    if not len(ip):
        return np.zeros(0), {"trace_corr": None}
    r = row_corr(np.asarray(py["C"][ip], np.float32), np.asarray(br["C"][ib], np.float32))
    return r, {
        "trace_corr": quantiles(r[np.isfinite(r)]),
        "trace_corr_ge_0.9": float((r >= 0.9).mean()),
    }


def motion_agreement(py: dict, br: dict) -> dict:
    sp, sb = py["shifts"], br["shifts"]
    if sp.shape != sb.shape:
        return {"motion": f"shapes differ: {sp.shape} vs {sb.shape}"}
    d = sb - sp
    return {
        "motion": {
            "frames": len(sp),
            "max_abs_diff_px": [float(v) for v in np.abs(d).max(axis=0)],
            "median_abs_diff_px": [float(v) for v in np.median(np.abs(d), axis=0)],
            "mean_diff_px": [float(v) for v in d.mean(axis=0)],
            "corr": [float(np.corrcoef(sp[:, k], sb[:, k])[0, 1]) for k in range(sp.shape[1])],
            "python_max_abs_shift_px": float(np.abs(sp).max()),
            "browser_max_abs_shift_px": float(np.abs(sb).max()),
        }
    }


def params_diff(py: dict, br: dict) -> dict:
    a, b = py["params"], br["params"]
    keys = sorted((set(a) | set(b)) - PARAMS_IGNORED)
    return {k: {"python": a.get(k), "browser": b.get(k)} for k in keys if a.get(k) != b.get(k)}


# --- scores -----------------------------------------------------------------------------


SCORE_KEYS = [
    "f1",
    "recall",
    "precision",
    "recall_detectable",
    "trace_corr_median",
    "n_units",
    "n_matched",
    "alignment_ok",
]


def scores(py: dict, br: dict, raw: Path) -> dict:
    """``minime score`` of both runs; the browser keeps only ``C.npy`` (no ``YrA.npy``)."""
    meta = raw.parent / "meta.json"
    out = {
        "python (minime score: YrA)": score_pipeline_dir(py["dir"], raw),
        "python (C, like the browser)": score_run(
            py["A"], np.asarray(py["C"]), py["H"], py["W"], meta, py["shifts"]
        ),
        "browser (C)": score_run(
            br["A"], np.asarray(br["C"]), br["H"], br["W"], meta, br["shifts"]
        ),
    }
    return {k: {s: v.get(s) for s in SCORE_KEYS} for k, v in out.items()}


# --- speed ------------------------------------------------------------------------------


def stage_seconds(report: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in report.get("stages", []):
        out[s["stage"]] = out.get(s["stage"], 0.0) + float(s["seconds"])
    return out


def speed_table(py: dict, br: dict) -> list[dict]:
    sp, sb = stage_seconds(py["report"]), stage_seconds(br["report"])
    T = int(py["shifts"].shape[0])
    rows, used = [], set()
    for stage, names in STAGE_MAP.items():
        p = sum(sp.get(n, 0.0) for n in names) if any(n in sp for n in names) else None
        used.update(names)
        rows.append(row(stage, "+".join(n for n in names if n in sp), p, sb.get(stage), T))
    for stage in ("stitch", "stream"):
        rows.append(row(stage, "", None, sb.get(stage), T, note="browser only"))
    for name, sec in sp.items():
        if name not in used:
            rows.append(row(name, name, sec, None, T, note="Python only"))
    rows.append(
        row(
            "wall total",
            "",
            py["report"].get("wall_total"),
            br["report"].get("wall_total"),
            T,
        )
    )
    return rows


def row(
    stage: str, py_names: str, p: float | None, b: float | None, T: int, note: str = ""
) -> dict:
    fps = stage in FRAME_STAGES or stage == "wall total"
    return {
        "stage": stage,
        "python_stages": py_names,
        "python_s": p,
        "browser_s": b,
        "browser_over_python": b / p if p and b else None,
        "python_fps": T / p if fps and p else None,
        "browser_fps": T / b if fps and b else None,
        "note": note,
    }


def memory(py: dict, br: dict, mem_py: dict | None, mem_br: dict | None) -> dict:
    rp, rb = py["report"], br["report"]
    return {
        "python_peak_rss_main_gib": gib(rp.get("peak_rss")),
        "python_peak_rss_children_gib": gib(rp.get("peak_rss_children")),
        "python_tree_peak_rss_gib (sampled)": gib(mem_py and mem_py["peak_rss_tree_bytes"]),
        "browser_wasm_heap_peak_mb": rb.get("wasm_heap_peak_mb"),
        "browser_chromium_peak_rss_gib (sampled)": gib(mem_br and mem_br["peak_rss_tree_bytes"]),
        "python_workers": rp.get("workers"),
        "browser_workers": rb.get("workers"),
    }


def gib(b: float | None) -> float | None:
    return None if b is None else round(b / 2**30, 3)


def hardware() -> str:
    def sysctl(k: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", k], check=False, capture_output=True, text=True
        ).stdout.strip()

    mem = int(sysctl("hw.memsize") or 0) / 2**30
    cpu, cores = sysctl("machdep.cpu.brand_string"), sysctl("hw.ncpu")
    return f"{cpu}, {cores} cores, {mem:.0f} GB, macOS {platform.mac_ver()[0]}"


# --- figure -----------------------------------------------------------------------------


def contours(ax: plt.Axes, A: sparse.csr_matrix, H: int, W: int, color: str) -> None:
    """Each footprint's outline at 30 % of its peak, drawn on its bounding box only."""
    for k in range(A.shape[0]):
        row_ = A.getrow(k)
        if row_.nnz == 0:
            continue
        pix, val = row_.indices, row_.data
        y, x = pix // W, pix % W
        y0, y1, x0, x1 = y.min() - 1, y.max() + 2, x.min() - 1, x.max() + 2
        img = np.zeros((y1 - y0, x1 - x0))
        img[y - y0, x - x0] = val
        ax.contour(
            np.arange(x0, x1),
            np.arange(y0, y1),
            img,
            levels=[0.3 * val.max()],
            colors=color,
            linewidths=0.6,
        )


def figure(
    py: dict,
    br: dict,
    ip: np.ndarray,
    ib: np.ndarray,
    r: np.ndarray,
    fps: float,
    out: Path,
) -> dict:
    fig = plt.figure(figsize=(13, 9.5), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[:, 0])
    mp = py["max_proj"]
    ax.imshow(mp, cmap="gray", vmin=np.percentile(mp, 1), vmax=np.percentile(mp, 99.5))
    contours(ax, py["A"], py["H"], py["W"], PY_COLOR)
    contours(ax, br["A"], br["H"], br["W"], BR_COLOR)
    ax.set_title(
        f"Python's max projection · Python {py['A'].shape[0]} units (blue), "
        f"browser {br['A'].shape[0]} (orange)",
        fontsize=10,
        loc="left",
    )
    pix = np.concatenate([py["A"].indices, br["A"].indices])
    if len(pix):  # zoom to where the units are
        y, x = pix // py["W"], pix % py["W"]
        ax.set_xlim(max(x.min() - 15, 0), min(x.max() + 15, py["W"] - 1))
        ax.set_ylim(min(y.max() + 15, py["H"] - 1), max(y.min() - 15, 0))
    ax.set_axis_off()
    ax.plot([], [], color=PY_COLOR, label="Python")
    ax.plot([], [], color=BR_COLOR, label="browser")
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    picked = {}
    if len(r):
        order = np.argsort(np.nan_to_num(r, nan=-2.0))
        picked = {
            "best": order[-1],
            "median": order[len(order) // 2],
            "worst": order[0],
        }
    for i, (label, m) in enumerate(picked.items()):
        a = fig.add_subplot(gs[i, 1])
        tp = np.asarray(py["C"][ip[m]], np.float64)
        tb = np.asarray(br["C"][ib[m]], np.float64)
        t = np.arange(len(tp)) / fps / 60
        a.plot(t, tp, color=PY_COLOR, linewidth=0.8, label="Python")
        a.plot(t, tb, color=BR_COLOR, linewidth=0.8, label="browser", alpha=0.85)
        a.set_title(
            f"{label} matched pair · r = {r[m]:.3f} · Python unit {ip[m]}, browser unit {ib[m]}",
            fontsize=9,
            loc="left",
        )
        a.set_xlabel("time (min)", fontsize=8)
        a.set_ylabel("C (a.u.)", fontsize=8)
        a.tick_params(labelsize=8)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e6e3", linewidth=0.6)
        if i == 0:
            a.legend(fontsize=8, loc="upper right", frameon=False)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {
        k: {"python_unit": int(ip[m]), "browser_unit": int(ib[m]), "r": float(r[m])}
        for k, m in picked.items()
    }


# --- report -----------------------------------------------------------------------------


def write_stages_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(v: float | None, nd: int = 1) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def q(d: dict | None, key: str, nd: int) -> str:
    """``fmt`` of one quantile of a ``quantiles`` dict (``–`` when there were no pairs)."""
    return fmt(d and d[key], nd)


def pair(v: list[float], nd: int) -> str:
    return f"{fmt(v[0], nd)} / {fmt(v[1], nd)}"


def readme(m: dict, args: argparse.Namespace) -> str:
    fm, tr, mo, mem = m["footprints"], m["traces"], m["motion"], m["memory"]
    raw = (
        [f"- raw (scored against the truth in `{args.raw.parent}/meta.json`): `{args.raw}`"]
        if args.raw
        else []
    )
    agree = fm[f"matched_with_cosine_ge_{COS_AGREE}"]
    ge09 = 100 * (tr.get("trace_corr_ge_0.9") or 0)
    lines = [
        f"# Browser vs Python run: {args.python.parent.parent.name}",
        "",
        "Inputs → outputs:",
        f"- Python run (main's `minime run`, default parameters): `{args.python}`",
        f"- browser run (browser build, headless Chromium, default parameters): `{args.browser}`",
        *raw,
        "- `metrics.json` (everything below), `stages.csv` (seconds per stage), `figure.png`",
        "  (contours on the max projection; best, median and worst matched traces). Made by",
        "  `experiments/browser_vs_python/compare.py` on branch `browser-app`.",
        "",
        "## Agreement (no ground truth)",
        (
            f"- Units: Python {fm['units_python']}, browser {fm['units_browser']}; "
            f"{fm['matched_one_to_one']} matched one-to-one ({fm['match_rule']}): "
            f"{fmt(100 * (fm['matched_fraction_of_python'] or 0))} % of Python's, "
            f"{fmt(100 * (fm['matched_fraction_of_browser'] or 0))} % of the browser's."
        ),
        (
            "- Matched pairs: centroid distance median "
            f"{q(fm['centroid_dist_px'], 'median', 2)} px, footprint cosine median "
            f"{q(fm['footprint_cosine'], 'median', 3)}; "
            f"{agree} pairs with cosine ≥ {COS_AGREE}."
        ),
        (
            f"- Trace correlation of matched pairs (C vs C): median "
            f"{q(tr['trace_corr'], 'median', 3)}, q25 {q(tr['trace_corr'], 'q25', 3)}; "
            f"≥ 0.9 for {fmt(ge09)} %."
        ),
    ]
    if isinstance(mo, dict):
        lines.append(
            f"- Motion shifts (y / x): max |Δ| {pair(mo['max_abs_diff_px'], 3)} px, "
            f"median |Δ| {pair(mo['median_abs_diff_px'], 4)} px, r {pair(mo['corr'], 4)}."
        )
    if m["params_diff"]:
        lines.append(f"- Parameters that differ: {', '.join(m['params_diff'])} (metrics.json).")
    if m.get("scores"):
        lines += [
            "",
            "## Scores against the truth (main's `minime.score`)",
            "",
            "| run | F1 | recall | precision | detectable recall | trace corr median | units |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, s in m["scores"].items():
            cells = [s[c] for c in ("f1", "recall", "precision", "recall_detectable")]
            cells.append(s["trace_corr_median"])
            lines.append(f"| {k} | {' | '.join(fmt(c, 3) for c in cells)} | {s['n_units']} |")
    lines += [
        "",
        "## Speed per stage (seconds)",
        "",
        "| stage | Python | browser | browser / Python | Python fps | browser fps | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in m["stages"]:
        names = r["python_stages"]
        name = r["stage"] + (f" ({names})" if names and names != r["stage"] else "")
        cells = [fmt(r["python_s"]), fmt(r["browser_s"]), fmt(r["browser_over_python"])]
        cells += [fmt(r["python_fps"], 0), fmt(r["browser_fps"], 0), r["note"]]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    tree_py = mem["python_tree_peak_rss_gib (sampled)"]
    tree_br = mem["browser_chromium_peak_rss_gib (sampled)"]
    lines += [
        "",
        (
            f"Memory: Python peak RSS {fmt(mem['python_peak_rss_main_gib'], 2)} GiB main + "
            f"{fmt(mem['python_peak_rss_children_gib'], 2)} GiB largest worker (sampled process "
            f"tree peak {fmt(tree_py, 2)} GiB); browser wasm heap peak "
            f"{fmt(mem['browser_wasm_heap_peak_mb'], 0)} MB (sampled Chromium tree peak "
            f"{fmt(tree_br, 2)} GiB; shared pages counted once per process)."
        ),
        (
            f"Hardware: {m['hardware']}; Python {mem['python_workers']} workers, "
            f"browser {mem['browser_workers']} worker."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--browser", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--mem-python", type=Path)
    ap.add_argument("--mem-browser", type=Path)
    ap.add_argument("--job-browser", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    py, br = load_run(args.python), load_run(args.browser)
    fm = match_footprints(py, br)
    r, tr = trace_agreement(py, br, fm["ip"], fm["ib"])
    fps = float(py["params"].get("fps") or br["params"].get("fps") or 20.0)
    m = {
        "footprints": fm["metrics"],
        "traces": tr,
        **motion_agreement(py, br),
        "params_diff": params_diff(py, br),
        "scores": scores(py, br, args.raw) if args.raw else None,
        "stages": speed_table(py, br),
        "memory": memory(py, br, read_json(args.mem_python), read_json(args.mem_browser)),
        "wall_measured_s": {
            "python": (read_json(args.mem_python) or {}).get("wall_s"),
            "browser_click_to_done": (read_json(args.job_browser) or {}).get(
                "wall_click_to_done_s"
            ),
        },
        "hardware": hardware(),
        "counts": {
            "python": py["report"].get("counts"),
            "browser": br["report"].get("counts"),
        },
    }
    m["figure_pairs"] = figure(py, br, fm["ip"], fm["ib"], r, fps, args.out / "figure.png")
    (args.out / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    write_stages_csv(m["stages"], args.out / "stages.csv")
    (args.out / "README.md").write_text(readme(m, args))
    print((args.out / "README.md").read_text())


if __name__ == "__main__":
    main()
