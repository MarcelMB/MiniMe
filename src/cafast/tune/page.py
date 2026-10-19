"""HTML of the ``cafast tune`` page: parameter form, run list, figures per stage, units table."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from cafast.tune import figures

if TYPE_CHECKING:
    from cafast.tune.server import TuneApp

# (group, [(key, hint)]) in pipeline order
FORM = [
    (
        "Seeds",
        [
            ("seeds_wnd", "frames per rolling max window (must fall on pass-1 blocks)"),
            ("seeds_stp", "frames between window centres"),
            ("seeds_max_wnd", "largest local-max disk radius (px), about a cell radius"),
            ("seeds_diff_thres", "min max-min contrast around a local max (uint8 levels)"),
            ("noise_freq", "low-pass cutoff (fraction of fs) for correlations and PNR"),
            ("pnr_thres", "seed peak-to-noise threshold, or auto (2-GMM)"),
            ("gmm_refine", "extra 2-GMM filter on seed amplitude (true / false)"),
            ("merge_dist", "seeds closer than this (px) may merge ..."),
            ("merge_corr", "... if their traces correlate above this"),
        ],
    ),
    (
        "Initialisation",
        [
            ("inita_corr", "footprint = window pixels correlating above this with the seed"),
            ("inita_wnd", "window radius (px)"),
            ("init_merge_corr", "merge overlapping initial units above this trace corr"),
        ],
    ),
    (
        "CNMF",
        [
            ("dl_wnd", "a footprint may grow this far (px) per spatial update"),
            ("sparse_penal", "L1 penalty (x T x pixel sigma): higher = tighter footprints"),
            ("size_thres", "min, max pixels kept after a spatial update (blank = none)"),
            ("n_iter", "spatial + temporal iterations"),
            ("loop_merge_corr", "merge after each iteration above this trace corr"),
        ],
    ),
]
PREVIEW_FORM = [
    ("median_ksize", "median blur kernel (odd; <= 1 skips)"),
    ("tophat_wnd", "tophat half-width (px; <= 0 skips)"),
    ("motion_bandpass", "registration band-pass sigmas lo, hi (px)"),
]
# (figure, title, parameters that change it)
SHARED = [
    ("max_proj.png", "Max projection", "pass 1"),
    ("motion.png", "Motion", "pass 1: motion_*"),
    ("noise.png", "Noise and mean per pixel", "pass 1-2; sigma scales the sparse penalty"),
]
STAGES = [
    ("seeds.png", "Seeds: init, PNR, merge", "seeds_*, noise_freq, pnr_thres, merge_*"),
    ("init.png", "initA and init merge", "inita_corr, inita_wnd, init_merge_corr"),
    (
        "cnmf.png",
        "CNMF: each spatial update and merge",
        "dl_wnd, sparse_penal, size_thres, n_iter, loop_merge_corr",
    ),
    ("sizes.png", "Footprint size and fragmentation", "all"),
    ("minian.png", "Final vs Minian", "all"),
    ("traces.png", "Example traces", "all"),
]

STATIC = Path(__file__).with_name("static")


def _e(v: Any) -> str:
    return html.escape(str(v), quote=True)


def fmt_value(v: Any) -> str:
    """A parameter value as typed in the form: ``true``, ``10, `` (size_thres), ``0.8``."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, tuple | list):
        return ", ".join("" if x is None else fmt_value(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))  # 8.0 -> "8"; a float field parses it back to 8.0
    return str(v)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _field(key: str, hint: str, value: Any, base: Any) -> str:
    v, b = fmt_value(value), fmt_value(base)
    return (
        f'<label><b>{key}</b> <span class="base">run: {_e(b)}</span>'
        f'<input type="text" name="{key}" value="{_e(v)}" data-base="{_e(b)}">'
        f'<span class="hint">{_e(hint)}</span></label>'
    )


def _sidebar(app: TuneApp, view: str | None) -> str:
    p, base = app.params_of(view), app.session.base
    out = ['<form id="runform">', f"<h2>Window 2 parameters (from {_e(view or 'run')})</h2>"]
    for group, keys in FORM:
        out.append(f"<h2>{group}</h2>")
        out += [_field(k, h, getattr(p, k), getattr(base, k)) for k, h in keys]
    out.append(
        f'<label><b>name</b><input type="text" name="name" placeholder="{app.next_name()}">'
        '<span class="hint">rerun name (blank = next free); an existing name is replaced</span>'
        '</label><button type="submit">Rerun window 2</button><div id="status"></div></form>'
    )
    if view:
        out.append(
            f'<h2>Save parameters</h2><form id="saveform"><input type="hidden" name="run" '
            f'value="{_e(view)}"><label><b>file</b> <span class="hint">in RUN_DIR/tune/; '
            "full set, for cafast run --params</span>"
            f'<input type="text" name="file" value="params_{_e(view)}.json"></label>'
            '<button type="submit" class="ghost">Save params of this run</button>'
            '<div id="saved" class="note"></div></form>'
        )
    out.append('<h2>Pass-1 preview (needs a full run)</h2><form id="pvform">')
    out += [_field(k, h, getattr(p, k), getattr(base, k)) for k, h in PREVIEW_FORM]
    out.append(
        '<button type="submit" class="ghost">Preview on first 300 frames</button>'
        '<div id="pvmsg" class="note"></div></form>'
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main column
# ---------------------------------------------------------------------------


def _changed_text(ch: dict[str, Any]) -> str:
    if not ch:
        return "(run params)"
    return ", ".join(f"{k} {fmt_value(a)}&rarr;{fmt_value(b)}" for k, (a, b) in ch.items())


def _num(v: Any, fmt: str = "{:.0f}") -> str:
    return "" if v is None else fmt.format(v)


def _runs_table(app: TuneApp, view: str | None, cmp: str | None) -> str:
    mn = app.minian_npz is not None
    head = [
        "view",
        "cmp",
        "name",
        "changed vs run",
        "seeds init / PNR / merged",
        "initA",
        "units",
        "size >10 % (median px)",
        "in pieces",
        "wall s",
    ]
    if mn:
        head[9:9] = ["Minian matched", "trace corr"]  # after "in pieces"
    rows = []
    for r in app.runs():
        n = r["name"]
        cls = "view" if n == view else "cmp" if n == cmp else ""
        cells = [
            f'<a href="/?run={_e(n)}&cmp={_e(view or "")}">view</a>',
            f'<a href="/?run={_e(view or n)}&cmp={_e(n)}">cmp</a>' if view and n != view else "",
            f"<b>{_e(n)}</b>",
            f'<span class="chg">{_changed_text(r.get("changed", {}))}</span>',
        ]
        if "error" in r:
            cells.append(f'<span class="err">failed: {_e(r["error"])}</span>')
            cells += [""] * (len(head) - len(cells))
        elif r.get("pending"):
            cells += ["running ..."] + [""] * (len(head) - len(cells) - 1)
        else:
            cells += [
                f"{r['seeds_init']} / {r['seeds_pnr']} / {r['seeds_merged']}",
                _num(r.get("units_initA")),
                _num(r.get("units_final")),
                _num(r.get("size10_median")),
                _num(r.get("in_pieces")),
            ]
            if mn:
                cells += [
                    _num(r.get("minian_matched")) + f" / {_num(r.get('minian_units'))}",
                    _num(r.get("trace_corr_median"), "{:.2f}"),
                ]
            cells.append(_num(r.get("wall_s"), "{:.1f}"))
        tds = "".join(
            f'<td class="{"chg l" if i == 3 else "l" if i < 3 else ""}">{c}</td>'
            for i, c in enumerate(cells)
        )
        rows.append(f'<tr class="{cls}">{tds}</tr>')
    ths = "".join(f'<th class="{"l" if i < 4 else ""}">{h}</th>' for i, h in enumerate(head))
    ref = ""
    if mn:
        ref = f'<div class="note">Minian reference: {_e(app.minian_npz)}</div>'
    table = f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    return f'<div class="scroll">{table}</div>{ref}'


def _img(app: TuneApp, run: str, fig: str) -> str:
    p = app.figure_path(run, fig)
    if p is None:
        return '<div class="note">not drawn</div>'
    return f'<img src="/fig/{_e(run)}/{fig}?v={int(p.stat().st_mtime)}">'


def _figure(app: TuneApp, fig: str, title: str, why: str, view: str, cmp: str | None) -> str:
    if app.figure_path(view, fig) is None and (cmp is None or app.figure_path(cmp, fig) is None):
        return ""
    body = _img(app, view, fig)
    if cmp:
        body = (
            f'<span class="tag view">{_e(view)}</span>{body}'
            f'<span class="tag cmp">{_e(cmp)}</span>{_img(app, cmp, fig)}'
        )
    return f'<div class="fig"><h3>{title}</h3><div class="why">changed by: {why}</div>{body}</div>'


def _units(app: TuneApp, view: str) -> str:
    run = app.run_data(view)
    df = figures.unit_table(run, app.overlay(run))
    ths = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    rows = []
    for k, r in enumerate(df.itertuples(index=False)):
        tds = "".join(f"<td>{'' if pd.isna(v) else _e(v)}</td>" for v in r)
        rows.append(f'<tr data-k="{k}">{tds}</tr>')
    return (
        '<div class="fig"><h3>Units (final)</h3><div class="why">click a column to sort, a row '
        "to show the unit; size10 = pixels above 10 % of peak, parts = pieces above 10 %, "
        "pnr = trace peak-to-noise</div>"
        f'<div class="unitwrap"><div id="units"><table><thead><tr>{ths}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        '</table></div><div id="unitfig"><img id="unitimg" hidden></div></div></div>'
    )


def render(app: TuneApp, view: str | None, cmp: str | None) -> str:
    names = app.names()
    view = view if view in names else (names[-1] if names else None)
    cmp = cmp if cmp in names and cmp != view else None
    main = ["<h2>Runs</h2>", _runs_table(app, view, cmp)]
    main.append("<details><summary>Pass 1 (shared by every rerun)</summary>")
    main += [_figure(app, f, t, w, "_shared", None) for f, t, w in SHARED]
    pv = app.figure_path("_shared", "preview.png")
    main.append(
        '<div class="fig"><h3>Pass-1 preview</h3><div class="why">form at the bottom left</div>'
        f'<img id="pvimg" src="/fig/_shared/preview.png" {"" if pv else "hidden"}></div>'
        "</details>"
    )
    if view and (app.reruns_dir / view / "figures").is_dir():
        main.append(f"<h2>Stages of {_e(view)}" + (f" vs {_e(cmp)}" if cmp else "") + "</h2>")
        main += [_figure(app, f, t, w, view, cmp) for f, t, w in STAGES]
        main.append(_units(app, view))
    elif view:
        main.append('<p class="note">figures are being drawn ...</p>')
    title = f"{app.run_dir.parent.parent.name} / {app.run_dir.name}"
    css = (STATIC / "tune.css").read_text()
    js = (
        f"const VIEW = {json.dumps(view)}, CMP = {json.dumps(cmp)};\n"
        + (STATIC / "tune.js").read_text()
    )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>CaFast tune</title><style>{css}</style></head><body>"
        f'<header><h1>CaFast tune</h1><span class="meta">{_e(title)}</span>'
        f'<span class="meta">{_e(app.run_dir)}</span></header>'
        f'<div class="wrap"><aside>{_sidebar(app, view)}</aside>'
        f"<main>{''.join(main)}</main></div><script>{js}</script></body></html>"
    )
