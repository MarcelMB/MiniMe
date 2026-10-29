"""The Tune page's data: a run's reruns, the auto-tune proposal with its sweep curves, and
the unit checks (flags) and pair statistics of any rerun, computed once into the cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from minime.autotune import run as at
from minime.autotune import score
from minime.autotune.sweep import MIN_DIST_FRAC, SEARCHED
from minime.serve.runs import RunView
from minime.window2 import RERUNS, WINDOW2_PARAMS

CORR_BINS = np.linspace(-0.4, 1.0, 29)
NN_BINS = np.linspace(0, 2.0, 41)


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(v))


def _changed(run: dict[str, Any], rerun: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        k: [_norm(run.get(k)), _norm(rerun[k])]
        for k in WINDOW2_PARAMS
        if k in rerun and _norm(rerun[k]) != _norm(run.get(k))
    }


def _row(run: RunView, d: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"name": d.name, "autotune": (d / at.DOC).is_file()}
    report, error = d / "report.json", d / "error.txt"
    if report.is_file():
        rep = json.loads(report.read_text())
        params = json.loads((d / "params.json").read_text())
        row.update(
            status="done",
            counts=rep.get("counts", {}),
            wall_total=rep.get("wall_total"),
            params={k: _norm(params[k]) for k in WINDOW2_PARAMS if k in params},
            changed=_changed(run.params, params),
        )
    elif error.is_file():
        row.update(status="failed", error=error.read_text().strip().splitlines()[-1:])
    else:
        row["status"] = "running"
    return row


def reruns(run: RunView) -> list[dict[str, Any]]:
    """One row per ``reruns/<name>``: counts and the window-2 parameters changed vs the run
    (``status`` done), or ``failed`` / ``running``; oldest first."""
    root = run.path / RERUNS
    if not root.is_dir():
        return []
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    return [_row(run, d) for d in dirs if (d / "params.json").is_file()]


# ---------------------------------------------------------------------------
# The proposal
# ---------------------------------------------------------------------------


def curves(sweep_csv: Path) -> dict[str, list[dict[str, Any]]]:
    """Per searched parameter: the best good and flagged count at each value tried."""
    df = pd.read_csv(sweep_csv)
    out = {}
    for key in SEARCHED:
        if key not in df:
            continue
        b = at.best_per_value(df, key)
        if b.empty:
            continue
        out[key] = [
            {"value": _norm(v), "good": int(g), "flagged": int(f)}
            for v, g, f in zip(b[key], b["good"], b["flagged"], strict=True)
        ]
    return out


def checks(run: RunView) -> score.Checks | None:
    """The unit checks of the run's last auto-tune, if any."""
    doc = at.load_doc(run.path / RERUNS / at.AUTO)
    return None if doc is None else at.checks_of(doc)


def proposal(run: RunView) -> dict[str, Any] | None:
    """``autotune.json`` of ``reruns/auto`` with the sweep curves; None without one."""
    auto = run.path / RERUNS / at.AUTO
    doc = at.load_doc(auto)
    if doc is None:
        return None
    return {**doc, "curves": curves(auto / "sweep.csv")}


# ---------------------------------------------------------------------------
# Unit checks and pair statistics of a rerun (or the run)
# ---------------------------------------------------------------------------


def _key(what: str, ch: score.Checks, *extra: Any) -> str:
    h = hashlib.sha1(json.dumps([asdict(ch), *extra], sort_keys=True).encode()).hexdigest()
    return f"{what}_{h[:10]}.json"


def flags(run: RunView, ch: score.Checks) -> dict[str, Any] | None:
    return run.cache.load_json(_key("flags", ch))


def compute_flags(run: RunView, ch: score.Checks) -> None:
    f = score.unit_flags(run.A, np.asarray(run.traces("YrA")), run.H, run.W, ch)
    doc = {
        **score.summarize(f),
        "reasons": {r: [int(k) for k in f.index[f[r]]] for r in score.REASONS},
        "checks": asdict(ch),
    }
    run.cache.save_json(_key("flags", ch), doc)


def pairs(run: RunView, ch: score.Checks, min_dist_frac: float) -> dict[str, Any] | None:
    return run.cache.load_json(_key("pairs", ch, min_dist_frac))


def compute_pairs(run: RunView, ch: score.Checks, min_dist_frac: float) -> None:
    """Histograms of what the two thresholds act on (``score.pair_stats``)."""
    s = score.pair_stats(run.A, np.asarray(run.traces("YrA")), run.H, run.W, ch, min_dist_frac)
    hist = lambda a, bins: np.histogram(a, bins)[0].tolist()  # noqa: E731
    doc = {
        "corr_bins": CORR_BINS.round(3).tolist(),
        "near_corr": hist(s["near_corr"], CORR_BINS),
        "far_corr": hist(s["far_corr"], CORR_BINS),
        "nn_bins": NN_BINS.round(3).tolist(),
        "nn_dist": hist(np.minimum(s["nn_dist"], 2), NN_BINS),
        "dup_corr": ch.dup_corr,
        "min_dist_frac": min_dist_frac,
        **{k: s[k] for k in ("n_near", "n_near_above", "n_closer", "far_p99")},
    }
    run.cache.save_json(_key("pairs", ch, min_dist_frac), doc)


def min_dist_frac_of(run: RunView) -> float:
    doc = at.load_doc(run.path / RERUNS / at.AUTO)
    return float((doc or {}).get("min_dist_frac", MIN_DIST_FRAC))
