"""Calibrate and accept auto-tune on a session with a Minian reference.

For each given rerun: fail rate of every check among Minian-matched vs unmatched units, plus
the acceptance numbers (matched / Minian units, median size10, flagged).
Usage: python bench/autotune_check.py RUN_DIR --minian NPZ [--runs base auto] [--diameter D]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cafast.autotune import run as at
from cafast.autotune import score
from cafast.tune import figures, minian


def check_rates(run_dir: Path, name: str, npz: Path, checks: score.Checks) -> dict:
    run = figures.load_run(run_dir / "reruns" / name)
    mn = minian.load(npz, run)
    f = score.unit_flags(run.A, run.YrA, run.H, run.W, checks)
    matched = np.zeros(len(f), bool)
    matched[mn.ui] = True
    rates = {
        r: {
            "fail_matched": float(f[r][matched].mean()) if matched.any() else None,
            "fail_unmatched": float(f[r][~matched].mean()) if (~matched).any() else None,
        }
        for r in score.REASONS
    }
    return {
        "name": name,
        "units": len(f),
        "matched": int(matched.sum()),
        "minian_units": int(mn.A.shape[0]),
        "size10_median": float(np.median(f["size10"])) if len(f) else None,
        **score.summarize(f),
        "rates": rates,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--minian", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=["base", at.AUTO])
    ap.add_argument("--diameter", type=float, default=None)
    a = ap.parse_args()
    doc = at.load_doc(a.run_dir / "reruns" / at.AUTO)
    if a.diameter:
        checks = score.Checks(a.diameter)
    elif doc is not None:
        checks = at.checks_of(doc)
    else:
        ap.error(f"no reruns/{at.AUTO}/autotune.json in {a.run_dir}; pass --diameter")
    out = [check_rates(a.run_dir, n, a.minian, checks) for n in a.runs]
    text = json.dumps(out, indent=1)
    print(text)
    acc = a.run_dir / "reruns" / at.AUTO / "acceptance.json"
    acc.parent.mkdir(parents=True, exist_ok=True)
    acc.write_text(text + "\n")


if __name__ == "__main__":
    main()
