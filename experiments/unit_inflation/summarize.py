"""Markdown tables from ``out/``: per session the classes of every run, and the prefix curves.

python summarize.py TAG [TAG ...]   (TAG as given to analyse.sh / prefix_stitch.py --out)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"
CLASSES = ("matched", "duplicate", "piece", "out_of_focus", "nothing")


def f1(matched: int, units: int, n_true: int) -> float:
    p, r = matched / units, matched / n_true
    return 2 * p * r / (p + r)


def classify_table(tag: str) -> str:
    d = json.loads((OUT / f"{tag}_classify.json").read_text())
    n = d["n_true"]
    rows = [
        "| run | units | " + " | ".join(CLASSES) + " | precision | recall | F1 |",
        "|---" * 9 + "|",
    ]
    for run in ("python", "browser"):
        r = d[run]
        rows.append(
            f"| {run} | {r['units']} | "
            + " | ".join(str(r[c]) for c in CLASSES)
            + f" | {r['matched'] / r['units']:.3f} | {r['matched'] / n:.3f}"
            + f" | {f1(r['matched'], r['units'], n):.3f} |"
        )
    ob, op = d["true_only_browser"], d["true_only_python"]
    cats = ("late", "dim", "small", "overlap", "near_out_of_focus")
    txt = "\n".join(rows)
    txt += (
        f"\n\nTrue cells found by both {d['true_found_by_both']}; only by the browser {ob['n']} ("
        + ", ".join(f"{c} {ob[c]}" for c in cats)
        + f"); only by Python {op['n']}. Late cells ({d['late_recall']['n']}): recall browser "
        f"{d['late_recall']['browser']:.2f}, Python {d['late_recall']['python']:.2f}."
    )
    return txt


def prefix_table(tag: str, rules: list[str]) -> str:
    rows = [
        "| rule | windows | units | " + " | ".join(CLASSES) + " | single-window |",
        "|---" * 9 + "|",
    ]
    for rule in rules:
        p = OUT / f"{tag}_prefix_{rule}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            rows.append(
                f"| {rule} | {r['windows']} | {r['units']} | "
                + " | ".join(str(r[c]) for c in CLASSES)
                + f" | {r['single_window']} |"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        print(f"### {tag}\n")
        if (OUT / f"{tag}_classify.json").exists():
            print(classify_table(tag) + "\n")
        print(prefix_table(tag, ["base", "mindistbp5", "mindist5", "attach40"]) + "\n")
