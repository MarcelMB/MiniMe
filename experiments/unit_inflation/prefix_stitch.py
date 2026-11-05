"""Stitch the first k windows of a browser run again (k = 1..n) and classify the units.

The browser run folder keeps every window's units (``win_NN_A_*``, ``win_NN_bright``); the
stitch is re-run in Python (``experiments/streaming/stream.py``'s ``stitch``, which the Rust
stitch matches bit for bit) on each prefix, so one long run gives the whole "units vs session
length" curve. Footprints only (no traces): every class of ``classify.py`` per prefix.

    python prefix_stitch.py --raw SESSION/raw --run BROWSER_FOLDER --out OUT.json [--rule NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streaming"))
import classify as cl  # noqa: E402
import stitch_rules as sr  # noqa: E402
import stream as st  # noqa: E402


def load_windows(folder: Path, P: int) -> list[st.WindowUnits]:
    out = []
    i = 0
    while (folder / f"win_{i:02d}_bright.npy").exists():
        A = cl.load_csr(folder, f"win_{i:02d}_A", P)
        b = np.load(folder / f"win_{i:02d}_bright.npy")
        out.append(st.WindowUnits(A=A, bright=b, t0=0, t1=0, seconds=0.0))
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="browser output folder (win_NN_*)")
    ap.add_argument("--rule", default="base", choices=sorted(sr.RULES))
    ap.add_argument("--cos", type=float, default=0.7)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    truth, At, H, W = cl.load_truth(a.raw)
    wins = load_windows(a.run, H * W)
    A_full = cl.load_run_A(a.run / "run")
    off = cl.offset(a.run / "run", truth, A_full, H, W)
    At_s = cl.shift_footprints(At, H, W, off)
    rows = []
    ks = sorted(set(range(a.step, len(wins) + 1, a.step)) | {len(wins)})
    for k in ks:
        A, _, n_win = sr.RULES[a.rule](wins[:k], a.cos)
        c = cl.classify(A.tocsr(), truth, At_s, off, H, W, 0.5)
        n = cl.counts(c)
        rows.append(
            {"windows": k, "units": int(A.shape[0]), **n, "single_window": int((n_win == 1).sum())}
        )
        print(json.dumps(rows[-1]), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"rule": a.rule, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
