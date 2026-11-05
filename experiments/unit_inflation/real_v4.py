"""Real V4 runs (no truth): browser units before/after the min_dist stitch vs the Python run.

For each length the browser run's window units are stitched again with and without
``min_dist`` (the length's tuned ``merge_min_dist``); units are matched one-to-one to the Python
run's by centroid (<= 5 px). Writes ``out/v4_real.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streaming"))
import classify as cl  # noqa: E402
import prefix_stitch as ps  # noqa: E402
import stream as st  # noqa: E402

from minime import score  # noqa: E402

SSD = Path("/Volumes/Extreme SSD/scaling/data")
PARAMS = Path(
    "/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/MiniMe/bench/compare3/params_tuned"
)


def main() -> None:
    rows = []
    for m in (15, 30, 60):
        proc = SSD / f"v4_{m}min/processed"
        d = float(json.loads((PARAMS / f"v4_{m}min.json").read_text())["merge_min_dist"])
        wins = ps.load_windows(proc / "browser_cpu_tuned", 600 * 600)
        Ap = cl.load_run_A(proc / "compare3_auto/reruns/auto")
        pc = score.footprint_centroids(Ap, 600, 600)
        for rule, kw in (("base", {}), ("min_dist", {"min_dist": d, "shape": (600, 600)})):
            A, _, nw = st.stitch(wins, 0.7, **kw)
            ui, _ = score.match_units(score.footprint_centroids(A, 600, 600), pc)
            single = np.zeros(A.shape[0], bool)
            single[nw == 1] = True
            hit = np.zeros(A.shape[0], bool)
            hit[ui] = True
            rows.append(
                {
                    "minutes": m,
                    "windows": len(wins),
                    "rule": rule,
                    "min_dist": d if kw else 0.0,
                    "units": int(A.shape[0]),
                    "python_units": int(Ap.shape[0]),
                    "matched_python": int(hit.sum()),
                    "single_window": int(single.sum()),
                    "single_window_matched_python": int((single & hit).sum()),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    (Path(__file__).parent / "out/v4_real.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
