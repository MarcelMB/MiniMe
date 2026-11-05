"""``minime score`` of run folders as one JSON line each: ``score_runs.py RAW RUN [RUN ...]``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from minime import score

KEYS = (
    "n_units",
    "n_matched",
    "precision",
    "recall",
    "f1",
    "trace_corr_median",
    "units_on_out_of_focus",
)

if __name__ == "__main__":
    raw = Path(sys.argv[1])
    for run in sys.argv[2:]:
        d = score.score_pipeline_dir(Path(run), raw)
        row = {k: round(d[k], 3) if isinstance(d[k], float) else d[k] for k in KEYS}
        row["late_recall"] = round(d["recall_by_category"]["late"]["recall"], 3)
        print(
            json.dumps(
                {"run": Path(run).parent.name if Path(run).name == "run" else Path(run).name, **row}
            )
        )
