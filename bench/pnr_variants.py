"""Time the PNR seed filter serial vs threaded on a run's saved factors.

    .venv/bin/python bench/pnr_variants.py RUN_DIR

Rebuilds the rolling-window seeds from ``RUN_DIR/factors`` with the run's parameters, then
scores them as:

- ``ptp_serial``: Minian's ptp score, one thread (the code before threading)
- ``ptp_threads``: the same score, batches on all cores (the default now)

Each score goes through ``pnr_refine``'s ``thres="auto"`` 2-GMM split. Reports seconds of
the scoring, seeds kept, and the agreement of each mask with ``ptp_serial`` (kept by both /
kept by only one). Writes ``RUN_DIR/pnr_variants.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caanfast import seeds as sd  # noqa: E402
from caanfast.factors import Factors  # noqa: E402
from caanfast.params import Params  # noqa: E402

Q = (0.1, 99.9)


def gmm_mask(pnr: np.ndarray) -> np.ndarray:
    """``pnr_refine``'s ``thres="auto"`` split."""
    dat = np.log(np.maximum(pnr, np.finfo(np.float64).tiny))[:, None]
    gmm = GaussianMixture(n_components=2, random_state=0).fit(dat)
    return gmm.predict(dat) == np.argmax(gmm.means_.ravel())


def timed(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    return fn(), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    p = Params.from_json(a.run / "params.json")
    f = Factors.load(a.run / "factors")
    s = sd.seeds_init(
        f.block_max,
        f.block_edges,
        f.T,
        p.seeds_wnd,
        p.seeds_stp,
        p.seeds_max_wnd,
        p.seeds_diff_thres,
    )
    print(f"{len(s)} seeds, {f.T} frames", flush=True)
    pix = sd.seed_pixels(f, s)
    sigma = f.sigma_flat()[pix].astype(np.float64)
    sigma[sigma <= 0] = np.median(f.sigma_flat())

    scores = {
        "ptp_serial": lambda: sd.seed_ptp(f, pix, Q, threads=1) / (6.18 * sigma),
        "ptp_threads": lambda: sd.seed_ptp(f, pix, Q, threads=os.cpu_count()) / (6.18 * sigma),
    }
    out: dict = {"seeds": len(s), "frames": f.T, "variants": {}}
    ref = ref_pnr = None
    for name, fn in scores.items():
        pnr, t = timed(fn)
        mask = gmm_mask(pnr)
        if ref is None:
            ref, ref_pnr = mask, pnr
        out["variants"][name] = {
            "seconds": t,
            "kept": int(mask.sum()),
            "kept_both": int((mask & ref).sum()),
            "only_this": int((mask & ~ref).sum()),
            "only_ptp_serial": int((~mask & ref).sum()),
            "spearman_vs_ptp": float(
                np.corrcoef(np.argsort(np.argsort(pnr)), np.argsort(np.argsort(ref_pnr)))[0, 1]
            ),
        }
        print(name, out["variants"][name], flush=True)
    (a.run / "pnr_variants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
