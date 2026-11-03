"""Reference outputs of MiniMe's seed stage on one window, for checking the Rust core.

Builds the ``Factors`` of a ``fresh`` window run from ``ref_pass1.py``'s and
``ref_pass2.py``'s references (pass 1's block maxima, pass 2's per-tile factors) and runs
the seed stage exactly as ``pipeline.run`` does: ``FactorCorr``, then ``seeds_init``,
``pnr_refine`` (``gmm_refine`` if the session turns it on) and ``seeds_merge``. The opt-in
``gmm_refine`` also runs once on the PNR survivors, outside the pipeline, so its port is
checked too. Writes, under ``processed/stream/ref_seeds_<t0>_<t1>/``:

* ``inputs/``: ``seeds.json`` (``H``, ``W``, ``T``, ``tile`` and the seed parameters), the
  stage's inputs as ``fac_*.npy`` (pass 2's flat ``W``, ``V``, ``mean``, ``sigma``,
  ``energy``, ``rank``; pass 1's ``block_max``, ``block_edges``, ``max_proj``) and a copy
  of ``ref_pass2.py``'s inputs (which hold pass 1's), so one browser run does pass 1, pass 2
  and the seeds on its own factors;
* ``ref/``: ``seeds_init.npy`` ((n, 3) int64 height, width, window count), ``pnr.npy``,
  ``mask_pnr.npy``, ``gmm_pnr.npy`` ((3, 2) weights, means, variances of the PNR mixture),
  ``mask_mrg.npy`` (over the seeds the merge saw), ``merge_pairs.npy`` ((m, 2) seeds within
  ``merge_dist``) with their ``merge_corr.npy``, ``seeds_merged.npy`` ((n, 2) positions),
  ``corr_norm2.npy`` (``FactorCorr.norm2``), the opt-in ``mask_gmm.npy`` and
  ``gmm_refine.npy`` (on the PNR survivors) and ``gmm_draws.npy`` (the three
  ``RandomState(0)`` uniforms a 2-component k-means++ start consumes; the Rust core draws
  its own from MT19937 and must get the same, so nothing needs replaying).

    OMP_NUM_THREADS=1 python experiments/streaming/ref_seeds.py --t0 0 --t1 3000
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from unittest import mock

import numpy as np
import stream as st
from scipy.spatial import cKDTree

from minime import corr, gmm, pipeline, seeds
from minime.factors import Factors, TileFactors
from minime.params import Params
from minime.report import StageTimer
from minime.tiles import tile_grid

FAC = ("W", "V", "mean", "sigma", "energy", "rank")
P1 = ("block_max", "block_edges", "max_proj")


def seeds_meta(params: Params, H: int, W: int, T: int) -> dict:
    return {
        "H": H,
        "W": W,
        "T": T,
        "tile": params.tile,
        "seeds_wnd": params.seeds_wnd,
        "seeds_stp": params.seeds_stp,
        "seeds_max_wnd": params.seeds_max_wnd,
        "seeds_diff_thres": params.seeds_diff_thres,
        "noise_freq": params.noise_freq,
        "pnr_thres": params.pnr_thres,
        "gmm_refine": params.gmm_refine,
        "merge_dist": params.merge_dist,
        "merge_corr": params.merge_corr,
    }


def load_factors(p1: Path, p2: Path, params: Params) -> Factors:
    """``pass2.build_factors`` from the flat reference files."""
    meta = json.loads((p2 / "inputs" / "pass2.json").read_text())
    H, W, T = meta["H"], meta["W"], meta["T"]
    grid = tile_grid(H, W, params.tile)
    f = {n: np.load(p2 / "ref" / f"{n}.npy") for n in FAC}
    tiles, ow, ov, op = [], 0, 0, 0
    for t, r in zip(grid, f["rank"].tolist(), strict=True):
        P = t.pixels
        tiles.append(
            TileFactors(
                f["W"][ow : ow + P * r].reshape(P, r),
                f["V"][ov : ov + T * r].reshape(T, r),
                f["mean"][op : op + P],
                f["sigma"][op : op + P],
                f["energy"][op : op + P],
            )
        )
        ow, ov, op = ow + P * r, ov + T * r, op + P
    b = {n: np.load(p1 / "ref" / f"{n}.npy") for n in P1}
    return Factors(
        H, W, T, grid, tiles, b["max_proj"], b["block_max"], b["block_edges"], {"params": {}}
    )


def kmeanspp_draws(n: int, k: int) -> np.ndarray:
    """The ``RandomState(0)`` values ``gmm._kmeans_plusplus`` consumes for ``n`` values: one
    inside ``choice`` (checked here), then ``2 + log k`` per further centre."""
    m = 1 + (k - 1) * (2 + int(np.log(k)))
    u = np.random.RandomState(gmm.SEED).random_sample(m)
    cdf = np.cumsum(np.full(n, 1 / n))
    cdf /= cdf[-1]
    first = np.random.RandomState(gmm.SEED).choice(n, p=np.full(n, 1 / n))
    assert first == cdf.searchsorted(u[0], side="right"), "choice draws differently"
    return u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=3000)
    a = ap.parse_args()

    s = st.Session(a.session)
    params = s.params()
    p1 = s.out(f"ref_pass1_{a.t0}_{a.t1}")
    p2 = s.out(f"ref_pass2_{a.t0}_{a.t1}")
    fac = load_factors(p1, p2, params)
    out = s.out(f"ref_seeds_{a.t0}_{a.t1}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)

    fits: list[gmm.Gmm1D] = []

    def fit(x: np.ndarray, k: int) -> gmm.Gmm1D:
        fits.append(gmm.fit_gmm_1d(x, k))
        return fits[-1]

    timer = StageTimer()
    t = time.perf_counter()
    with mock.patch.object(seeds, "fit_gmm_1d", fit):
        with timer("corr_setup"):
            fc = corr.FactorCorr(fac, params.noise_freq)
        stages: dict = {}
        unmerged = pipeline.seeds_unmerged(fac, params, timer, stages)
        merged = pipeline.seeds_merged(fac, fc, unmerged, params, timer, stages)
        seconds = time.perf_counter() - t
        n_pipeline_fits = len(fits)
        pnr_kept = pipeline._filter(stages["seeds_pnr"], "mask_pnr")
        with timer("gmm_refine_optin"):
            g = seeds.gmm_refine(fac, pnr_kept)
    # seeds_merge's pairs and their correlations, for diagnosing a different merge
    hw = unmerged[["height", "width"]].to_numpy(np.int64)
    pr = cKDTree(hw).query_pairs(params.merge_dist, output_type="ndarray")
    pr = pr[np.lexsort((pr[:, 1], pr[:, 0]))] if len(pr) else np.zeros((0, 2), np.int64)
    pix = hw[:, 0] * fac.W_ + hw[:, 1]
    pc = fc.pairs(pix[pr[:, 0]], pix[pr[:, 1]]) if len(pr) else np.zeros(0)
    mrg = corr.seeds_merge(fc, unmerged, fac.max_proj, params.merge_dist, params.merge_corr)

    (inp / "seeds.json").write_text(json.dumps(seeds_meta(params, fac.H, fac.W_, fac.T), indent=1))
    for n in FAC:
        np.save(inp / f"fac_{n}.npy", np.load(p2 / "ref" / f"{n}.npy"))
    for n in P1:
        np.save(inp / f"fac_{n}.npy", np.load(p1 / "ref" / f"{n}.npy"))
    for f in (p2 / "inputs").iterdir():
        shutil.copy2(f, inp / f.name)

    init = stages["seeds_init"]
    sp = stages["seeds_pnr"]
    np.save(ref / "seeds_init.npy", init[["height", "width", "seeds"]].to_numpy(np.int64))
    np.save(ref / "pnr.npy", sp["pnr"].to_numpy(np.float64))
    np.save(ref / "mask_pnr.npy", sp["mask_pnr"].to_numpy(bool))
    gp = fits[0]
    np.save(ref / "gmm_pnr.npy", np.stack([gp.weights, gp.means, gp.variances]))
    np.save(ref / "mask_mrg.npy", mrg["mask_mrg"].to_numpy(bool))
    np.save(ref / "merge_pairs.npy", pr.astype(np.int64))
    np.save(ref / "merge_corr.npy", pc)
    np.save(ref / "seeds_merged.npy", merged[["height", "width"]].to_numpy(np.int64))
    np.save(ref / "corr_norm2.npy", fc.norm2)
    np.save(ref / "mask_gmm.npy", g["mask_gmm"].to_numpy(bool))
    gr = fits[-1]
    np.save(ref / "gmm_refine.npy", np.stack([gr.weights, gr.means, gr.variances]))
    np.save(ref / "gmm_draws.npy", kmeanspp_draws(int(sp["mask_pnr"].size), 2))
    summary = {
        "frames": fac.T,
        "seeds_init": len(init),
        "seeds_pnr": int(sp["mask_pnr"].sum()),
        "seeds_gmm": len(unmerged),
        "seeds_merged": len(merged),
        "merge_pairs": len(pr),
        "gmm_refine_optin_kept": int(g["mask_gmm"].sum()),
        "gmm_fits_in_pipeline": n_pipeline_fits,
        "seconds": seconds,
        "stages": {r["stage"]: r["seconds"] for r in timer.rows},
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
