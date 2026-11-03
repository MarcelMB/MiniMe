"""Reference outputs of MiniMe's init stage on one window, for checking the Rust core.

Runs ``pipeline.init_units`` (``initA``, ``initC``, the initial ``unit_merge``) on the
factors ``ref_seeds.py`` built and on its merged seeds, exactly as ``pipeline.run`` does
after ``seeds_merge``. Writes, under ``processed/stream/ref_init_<t0>_<t1>/``:

* ``inputs/``: ``init.json`` (``H``, ``W``, ``T``, ``tile``, ``noise_freq`` and the init
  parameters), ``init_seeds.npy`` ((n, 2) int64 merged seeds, the native run's input) and a
  copy of ``ref_seeds.py``'s inputs (factors as ``fac_*.npy``, ``seeds.json``, and pass 1's
  and pass 2's inputs), so one browser run does pass 1, pass 2, the seeds and init;
* ``ref/``: ``initA_{indptr,indices,data}.npy`` (the footprints as CSR: int64, int64,
  float32), ``initA_window_corr.npy`` (float64 correlation of every window pixel to its
  seed, seeds in order, pixels row-major; the values ``initA`` thresholds),
  ``initA_centroids.npy``, ``C_initC.npy`` ((K, T) float32), ``unit_pairs.npy`` ((m, 2)
  units whose footprints overlap, ``i < j``) with their ``unit_corr.npy``
  (``FactorCorr.traces_corr``), ``merge_labels.npy`` (group of each unit, numbered in order
  of first appearance), ``merge_counts.npy``, ``A_merge_{indptr,indices,data}.npy``,
  ``C_merge.npy``, ``merge_centroids.npy`` and ``summary.json`` (counts, ``cond(G)`` and
  which solver ``_solve_normal`` took, timings).

    OMP_NUM_THREADS=1 python experiments/streaming/ref_init.py --t0 0 --t1 3000
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
import stream as st
from ref_seeds import load_factors

from minime import corr, init, pipeline
from minime.params import Params
from minime.report import StageTimer


def init_meta(params: Params, H: int, W: int, T: int) -> dict:
    return {
        "H": H,
        "W": W,
        "T": T,
        "tile": params.tile,
        "noise_freq": params.noise_freq,
        "inita_corr": params.inita_corr,
        "inita_wnd": params.inita_wnd,
        "init_merge_corr": params.init_merge_corr,
        "merge_min_dist": params.merge_min_dist,
    }


def save_csr(d: Path, name: str, A: sps.csr_matrix) -> None:
    A = sps.csr_matrix(A, dtype=np.float32, copy=True)
    A.sum_duplicates()
    A.sort_indices()
    np.save(d / f"{name}_indptr.npy", A.indptr.astype(np.int64))
    np.save(d / f"{name}_indices.npy", A.indices.astype(np.int64))
    np.save(d / f"{name}_data.npy", A.data)


def first_appearance(labels: np.ndarray) -> np.ndarray:
    """Relabel groups 0, 1, ... in order of first appearance (``_group_mean_matrix``'s rows)."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    return rank[inv]


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
    sd = s.out(f"ref_seeds_{a.t0}_{a.t1}")
    fac = load_factors(p1, p2, params)
    hw = np.load(sd / "ref" / "seeds_merged.npy")
    seeds = pd.DataFrame({"height": hw[:, 0], "width": hw[:, 1]})
    out = s.out(f"ref_init_{a.t0}_{a.t1}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)

    timer = StageTimer()
    with timer("corr_setup"):
        fc = corr.FactorCorr(fac, params.noise_freq)
    t = time.perf_counter()
    fp: dict[str, sps.csr_matrix] = {}
    A, C = pipeline.init_units(fac, fc, seeds, params, timer, fp)
    seconds = time.perf_counter() - t

    # the stage's intermediate values, recomputed for the check and its margins
    A0 = fp["initA"]
    C0 = init.initC(fac, A0)
    win = np.concatenate([fc.window(int(h) * fac.W_ + int(w), params.inita_wnd)[1] for h, w in hw])
    A64 = sps.csr_matrix(A0, dtype=np.float64)
    G = (A64 @ A64.T).toarray()
    G = 0.5 * (G + G.T)
    cond = float(np.linalg.cond(G))
    overlap = sps.triu((sps.csr_matrix(A0) @ sps.csr_matrix(A0).T) > 0, k=1).tocoo()
    pairs = np.stack([overlap.row, overlap.col], axis=1).astype(np.int64)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    R = fc.traces_corr(C0)
    pc = R[pairs[:, 0], pairs[:, 1]]
    adj = np.zeros(R.shape, bool)
    adj[pairs[:, 0], pairs[:, 1]] = pc > params.init_merge_corr
    adj |= adj.T
    if params.merge_min_dist > 0:
        raise SystemExit("merge_min_dist > 0: the labels below ignore the centroid rule")
    labels = first_appearance(corr.merge_labels(sps.csr_matrix(adj)))
    counts = np.bincount(labels)
    M = init._group_mean_matrix(labels)
    assert np.array_equal(np.asarray(M @ C0).astype(np.float32), C), "labels disagree"

    (inp / "init.json").write_text(json.dumps(init_meta(params, fac.H, fac.W_, fac.T), indent=1))
    np.save(inp / "init_seeds.npy", np.ascontiguousarray(hw, dtype=np.int64))
    for f in (sd / "inputs").iterdir():
        shutil.copy2(f, inp / f.name)

    save_csr(ref, "initA", A0)
    np.save(ref / "initA_window_corr.npy", win)
    np.save(ref / "initA_centroids.npy", init.centroids(A0, fac.H, fac.W_))
    np.save(ref / "C_initC.npy", C0)
    np.save(ref / "unit_pairs.npy", pairs)
    np.save(ref / "unit_corr.npy", pc)
    np.save(ref / "merge_labels.npy", labels)
    np.save(ref / "merge_counts.npy", counts.astype(np.int64))
    save_csr(ref, "A_merge", A)
    np.save(ref / "C_merge.npy", C)
    np.save(ref / "merge_centroids.npy", init.centroids(A, fac.H, fac.W_))
    summary = {
        "frames": fac.T,
        "seeds": len(hw),
        "units_initA": int(A0.shape[0]),
        "initA_nnz": int(A0.nnz),
        "units_init_merge": int(A.shape[0]),
        "overlap_pairs": len(pairs),
        "merged_pairs": int((pc > params.init_merge_corr).sum()),
        "cond_G": cond,
        "solver": "cholesky" if cond <= init.COND_MAX else "pinv",
        "seconds": seconds,
        "stages": {r["stage"]: r["seconds"] for r in timer.rows},
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
