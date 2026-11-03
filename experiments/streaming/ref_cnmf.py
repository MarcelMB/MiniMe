"""Reference outputs of MiniMe's CNMF loop on one window, for checking the Rust core.

Runs ``cnmf.run_cnmf`` (initial background, then ``n_iter`` x spatial update, background,
traces, loop merge) on the factors ``ref_seeds.py`` built and on ``ref_init.py``'s merged
units, exactly as ``pipeline.run_window2`` does after ``init_units``. A second run wraps
the loop's steps (``dilate_footprints``, ``solve_nnqp``, ``update_spatial``,
``update_background``, ``compute_trace``, ``unit_merge``) to record what each returns; it
must give the first run's result bit for bit. Writes, under
``processed/stream/ref_cnmf_<t0>_<t1>/`` (``ref_cnmf_clip_<t0>_<t1>/`` with ``--clip``):

* ``inputs/``: ``cnmf.json`` (``H``, ``W``, ``T``, ``tile``, ``noise_freq`` and the loop's
  parameters), ``cnmf_A_{indptr,indices,data}.npy`` and ``cnmf_C.npy`` (init's merged
  units, the native run's input) and a copy of ``ref_init.py``'s inputs, so one browser run
  does pass 1, pass 2, the seeds, init and the loop;
* ``ref/``, per iteration ``i``: ``sub_{i}_{active,indptr,units}.npy`` (the dilated
  supports by pixel: active pixels, their ``indptr``, allowed units), ``B_{i}.npy``
  (float64 of ``YCt``, one value per (pixel, unit) pair), ``lam_{i}.npy``, ``G_{i}.npy``
  (the unit-norm traces' Gram), ``w_{i}.npy`` (the NNQP solution per pair) and ``kkt_{i}.npy``
  (``(G w)_k - b_k + lam`` per pair, the margin of a zero), ``nnz_{i}.npy`` (support size
  per unit before the size filter), ``mask_{i}.npy``, ``norm_fac_{i}.npy``,
  ``spatial_{i}_{indptr,indices,data}.npy``, ``C_sp_{i}.npy``, ``b_{i}.npy``, ``f_{i}.npy``,
  ``YrA_{i}.npy`` (before the merge), ``unit_pairs_{i}.npy`` (overlapping units, ``i < j``)
  with ``unit_corr_{i}.npy`` (``traces_corr`` of ``YrA``), ``merge_labels_{i}.npy``,
  ``merge_counts_{i}.npy``, ``merge_{i}_{indptr,indices,data}.npy``; ``b_init.npy``,
  ``f_init.npy`` (the background before the loop); the result ``A_{indptr,indices,data}``,
  ``C``, ``YrA``, ``b``, ``f`` and ``summary.json`` (counts, timings).

    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python experiments/streaming/ref_cnmf.py --t0 0 --t1 3000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sps
import stream as st
from ref_init import first_appearance, save_csr
from ref_seeds import load_factors

from minime import cnmf, corr, init
from minime.params import Params


def cnmf_meta(params: Params, H: int, W: int, T: int) -> dict:
    return {
        "H": H,
        "W": W,
        "T": T,
        "tile": params.tile,
        "noise_freq": params.noise_freq,
        "dl_wnd": params.dl_wnd,
        "sparse_snr": params.sparse_snr,
        "size_thres": list(params.size_thres),
        "n_iter": params.n_iter,
        "background_clip": params.background_clip,
        "loop_merge_corr": params.loop_merge_corr,
        "merge_min_dist": params.merge_min_dist,
    }


def load_csr(d: Path, name: str, shape: tuple[int, int]) -> sps.csr_matrix:
    return sps.csr_matrix(
        (
            np.load(d / f"{name}_data.npy"),
            np.load(d / f"{name}_indices.npy"),
            np.load(d / f"{name}_indptr.npy"),
        ),
        shape=shape,
    )


def kkt(
    G: np.ndarray,
    B: np.ndarray,
    lam: np.ndarray,
    indptr: np.ndarray,
    units: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """``(G_s w)_k - b_k + lam`` for every (pixel, unit) pair, ``B`` per pair."""
    out = np.empty(w.size)
    for p in range(indptr.size - 1):
        s, e = indptr[p], indptr[p + 1]
        u = units[s:e]
        out[s:e] = G[np.ix_(u, u)] @ w[s:e] - B[s:e] + lam[p]
    return out


class Recorder:
    """Wraps the loop's steps to record their outputs; the wrapped calls are unchanged."""

    def __init__(self) -> None:
        self.it = 0
        self.out: dict[str, np.ndarray] = {}
        self.seconds: dict[str, float] = {}
        self.bg_calls = 0
        self.real = {
            "dilate": cnmf.dilate_footprints,
            "nnqp": cnmf.solve_nnqp,
            "spatial": cnmf.update_spatial,
            "background": cnmf.update_background,
            "trace": cnmf.compute_trace,
            "merge": init.unit_merge,
        }

    def _time(self, name: str, fn, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        t = time.perf_counter()
        r = fn(*a, **k)
        key = f"{name}_{self.it}"
        self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - t
        return r

    def dilate(self, A, H, W, dl_wnd):  # noqa: ANN001, ANN201
        sub = self._time("dilate", self.real["dilate"], A, H, W, dl_wnd)
        subT = sps.csr_matrix(sub.T)
        counts = np.diff(subT.indptr)
        active = np.flatnonzero(counts)
        self.out[f"sub_{self.it}_active"] = active.astype(np.int64)
        return sub

    def nnqp(self, G, B, lam, indptr, indices):  # noqa: ANN001, ANN201
        values = self._time("nnqp", self.real["nnqp"], G, B, lam, indptr, indices)
        rows = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
        b = np.asarray(B)[rows, indices]
        i = self.it
        self.out |= {
            f"sub_{i}_indptr": np.asarray(indptr, dtype=np.int64),
            f"sub_{i}_units": np.asarray(indices, dtype=np.int64),
            f"B_{i}": b,
            f"lam_{i}": np.asarray(lam, dtype=np.float64),
            f"G_{i}": np.asarray(G, dtype=np.float64),
            f"w_{i}": values.copy(),
            f"kkt_{i}": kkt(G, b, lam, indptr, indices, values),
        }
        return values

    def spatial(self, f, A, C, sn, dl_wnd, sparse_snr, size_thres):  # noqa: ANN001, ANN201
        A_new, mask, nf = self._time(
            "spatial", self.real["spatial"], f, A, C, sn, dl_wnd, sparse_snr, size_thres
        )
        i = self.it
        w, units = self.out[f"w_{i}"], self.out[f"sub_{i}_units"]
        self.out[f"nnz_{i}"] = np.bincount(units[w != 0], minlength=A.shape[0]).astype(np.int64)
        self.out[f"mask_{i}"] = mask
        self.out[f"norm_fac_{i}"] = nf
        self.out[f"C_sp_{i}"] = (np.asarray(C)[mask] * nf[:, None]).astype(np.float32)
        return A_new, mask, nf

    def background(self, f, A, C, b=None, **kw):  # noqa: ANN001, ANN003, ANN201
        name = "background_init" if self.bg_calls == 0 else "background"
        r = self._time(name, self.real["background"], f, A, C, b, **kw)
        tag = "init" if self.bg_calls == 0 else str(self.it)
        self.out[f"b_{tag}"], self.out[f"f_{tag}"] = r
        self.bg_calls += 1
        return r

    def trace(self, f, A, b, C, fb):  # noqa: ANN001, ANN201
        YrA = self._time("trace", self.real["trace"], f, A, b, C, fb)
        self.out[f"YrA_{self.it}"] = YrA
        return YrA

    def merge(self, A, C, fc_traces, thres_corr, extra=None, min_dist=0.0, shape=None):  # noqa: ANN001, ANN201
        t = time.perf_counter()
        r = self.real["merge"](A, C, fc_traces, thres_corr, extra, min_dist=min_dist, shape=shape)
        self.seconds[f"merge_{self.it}"] = time.perf_counter() - t
        if min_dist > 0:
            raise SystemExit("merge_min_dist > 0: the labels below ignore the centroid rule")
        A = sps.csr_matrix(A)
        ov = sps.triu((A @ A.T) > 0, k=1).tocoo()
        pairs = np.stack([ov.row, ov.col], axis=1).astype(np.int64)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        R = fc_traces(C)
        pc = R[pairs[:, 0], pairs[:, 1]]
        adj = np.zeros(R.shape, bool)
        adj[pairs[:, 0], pairs[:, 1]] = pc > thres_corr
        adj |= adj.T
        labels = first_appearance(corr.merge_labels(sps.csr_matrix(adj)))
        assert labels.max() + 1 == r[0].shape[0], "labels disagree"
        i = self.it
        self.out |= {
            f"unit_pairs_{i}": pairs,
            f"unit_corr_{i}": pc,
            f"merge_labels_{i}": labels,
            f"merge_counts_{i}": np.bincount(labels).astype(np.int64),
        }
        self.it += 1
        return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=3000)
    ap.add_argument("--clip", action="store_true", help="background_clip=True (Minian's)")
    a = ap.parse_args()

    s = st.Session(a.session)
    params = s.params()
    if a.clip:
        params = dataclasses.replace(params, background_clip=True)
    p1 = s.out(f"ref_pass1_{a.t0}_{a.t1}")
    p2 = s.out(f"ref_pass2_{a.t0}_{a.t1}")
    ri = s.out(f"ref_init_{a.t0}_{a.t1}")
    fac = load_factors(p1, p2, params)
    C0 = np.load(ri / "ref" / "C_merge.npy")
    A0 = load_csr(ri / "ref", "A_merge", (C0.shape[0], fac.P))
    out = s.out(f"ref_cnmf{'_clip' if a.clip else ''}_{a.t0}_{a.t1}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)

    fc = corr.FactorCorr(fac, params.noise_freq)
    fp: dict[str, sps.csr_matrix] = {}
    res = cnmf.run_cnmf(fac, A0, C0, params, fc.traces_corr, record=fp.__setitem__)
    t = time.perf_counter()  # timed once numba has compiled the NNQP kernel
    cnmf.run_cnmf(fac, A0, C0, params, fc.traces_corr)
    seconds = time.perf_counter() - t

    rec = Recorder()
    with (
        mock.patch.object(cnmf, "dilate_footprints", rec.dilate),
        mock.patch.object(cnmf, "solve_nnqp", rec.nnqp),
        mock.patch.object(cnmf, "update_spatial", rec.spatial),
        mock.patch.object(cnmf, "update_background", rec.background),
        mock.patch.object(cnmf, "compute_trace", rec.trace),
        mock.patch.object(init, "unit_merge", rec.merge),
    ):
        res2 = cnmf.run_cnmf(fac, A0, C0, params, fc.traces_corr)
    for name in ("C", "YrA", "b", "f"):
        assert np.array_equal(getattr(res, name), getattr(res2, name)), f"{name} differs"
    assert (res.A != res2.A).nnz == 0, "A differs"

    (inp / "cnmf.json").write_text(json.dumps(cnmf_meta(params, fac.H, fac.W_, fac.T), indent=1))
    save_csr(inp, "cnmf_A", A0)
    np.save(inp / "cnmf_C.npy", C0)
    for f in (ri / "inputs").iterdir():
        shutil.copy2(f, inp / f.name)

    for name, arr in rec.out.items():
        np.save(ref / f"{name}.npy", arr)
    for name, A in fp.items():
        save_csr(ref, name, A)
    save_csr(ref, "A", res.A)
    for name in ("C", "YrA", "b", "f"):
        np.save(ref / f"{name}.npy", getattr(res, name))

    w0 = s.out("t1_fresh") / "w3000" / f"win_{a.t0:06d}_{a.t1:06d}.A.npz"
    stored = int(sps.load_npz(w0).shape[0]) if w0.exists() and not a.clip else None
    summary = {
        "frames": fac.T,
        "units_in": int(A0.shape[0]),
        "units_final": int(res.A.shape[0]),
        "units_stored_window_run": stored,
        "counts": res.counts,
        "support_pairs": {i: int(rec.out[f"w_{i}"].size) for i in range(params.n_iter)},
        "overlap_pairs": {i: len(rec.out[f"unit_pairs_{i}"]) for i in range(params.n_iter)},
        "background_clip": params.background_clip,
        "seconds": seconds,
        "stages": rec.seconds,
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
