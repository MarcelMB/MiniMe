"""Reference outputs of the stitch of the ``fresh`` window units, for checking the Rust core.

Reads the window units ``t1_discovery.py --mode fresh`` stored
(``processed/stream/t1_fresh/w<length>/win_*``) and redoes ``stream.stitch`` step by step,
recording what each step decides; the result must equal ``stream.stitch``'s bit for bit,
and with all windows also the stored ``stitched_A.npz`` / ``stitched.npz``. Writes, under
``processed/stream/ref_stitch_w<length>/`` (``..._n<n>/`` with ``--n``, the first ``n``
windows only):

* ``inputs/``: ``stitch.json`` (``cos_thres``, ``H``, ``W``, the windows' ``[t0, t1)``),
  per window ``i`` ``win_{i:02d}_A_{indptr,indices,data}.npy`` (float32, in the stored row
  order) and ``win_{i:02d}_bright.npy`` (float64, the peak of each unit's ``YrA``);
* ``ref/``: ``norms.npy`` (row norms of the stacked footprints scaled to maximum 1),
  ``pairs.npy`` / ``pair_cos.npy`` (every pair of units of different windows with a
  non-zero cosine, ``i < j``), ``edges.npy`` (the pairs above ``cos_thres``), ``parent.npy``
  (``corr.brightest_parent``), ``root.npy``, ``group.npy`` (each unit's stitched unit),
  ``A_{indptr,indices,data}.npy`` (stitched, float32, rows with maximum 1), ``bright.npy``,
  ``n_windows.npy`` and ``summary.json``;
* with ``--session-inputs``: ``session/``, the browser's window-loop inputs
  (``session.json``: the window length, every stage's parameters without ``T`` / ``t0`` /
  ``t1`` and the stitch's ``cos_thres``; ``shifts.npy``, the whole session's baseline
  shifts (T, 2) float32; ``template.npy``), plus
  ``omega_<t0>_<k>.npy`` for every window whose ``ref_pass2_<t0>_<t1>`` exists, so the
  browser replays numpy's draws there.

    python experiments/streaming/ref_stitch.py                         # all 12 windows
    python experiments/streaming/ref_stitch.py --n 2 --session-inputs  # 0-6000, browser
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import stream as st
from ref_cnmf import cnmf_meta
from ref_init import init_meta, save_csr
from ref_pass1 import pass1_meta
from ref_pass2 import pass2_meta
from ref_seeds import seeds_meta

from minime import corr


def stitch_steps(units: list[st.WindowUnits], cos_thres: float) -> dict[str, np.ndarray]:
    """``stream.stitch`` with every intermediate kept (same operations, same order)."""
    A = st._row_max_one(sps.vstack([u.A for u in units]).tocsr().astype(np.float64))
    win = np.concatenate([np.full(u.A.shape[0], i) for i, u in enumerate(units)])
    bright = np.concatenate([u.bright for u in units]).astype(np.float64)
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    An = sps.csr_matrix(sps.diags(1.0 / norms) @ A)
    S = sps.triu(An @ An.T, k=1).tocoo()
    cross = win[S.row] != win[S.col]
    o = np.lexsort((S.col[cross], S.row[cross]))
    pairs = np.stack([S.row[cross], S.col[cross]], axis=1)[o].astype(np.int64)
    pair_cos = S.data[cross][o]
    keep = (S.data > cos_thres) & (win[S.row] != win[S.col])
    edges = np.stack([S.row[keep], S.col[keep]], axis=1).astype(np.int64)
    parent = corr.brightest_parent(edges, bright)
    root = st._roots(parent)
    _, g = np.unique(root, return_inverse=True)
    n_g, N = int(g.max()) + 1, g.size
    M = sps.csr_matrix((1.0 / np.bincount(g)[g], (g, np.arange(N))), shape=(n_g, N))
    A_st = st._row_max_one(sps.csr_matrix(M @ A)).astype(np.float32)
    bright_g = np.zeros(n_g)
    np.maximum.at(bright_g, g, bright)
    n_windows = np.bincount(np.unique(np.stack([g, win]), axis=1)[0], minlength=n_g)
    return {
        "norms": norms,
        "pairs": pairs,
        "pair_cos": pair_cos,
        "edges": edges[np.lexsort((edges[:, 1], edges[:, 0]))] if len(edges) else edges,
        "parent": parent.astype(np.int64),
        "root": root.astype(np.int64),
        "group": g.astype(np.int64),
        "A": A_st,
        "bright": bright_g,
        "n_windows": n_windows.astype(np.int64),
    }


def same_csr(a: sps.csr_matrix, b: sps.csr_matrix) -> bool:
    a, b = sps.csr_matrix(a, copy=True), sps.csr_matrix(b, copy=True)
    a.sort_indices()
    b.sort_indices()
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


def write_session(s: st.Session, out: Path, length: int, cos_thres: float) -> None:
    """The browser window loop's inputs: one parameter file for every window."""
    params = s.params()
    H, W, T = s.shape()
    sc = s.sidecar()
    d = out / "session"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    drop = ("T", "t0", "t1")
    session = {
        "window_frames": length,
        "pass1": {k: v for k, v in pass1_meta(params, H, W, 0, 0).items() if k not in drop},
        "pass2": {k: v for k, v in pass2_meta(params, H, W, 0).items() if k not in drop},
        "seeds": {k: v for k, v in seeds_meta(params, H, W, 0).items() if k not in drop},
        "init": {k: v for k, v in init_meta(params, H, W, 0).items() if k not in drop},
        "cnmf": {k: v for k, v in cnmf_meta(params, H, W, 0).items() if k not in drop},
        "stitch": {"cos_thres": cos_thres},
    }
    (d / "session.json").write_text(json.dumps(session, indent=1))
    np.save(d / "shifts.npy", np.ascontiguousarray(sc.shifts, dtype=np.float32))
    np.save(d / "template.npy", sc.template.astype(np.float32))
    for p2 in sorted((s.root / "processed" / "stream").glob("ref_pass2_*_*")):
        t0 = int(p2.name.split("_")[2])
        for f in sorted((p2 / "inputs").glob("omega_*.npy")):
            shutil.copy2(f, d / f"omega_{t0:06d}_{f.name[6:]}")  # file inputs skip symlinks
        print(f"  replaying numpy's draws in window {t0}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", dest="root", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--length", type=int, default=3000, help="window length (frames)")
    ap.add_argument("--n", type=int, default=None, help="first n windows only")
    ap.add_argument("--session-inputs", dest="session", action="store_true")
    a = ap.parse_args()

    s = st.Session(a.root)
    src = s.out("t1_fresh") / f"w{a.length}"
    cos_thres = float(json.loads((s.out("t1_fresh") / "summary.json").read_text())["cos"])
    paths = sorted(src.glob("win_*_*.npz"))
    paths = [p for p in paths if not p.name.endswith(".A.npz")][: a.n]
    units = [st.WindowUnits.load(p) for p in paths]
    H, W, _ = s.shape()
    out = s.out(f"ref_stitch_w{a.length}" + (f"_n{a.n}" if a.n else ""))
    inp, ref = out / "inputs", out / "ref"
    for d in (inp, ref):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()

    r = stitch_steps(units, cos_thres)
    A_st, bright, n_win = st.stitch(units, cos_thres)
    assert same_csr(r["A"], A_st), "A differs from stream.stitch"
    assert np.array_equal(r["bright"], bright) and np.array_equal(r["n_windows"], n_win)
    stored = None
    if a.n is None:
        z = np.load(src / "stitched.npz")
        stored = same_csr(r["A"], sps.load_npz(src / "stitched_A.npz")) and all(
            np.array_equal(r[k], z[k]) for k in ("bright", "n_windows")
        )
        assert stored, "differs from the stored stitched_A.npz / stitched.npz"

    meta = {"cos_thres": cos_thres, "H": H, "W": W, "windows": [[u.t0, u.t1] for u in units]}
    (inp / "stitch.json").write_text(json.dumps(meta, indent=1))
    for i, u in enumerate(units):
        A = sps.csr_matrix(u.A, dtype=np.float32)  # stored row order (the stitch sorts)
        np.save(inp / f"win_{i:02d}_A_indptr.npy", A.indptr.astype(np.int64))
        np.save(inp / f"win_{i:02d}_A_indices.npy", A.indices.astype(np.int64))
        np.save(inp / f"win_{i:02d}_A_data.npy", A.data)
        np.save(inp / f"win_{i:02d}_bright.npy", u.bright.astype(np.float64))
    save_csr(ref, "A", r.pop("A"))
    for k, v in r.items():
        np.save(ref / f"{k}.npy", v)
    summary = {
        "windows": len(units),
        "units_in": int(r["group"].size),
        "pairs": int(len(r["pairs"])),
        "edges": int(len(r["edges"])),
        "units_out": int(bright.size),
        "found_in_n_windows": np.bincount(n_win).tolist(),
        "equals_stored": stored,
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if a.session:
        write_session(s, out, a.length, cos_thres)
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
