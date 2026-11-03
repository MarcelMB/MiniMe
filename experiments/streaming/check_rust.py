"""Compare a Rust streaming pass (native or browser) with ``ref_dump.py``'s reference.

    python experiments/streaming/check_rust.py REF_DIR RUST_DIR

``REF_DIR`` is ``processed/stream/ref_<name>/``; ``RUST_DIR`` holds the Rust core's
``.npy`` outputs. Prints the largest relative error per array and exits 1 if any check
fails its tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps

# (name, relative tolerance against the array's largest magnitude)
DENSE = [
    ("sos", 1e-12),
    ("b", 1e-9),
    ("f", 1e-5),
    ("C", 1e-5),
    ("stats_n", 0.0),
    ("stats_csum", 1e-9),
    ("stats_ysum", 0.0),
    ("stats_dsq", 0.0),
    ("stats_M", 1e-9),
    ("stats_L", 1e-9),
    ("stats_Mlp", 1e-9),
    ("stats_lsum", 1e-9),
    ("support_pix", 0.0),
    ("support_unit", 0.0),
    ("mask", 0.0),
    ("norm_fac", 1e-6),
]
SPARSE = [("Asp", 1e-6), ("Amerged", 1e-6)]


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    scale = max(np.abs(a).max(initial=0.0), 1e-300)
    return float(np.abs(a - b).max(initial=0.0) / scale)


def load_csr(d: Path, name: str, shape: tuple[int, int] | None = None) -> sps.csr_matrix:
    data = np.load(d / f"{name}_data.npy")
    indices = np.load(d / f"{name}_indices.npy")
    indptr = np.load(d / f"{name}_indptr.npy")
    n = indptr.size - 1
    A = sps.csr_matrix((data, indices, indptr), shape=(n, shape[1] if shape else indices.max() + 1))
    A.sort_indices()
    return A


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    a = ap.parse_args()
    ref, rust = a.ref / "ref", a.rust
    meta = json.loads((a.ref / "inputs" / "params.json").read_text(encoding="utf-8"))
    P = meta["H"] * meta["W"]
    ok = True
    rows = []
    if (rust / "frames.npy").exists():
        fr, fw = np.load(ref / "frames.npy"), np.load(rust / "frames.npy")
        fr, fw = fr.reshape(fr.shape[0], -1), fw.reshape(fw.shape[0], -1)
        same = fr.shape == fw.shape and bool(np.array_equal(fr, fw))
        n_diff = int((fr != fw).sum()) if fr.shape == fw.shape else -1
        rows.append(("frames (uint8)", f"{n_diff} px differ", same))
        ok &= same
    for name, tol in DENSE:
        src = a.ref / "inputs" if name == "sos" else ref
        e = rel_err(np.load(src / f"{name}.npy"), np.load(rust / f"{name}.npy"))
        rows.append((name, f"{e:.2e} (tol {tol:.0e})", e <= tol))
        ok &= e <= tol
    for name, tol in SPARSE:
        A, B = load_csr(ref, name, (0, P)), load_csr(rust, name, (0, P))
        same_shape = A.shape == B.shape
        same_support = same_shape and (A != 0).astype(int).sum() == (B != 0).astype(int).sum()
        e = rel_err(A.toarray(), B.toarray()) if same_shape else float("inf")
        good = same_shape and e <= tol
        rows.append(
            (name, f"{A.shape[0]} vs {B.shape[0]} units, nnz {A.nnz} vs {B.nnz}, {e:.2e}", good)
        )
        ok &= good and bool(same_support)
    width = max(len(r[0]) for r in rows)
    for name, msg, good in rows:
        print(f"{'ok ' if good else 'BAD'} {name:<{width}}  {msg}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
