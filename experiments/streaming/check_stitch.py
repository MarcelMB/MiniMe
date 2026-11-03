"""Compare the Rust stitch (native or browser) with ``ref_stitch.py``'s reference.

    python experiments/streaming/check_stitch.py REF_DIR RUST_DIR [--own-windows]

``REF_DIR`` is ``processed/stream/ref_stitch_w<length>[_n<n>]/``; ``RUST_DIR`` holds the Rust
core's outputs. The decisions must be identical: the pairs of units with a non-zero cosine,
the edges above ``cos_thres``, the parents (``brightest_parent``), roots, groups and the
number of windows of every stitched unit, and the stitched footprints' supports.

On the same window units (``minime-core stitch``, or the browser's ``StitchEngine`` on
``REF_DIR/inputs``) the port follows scipy's and numpy's summation orders, so every value is
expected bit for bit: tolerance 0 on the norms, cosines, footprint values and brightness.

``--own-windows`` (the browser's window loop: ``RUST_DIR`` also holds its own window units
as ``stitch.json`` + ``win_*``) first compares each window with the reference's: same unit
count and supports, footprint values and brightness within the CNMF check's own-factors
tolerance (1e-3 of the largest value, ``check_cnmf.py --own-factors``); the stitch values
then inherit that tolerance, and the cosines move by about as much.

Next to the observed differences the closest decisions are printed: the cross-window pair
whose cosine lies nearest ``cos_thres`` (with the cosine difference at it) and the linked
pair nearest a brightness tie in ``brightest_parent`` (relative gap, with the brightness
difference of its units). Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOL_OWN = 1e-3


def load_csr(d: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.load(d / f"{name}_{k}.npy") for k in ("indptr", "indices", "data"))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Largest difference relative to the largest reference value."""
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b)) / max(np.max(np.abs(b)), 1e-300))


class Checks:
    def __init__(self) -> None:
        self.ok = True

    def same(self, name: str, a: np.ndarray, b: np.ndarray) -> bool:
        good = a.shape == b.shape and np.array_equal(a, b)
        detail = "" if good else f"  (shapes {a.shape} vs {b.shape}"
        if not good and a.shape == b.shape:
            detail += f", {int(np.sum(a != b))} differ"
        print(f"{'ok ' if good else 'BAD'} {name}: identical{'' if good else detail + ')'}")
        self.ok &= good
        return good

    def close(self, name: str, a: np.ndarray, b: np.ndarray, tol: float) -> float:
        if a.shape != b.shape:
            print(f"BAD {name}: shapes {a.shape} vs {b.shape}")
            self.ok = False
            return np.inf
        d = rel(a, b)
        good = d <= tol
        print(f"{'ok ' if good else 'BAD'} {name}: rel diff {d:.2e} (tol {tol:.0e})")
        self.ok &= good
        return d


def check_windows(c: Checks, ref_inp: Path, rust: Path) -> None:
    meta_r = json.loads((ref_inp / "stitch.json").read_text())
    meta_b = json.loads((rust / "stitch.json").read_text())
    c.same("windows [t0, t1)", np.asarray(meta_b["windows"]), np.asarray(meta_r["windows"]))
    for i, (t0, t1) in enumerate(meta_r["windows"]):
        pr, ir, dr = load_csr(ref_inp, f"win_{i:02d}_A")
        pb, ib, db = load_csr(rust, f"win_{i:02d}_A")
        print(f"window {i} ({t0}-{t1}): {len(pr) - 1} units (reference) vs {len(pb) - 1}")
        if not c.same(f"  window {i} unit count", np.asarray(len(pb)), np.asarray(len(pr))):
            continue

        # the reference rows may be stored in another order: compare sorted
        def sort_rows(p: np.ndarray, ix: np.ndarray, d: np.ndarray) -> tuple:
            o = np.concatenate([p[k] + np.argsort(ix[p[k] : p[k + 1]]) for k in range(len(p) - 1)])
            return ix[o.astype(np.int64)], d[o.astype(np.int64)]

        isr, dsr = sort_rows(pr, ir, dr)
        isb, dsb = sort_rows(pb, ib, db)
        if c.same(f"  window {i} supports", np.concatenate([pb, isb]), np.concatenate([pr, isr])):
            c.close(f"  window {i} footprint values", dsb, dsr, TOL_OWN)
        c.close(
            f"  window {i} brightness (YrA peaks)",
            np.load(rust / f"win_{i:02d}_bright.npy"),
            np.load(ref_inp / f"win_{i:02d}_bright.npy"),
            TOL_OWN,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("rust", type=Path)
    ap.add_argument("--own-windows", action="store_true")
    a = ap.parse_args()
    ref, rust = a.ref / "ref", a.rust
    meta = json.loads((a.ref / "inputs" / "stitch.json").read_text())
    cos_thres = meta["cos_thres"]
    tol = TOL_OWN if a.own_windows else 0.0
    c = Checks()
    if a.own_windows:
        check_windows(c, a.ref / "inputs", rust)

    c.close("row norms", np.load(rust / "norms.npy"), np.load(ref / "norms.npy"), tol)
    pairs_same = c.same(
        "cross-window pairs (non-zero cosine)",
        np.load(rust / "pairs.npy"),
        np.load(ref / "pairs.npy"),
    )
    cos_r = np.load(ref / "pair_cos.npy")
    dcos = np.zeros_like(cos_r)
    if pairs_same:
        cos_b = np.load(rust / "pair_cos.npy")
        c.close("pair cosines", cos_b, cos_r, tol)
        dcos = np.abs(cos_b - cos_r)
    for name in ("edges", "parent", "root", "group", "n_windows"):
        c.same(name, np.load(rust / f"{name}.npy"), np.load(ref / f"{name}.npy"))
    pr, ir, dr = load_csr(ref, "A")
    pb, ib, db = load_csr(rust, "A")
    if c.same("stitched supports", np.concatenate([pb, ib]), np.concatenate([pr, ir])):
        c.close("stitched footprint values", db, dr, tol)
    c.close("stitched brightness", np.load(rust / "bright.npy"), np.load(ref / "bright.npy"), tol)

    # margins: the cosine nearest cos_thres, the linked pair nearest a brightness tie
    if cos_r.size:
        k = int(np.argmin(np.abs(cos_r - cos_thres)))
        near = int(np.sum(np.abs(cos_r - cos_thres) <= dcos.max())) if dcos.max() > 0 else 0
        print(
            f"closest cosine: pair {np.load(ref / 'pairs.npy')[k].tolist()} at {cos_r[k]:.6f}, "
            f"{abs(cos_r[k] - cos_thres):.2e} from {cos_thres} (diff there {dcos[k]:.1e}; "
            f"largest cosine diff {dcos.max():.1e}; pairs within it of the threshold: {near})"
        )
    edges = np.load(ref / "edges.npy")
    if len(edges):
        units = [
            np.load(a.ref / "inputs" / f"win_{i:02d}_bright.npy")
            for i in range(len(meta["windows"]))
        ]
        br = np.concatenate(units)
        gap = np.abs(br[edges[:, 0]] - br[edges[:, 1]]) / np.maximum(
            br[edges[:, 0]], br[edges[:, 1]]
        )
        k = int(np.argmin(gap))
        msg = (
            f"closest brightness tie among {len(edges)} edges: units {edges[k].tolist()} "
            f"({br[edges[k, 0]]:.6g} vs {br[edges[k, 1]]:.6g}), relative gap {gap[k]:.2e}"
        )
        if a.own_windows and (rust / "win_00_bright.npy").exists():
            bb = np.concatenate(
                [np.load(rust / f"win_{i:02d}_bright.npy") for i in range(len(meta["windows"]))]
            )
            if bb.shape == br.shape:
                db_rel = np.abs(bb - br) / np.abs(br).max()
                there = np.abs(bb - br)[edges[k]].max() / br[edges[k]].max()
                msg += f" (brightness diff there {there:.1e} rel; largest {db_rel.max():.1e})"
        print(msg)
    print("PASS" if c.ok else "FAIL")
    sys.exit(0 if c.ok else 1)


if __name__ == "__main__":
    main()
