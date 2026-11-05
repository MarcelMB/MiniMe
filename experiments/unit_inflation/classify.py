"""Classify a run's units against the synthetic truth: real cells, duplicates, pieces, nothing.

Every unit is matched one-to-one to the truth as ``minime score`` does it (greedy by centroid
distance, <= 5 px, in template coordinates). Each unit also gets its best true cell by footprint
cosine (true footprints shifted by the run's offset). An unmatched unit is then

* ``duplicate``: its best true cell (cosine >= ``--cos``) is matched to another unit of the run;
* ``piece``: its best true cell (cosine >= ``--cos``) is matched to no unit (the footprint is off
  centre: a part of a cell, or a mixture of neighbours);
* ``out_of_focus``: centred within 1.5 sigma of an out-of-focus cell (real signal, not in truth);
* ``nothing``: none of these.

With ``--python`` the true cells a browser run found and the Python run did not are listed by
category (late, dim, overlap, ...). With ``--stitch`` (a browser run folder that holds the
stitch's ``group``, ``pairs``, ``pair_cos`` and the window units) every duplicate group is
explained: does it share a window with the group of the same cell (two units of one cell in one
window, which the stitch never links), and is there a link above the threshold between them
(the brightest-parent rule kept them apart) or not (footprints too different)?

    python classify.py --raw SESSION/raw --browser RUN [--python PYRUN] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from minime import score

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_csr(folder: Path, stem: str, P: int) -> sparse.csr_matrix:
    """``<stem>_{data,indices,indptr}.npy`` (the browser's stitch outputs) as (K, P) CSR."""
    d, i, p = (np.load(folder / f"{stem}_{k}.npy") for k in ("data", "indices", "indptr"))
    return sparse.csr_matrix((d, i, p), shape=(p.size - 1, P))


def load_run_A(run: Path, which: str = "A") -> sparse.csr_matrix:
    """A run folder's footprints: ``cnmf/<which>.npz`` (``A`` or ``Amerged``)."""
    return sparse.load_npz(run / "cnmf" / f"{which}.npz").tocsr()


def load_truth(raw: Path) -> tuple[dict, sparse.csr_matrix, int, int]:
    meta = json.loads((raw.parent / "meta.json").read_text())
    H, W = meta["height"], meta["width"]
    with np.load(raw.parent / "truth.npz") as z:
        t = {k: z[k] for k in z.files}
    At = sparse.csr_matrix((t["A_data"], t["A_indices"], t["A_indptr"]), shape=tuple(t["A_shape"]))
    return t, At, H, W


def offset(run: Path, truth: dict, A: sparse.csr_matrix, H: int, W: int) -> np.ndarray:
    """``score_run``'s template offset of the run (sign checked with the units)."""
    shifts = np.load(run / "motion" / "arrays.npz")["shifts"]
    c, _, sign = score.motion_fit(shifts.astype(float), truth["shifts"].astype(float))
    uc = score.footprint_centroids(A, H, W)
    best = min(
        ((score._median_dist(uc, truth["centroids"] + m * c)[0], m) for m in (sign, -sign)),
        key=lambda v: v[0],
    )
    return best[1] * c


def shift_footprints(At: sparse.csr_matrix, H: int, W: int, off: np.ndarray) -> sparse.csr_matrix:
    """True footprints moved by the rounded offset (frame -> template coordinates)."""
    dy, dx = (int(round(v)) for v in off)
    coo = At.tocoo()
    y, x = np.divmod(coo.col, W)
    y, x = y + dy, x + dx
    ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    return sparse.csr_matrix(
        (coo.data[ok], (coo.row[ok], y[ok] * W + x[ok])), shape=At.shape
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def unit_norm(A: sparse.csr_matrix) -> sparse.csr_matrix:
    A = sparse.csr_matrix(A, dtype=np.float64)
    n = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sparse.csr_matrix(sparse.diags(1.0 / n) @ A)


def classify(
    A: sparse.csr_matrix,
    truth: dict,
    At_s: sparse.csr_matrix,
    off: np.ndarray,
    H: int,
    W: int,
    cos_min: float,
) -> dict:
    """Per-unit labels and the matching of ``A`` against the truth."""
    uc = score.footprint_centroids(A, H, W)
    tc = truth["centroids"] + off
    _, ui, ti = score._median_dist(uc, tc)
    K = A.shape[0]
    match_t = np.full(K, -1)
    match_t[ui] = ti
    unit_of_true = np.full(len(tc), -1)
    unit_of_true[ti] = ui
    cos = (unit_norm(A) @ unit_norm(At_s).T).toarray()
    best_t = cos.argmax(axis=1)
    best_cos = cos[np.arange(K), best_t]
    oc = truth["oof_centroids"].reshape(-1, 2) + off
    d_oof = np.hypot(uc[:, None, 0] - oc[None, :, 0], uc[:, None, 1] - oc[None, :, 1])
    near_oof = (d_oof < 1.5 * truth["oof_sigma"][None]).any(axis=1)
    label = np.full(K, "nothing", dtype=object)
    for k in range(K):
        if match_t[k] >= 0:
            label[k] = "matched"
        elif best_cos[k] >= cos_min and unit_of_true[best_t[k]] >= 0:
            label[k] = "duplicate"
        elif best_cos[k] >= cos_min:
            label[k] = "piece"
        elif near_oof[k]:
            label[k] = "out_of_focus"
    return {
        "label": label,
        "match_t": match_t,
        "unit_of_true": unit_of_true,
        "best_t": best_t,
        "best_cos": best_cos,
        "centroids": uc,
    }


def counts(c: dict) -> dict:
    lab = c["label"]
    return {
        k: int((lab == k).sum())
        for k in ("matched", "duplicate", "piece", "out_of_focus", "nothing")
    }


def categories_of(truth: dict, cells: np.ndarray) -> dict:
    cats = sorted(k[4:] for k in truth if k.startswith("cat_"))
    return {c: int(truth[f"cat_{c}"][cells].sum()) for c in cats}


# ---------------------------------------------------------------------------
# Stitch diagnostics
# ---------------------------------------------------------------------------


def window_units(folder: Path, P: int) -> tuple[np.ndarray, np.ndarray]:
    """Window index and brightness of every stitch input unit (stitch order)."""
    win, bright = [], []
    i = 0
    while (folder / f"win_{i:02d}_bright.npy").exists():
        b = np.load(folder / f"win_{i:02d}_bright.npy")
        win.append(np.full(b.size, i))
        bright.append(b)
        i += 1
    return np.concatenate(win), np.concatenate(bright)


def explain_duplicates(folder: Path, c: dict, cos_thres: float) -> dict:
    """Why each duplicate group of the stitch was not joined with its cell's matched group."""
    group = np.load(folder / "group.npy")
    pairs = np.load(folder / "pairs.npy").reshape(-1, 2)
    pcos = np.load(folder / "pair_cos.npy")
    win, _ = window_units(folder, 0)
    members = [np.flatnonzero(group == g) for g in range(group.max() + 1)]
    gpair = group[pairs]
    out = {"shares_window": 0, "edge_above_thres": 0, "best_link_below_thres": 0, "no_link": 0}
    max_links = []
    for g in np.flatnonzero(c["label"] == "duplicate"):
        g0 = c["unit_of_true"][c["best_t"][g]]
        wg, w0 = set(win[members[g]]), set(win[members[g0]])
        if wg & w0:
            out["shares_window"] += 1
        sel = ((gpair[:, 0] == g) & (gpair[:, 1] == g0)) | (
            (gpair[:, 0] == g0) & (gpair[:, 1] == g)
        )
        m = float(pcos[sel].max()) if sel.any() else 0.0
        max_links.append(m)
        if not sel.any():
            out["no_link"] += 1
        elif m > cos_thres:
            out["edge_above_thres"] += 1
        else:
            out["best_link_below_thres"] += 1
    out["max_link_cos_quantiles"] = (
        np.quantile(max_links, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if max_links else []
    )
    nwin = np.load(folder / "n_windows.npy")
    for lab in ("matched", "duplicate", "piece", "nothing", "out_of_focus"):
        sel = c["label"] == lab
        out[f"n_windows_{lab}"] = np.bincount(nwin[sel], minlength=nwin.max() + 1).tolist()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--browser", type=Path, required=True, help="browser run folder (has cnmf/)")
    ap.add_argument("--which", default="A", help="cnmf/<which>.npz of the browser run")
    ap.add_argument("--python", type=Path, help="Python run folder")
    ap.add_argument("--stitch", type=Path, help="folder with the stitch outputs (group.npy, ...)")
    ap.add_argument("--cos", type=float, default=0.5, help="footprint cosine to call a true cell")
    ap.add_argument("--cos-thres", type=float, default=0.7, help="the stitch's threshold")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    truth, At, H, W = load_truth(a.raw)
    Ab = load_run_A(a.browser, a.which)
    off = offset(a.browser, truth, Ab, H, W)
    At_s = shift_footprints(At, H, W, off)
    det = truth["cat_detectable"]
    cb = classify(Ab, truth, At_s, off, H, W, a.cos)
    res = {
        "n_true": int(len(det)),
        "n_detectable": int(det.sum()),
        "offset_px": off.tolist(),
        "browser": {"units": int(Ab.shape[0]), **counts(cb)},
    }
    if a.python:
        Ap = load_run_A(a.python)
        cp = classify(Ap, truth, At_s, off, H, W, a.cos)
        res["python"] = {"units": int(Ap.shape[0]), **counts(cp)}
        fb, fp = cb["unit_of_true"] >= 0, cp["unit_of_true"] >= 0
        only_b, only_p = np.flatnonzero(fb & ~fp), np.flatnonzero(fp & ~fb)
        res["true_found_by_both"] = int((fb & fp).sum())
        res["true_only_browser"] = {"n": int(only_b.size), **categories_of(truth, only_b)}
        res["true_only_python"] = {"n": int(only_p.size), **categories_of(truth, only_p)}
        late = truth["cat_late"]
        res["late_recall"] = {
            "n": int(late.sum()),
            "browser": float(fb[late].mean()) if late.any() else None,
            "python": float(fp[late].mean()) if late.any() else None,
        }
    if a.stitch:
        res["stitch"] = explain_duplicates(a.stitch, cb, a.cos_thres)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt)


if __name__ == "__main__":
    main()
