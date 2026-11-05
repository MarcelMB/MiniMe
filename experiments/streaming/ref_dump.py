"""Reference outputs of one streaming pass + footprint update, for checking the Rust core.

Runs the prototype (``stream.FrameStream``, ``stream_pass``, ``spatial_from_stats``,
``merge_from_stats``) on the first ``--frames`` frames and writes, under
``processed/stream/ref_<footprints>_<frames>/``:

* ``inputs/``: everything the Rust core reads besides the raw AVIs, as plain ``.npy``
  (footprints as CSR triples ``A_data``/``A_indices``/``A_indptr``, ``shifts``, ``ymean``,
  ``sn``) plus ``params.json`` and scipy's low-pass ``sos`` (to check Rust's own design).
* ``ref/``: preprocessed check frames (``frames_t.npy``, ``frames.npy`` uint8), traces
  ``C`` (K, T) float32 and ``f``, the summed statistics (``stats_*.npy``, float64), and the
  update: ``Asp_*`` (CSR), ``mask``, ``norm_fac``, ``Amerged_*`` (CSR).

    python experiments/streaming/ref_dump.py --footprints stitched --frames 2000 \\
        --stitched "/Volumes/Extreme SSD/synth_v4_grin30/processed/stream/t1_fresh/w3000"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import stream as st
from scipy.signal import butter
from t2_stream import start_footprints

from minime import io

CHECK_FRAMES = (0, 1, 99, 100, 999, 1000, 1999, 5000, 17999, 35999)


class FirstFrames(st.FrameStream):
    """``FrameStream`` stopped after ``limit`` frames (a multiple of ``batch``)."""

    def __init__(self, *args: object, limit: int, **kw: object) -> None:
        super().__init__(*args, **kw)  # type: ignore[arg-type]
        if limit % self.batch:
            raise ValueError(f"--frames {limit} is not a multiple of --batch {self.batch}")
        self.limit = limit

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for t0, Y in super().__iter__():
            if t0 >= self.limit:
                return
            yield t0, Y


def save_csr(d: Path, name: str, A: sps.csr_matrix) -> None:
    A = sps.csr_matrix(A)
    np.save(d / f"{name}_data.npy", A.data.astype(np.float32))
    np.save(d / f"{name}_indices.npy", A.indices.astype(np.int32))
    np.save(d / f"{name}_indptr.npy", A.indptr.astype(np.int64))


def check_frames(frames: st.FrameStream, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Preprocessed uint8 frames at ``CHECK_FRAMES`` below ``T`` (one more read)."""
    want = [t for t in CHECK_FRAMES if t < T]
    out = []
    for t0, Y in frames:
        out += [Y[t - t0] for t in want if t0 <= t < t0 + Y.shape[0]]
        if t0 + Y.shape[0] > want[-1]:
            break
    return np.asarray(want, np.int64), np.stack(out).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument(
        "--footprints", choices=("baseline_final", "baseline_init", "stitched"), required=True
    )
    ap.add_argument("--stitched", type=Path, help="t1 output dir with stitched_A.npz")
    ap.add_argument("--frames", type=int, help="first frames only (default: all)")
    ap.add_argument("--segment-frames", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--baseline", default="baseline", help="the batch run to start from")
    ap.add_argument("--name", default="ref", help="output folder prefix")
    a = ap.parse_args()

    s = st.Session(a.session, baseline=a.baseline)
    params = s.params()
    H, W, T_all = s.shape()
    T = T_all if a.frames is None else a.frames
    sc = s.sidecar()
    fac = s.factors()
    ymean, sn = fac.mean_flat(), fac.sigma_flat()
    del fac
    out = s.out(f"{a.name}_{a.footprints}_{T}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)

    A = sps.csr_matrix(start_footprints(s, a.footprints, a.stitched), dtype=np.float32)
    save_csr(inp, "A", A)
    np.save(inp / "shifts.npy", sc.shifts[:T].astype(np.float32))
    np.save(inp / "ymean.npy", ymean.astype(np.float32))
    np.save(inp / "sn.npy", sn.astype(np.float32))
    np.save(inp / "sos.npy", butter(4, 2 * params.noise_freq, output="sos"))
    meta = {
        "H": H,
        "W": W,
        "T": T,
        "K": int(A.shape[0]),
        "segment_frames": a.segment_frames,
        "batch": a.batch,
        "median_ksize": params.median_ksize,
        "tophat_wnd": params.tophat_wnd,
        "motion_before_tophat": params.motion_before_tophat,
        "motion_fill": params.motion_fill,
        "dl_wnd": params.dl_wnd,
        "sparse_snr": params.sparse_snr,
        "size_thres": list(params.size_thres),
        "noise_freq": params.noise_freq,
        "loop_merge_corr": params.loop_merge_corr,
    }
    (inp / "params.json").write_text(json.dumps(meta, indent=1))
    if sc.fill is not None:
        np.save(inp / "fill.npy", sc.fill)

    frames = FirstFrames(io.probe(s.raw), sc.shifts, params, a.batch, limit=T, fill=sc.fill)
    ft, fr = check_frames(frames, T)
    np.save(ref / "frames_t.npy", ft)
    np.save(ref / "frames.npy", fr)

    A64 = sps.csr_matrix(A, dtype=np.float64)
    model = st.TraceModel.build(A64, ymean)
    sup = st.Support.build(A64, H, W, params.dl_wnd)
    res = st.stream_pass(frames, model, sup, T, a.segment_frames, params.noise_freq)
    np.save(ref / "C.npy", res.C)
    np.save(ref / "f.npy", res.f)
    np.save(ref / "b.npy", model.b)
    np.save(ref / "support_pix.npy", sup.pix)
    np.save(ref / "support_unit.npy", sup.indices)
    tot = st.total(res.stats)
    for k in st.SegmentStats.SUMS:
        np.save(ref / f"stats_{k}.npy", getattr(tot, k))
    np.save(ref / "stats_n.npy", np.asarray([tot.n]))

    A_sp, mask, norm_fac = st.spatial_from_stats(
        A64, sup, tot, sn, params.sparse_snr, params.size_thres
    )
    save_csr(ref, "Asp", A_sp)
    np.save(ref / "mask.npy", mask)
    np.save(ref / "norm_fac.npy", norm_fac)
    A_m = st.merge_from_stats(A_sp, tot, mask, params.loop_merge_corr, lowpassed=True)
    save_csr(ref, "Amerged", A_m)
    summary = {
        "T": T,
        "K": int(A.shape[0]),
        "kept": int(mask.sum()),
        "merged": int(A_m.shape[0]),
        "seconds": res.seconds,
        "timing": res.timing,
        "state_mb": res.state_bytes / 2**20,
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
