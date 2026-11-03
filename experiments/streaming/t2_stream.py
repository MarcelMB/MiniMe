"""Tests 2-4: traces and spatial updates streamed over the whole session.

Starting footprints (``--footprints``):

* ``baseline_final``: the batch run's final footprints. Pass 0 alone answers test 3: are
  per-frame least-squares traces with fixed footprints as good as the batch loop's traces?
* ``baseline_init``: the batch run's footprints before the CNMF loop. With
  ``--n-updates 1`` and ``2`` this answers test 2: does one spatial update from streamed
  sums match the batch loop's ``n_iter`` iterations?
* ``stitched``: the output of ``t1_discovery.py`` (``--stitched DIR``): the whole
  bounded-memory pipeline.

Every pass reads the movie once (``FrameStream``), solves each frame's traces with fixed
footprints, keeps per-segment sums, and is scored; between passes the footprints are
updated from the sums (``spatial_from_stats``) and merged (``merge_from_stats``). Test 4:
after the first update the footprints are also fitted from each segment's sums alone
(``segment_stability``), to see how far they move over the session.

    python experiments/streaming/t2_stream.py --footprints baseline_final
    python experiments/streaming/t2_stream.py --footprints baseline_init --n-updates 2
    python experiments/streaming/t2_stream.py --footprints stitched \\
        --stitched "/Volumes/Extreme SSD/synth_v4_grin30/processed/stream/t1_fresh/w3000"

``--background mean`` fits ``[A; ymean]`` with the plain mean image as ``b`` instead of
carving the units' LS share out of it (``stream.TraceModel.build``).

``--project`` runs every pass on projected frames (``stream.TileBasis``, test 5, no gain:
see the README): ``windows`` (each frame onto its own ``fresh`` window's pass-2 basis) or
``merged`` (the windows' bases folded into one per tile, ``--cap`` columns).

Shortcuts (marked in ``results.json``): the first pass's background uses the baseline
factors' mean image, and the spatial update uses the factors' FFT noise unless
``--sn diff``. ``--mean-noise windows`` takes both from the stitched windows' pass-2
factors instead (``stream.window_mean_noise``), as a browser-only run does.

Writes ``processed/stream/t2_<footprints>_u<n>/`` (``pass<i>_A.npz``, ``pass<i>_C.npy``,
``stability.npz``, ``results.json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import stream as st

from minime import io


def start_footprints(s: st.Session, which: str, stitched: Path | None) -> sps.csr_matrix:
    if which == "stitched":
        if stitched is None:
            raise SystemExit("--footprints stitched needs --stitched DIR (a t1 output)")
        return sps.csr_matrix(sps.load_npz(stitched / "stitched_A.npz"))
    return s.baseline_units("cnmf" if which == "baseline_final" else "init")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument(
        "--footprints", choices=("baseline_final", "baseline_init", "stitched"), required=True
    )
    ap.add_argument("--stitched", type=Path, help="t1 output dir with stitched_A.npz")
    ap.add_argument("--n-updates", type=int, default=0, help="spatial updates between passes")
    ap.add_argument("--segment-frames", type=int, default=6000, help="frames per sum segment")
    ap.add_argument(
        "--batch", type=int, default=100, help="frames per read (100 x 600 x 600 float32 = 144 MB)"
    )
    ap.add_argument("--sn", choices=("factors", "diff"), default="factors")
    ap.add_argument(
        "--mean-noise",
        choices=("factors", "windows"),
        default="factors",
        help="first mean image and noise: baseline factors, or the --stitched windows' runs",
    )
    ap.add_argument("--merge-corr", type=float, help="default: params.loop_merge_corr")
    ap.add_argument("--sparse-snr", type=float, help="update penalty; default: params.sparse_snr")
    ap.add_argument(
        "--merge-on",
        choices=("lowpass", "raw"),
        default="lowpass",
        help="traces the merge correlates: causally low-passed at params.noise_freq, or raw",
    )
    ap.add_argument(
        "--project",
        choices=("none", "windows", "merged"),
        default="none",
        help="fit projected frames: the --stitched windows' pass-2 bases, per window or merged",
    )
    ap.add_argument("--cap", type=int, default=24, help="merged basis: columns per tile")
    ap.add_argument(
        "--background",
        choices=("fit", "mean"),
        default="fit",
        help="b: the mean image minus its LS fit on A (today), or the mean image itself",
    )
    a = ap.parse_args()

    s = st.Session(a.session)
    params = s.params()
    H, W, T = s.shape()
    sc = s.sidecar()
    if a.mean_noise == "windows":
        ymean, sn_fft = st.window_mean_noise(a.stitched / "runs")
    else:
        fac = s.factors()  # mean image and noise only (see the module docstring)
        ymean, sn_fft = fac.mean_flat(), fac.sigma_flat()
        del fac
    frames = st.FrameStream(io.probe(s.raw), sc.shifts, params, a.batch)
    merge_corr = params.loop_merge_corr if a.merge_corr is None else a.merge_corr
    # the first runs merged on raw traces and keep their folder names
    suffix = "" if a.merge_on == "raw" or a.n_updates == 0 else "_lp"
    suffix += "_win" if a.mean_noise == "windows" else ""
    project = None
    if a.project != "none":
        spans, bases = st.basis_from_windows(a.stitched / "runs", a.project, a.cap)
        project = st.WindowProjector(spans, bases)
        suffix += f"_{a.project}" + (f"{a.cap}" if a.project == "merged" else "")
    suffix += "_bmean" if a.background == "mean" else ""
    sparse_snr = params.sparse_snr if a.sparse_snr is None else a.sparse_snr
    suffix += "" if a.sparse_snr is None else f"_snr{a.sparse_snr:g}"
    out = s.out(f"t2_{a.footprints}_u{a.n_updates}{suffix}")

    A = start_footprints(s, a.footprints, a.stitched)
    C_ref = s.baseline_units("cnmf")[1] if a.footprints == "baseline_final" else None
    results: dict = {
        "args": {k: str(v) for k, v in vars(a).items()},
        "shortcuts": {
            "first_mean_image": "baseline factors" if a.mean_noise == "factors" else "windows",
            "noise": a.sn if a.mean_noise == "factors" else "windows (quadratic mean)",
        },
        "project": a.project,
        "background": a.background,
        "basis_mb": None if project is None else max(b.nbytes for b in bases) / 2**20,
        "baseline": st.brief(s.baseline_score()),
        "passes": [],
    }
    for it in range(a.n_updates + 1):
        print(f"pass {it}: {A.shape[0]} units", flush=True)
        model = st.TraceModel.build(A, ymean, a.background)
        sup = st.Support.build(A, H, W, params.dl_wnd)
        res = st.stream_pass(
            frames, model, sup, T, a.segment_frames, params.noise_freq, project=project
        )
        sps.save_npz(out / f"pass{it}_A.npz", sps.csr_matrix(A))
        np.save(out / f"pass{it}_C.npy", res.C)
        rec = {
            "pass": it,
            "score": st.brief(st.score(s, A, res.C, sc.shifts)),
            "seconds": res.seconds,
            "timing": res.timing,
            "state_mb": res.state_bytes / 2**20,
            "peak_rss_mb": st.rss_mb(),
        }
        if C_ref is not None and it == 0:  # same units, same order as the batch traces
            rec["corr_with_batch_traces"] = st.trace_agreement(res.C, C_ref)
        results["passes"].append(rec)
        b = rec["score"]
        print(
            f"  recall {b['recall']:.3f} precision {b['precision']:.3f} F1 {b['f1']:.3f} "
            f"trace corr {b['trace_corr_median']}, state {rec['state_mb']:.0f} MB, "
            f"{res.seconds:.0f} s",
            flush=True,
        )
        (out / "results.json").write_text(json.dumps(results, indent=1))
        if it == a.n_updates:
            break

        tot = st.total(res.stats)
        ymean = tot.ysum / tot.n  # later passes: the streamed mean image
        sn = sn_fft if a.sn == "factors" else st.diff_noise(tot)
        A_sp, mask, _ = st.spatial_from_stats(A, sup, tot, sn, sparse_snr, params.size_thres)
        if it == 0:
            stab = st.segment_stability(A_sp, mask, A, sup, res.stats, sn, sparse_snr, H, W)
            np.savez(out / "stability.npz", **stab)
            results["stability"] = {
                "cos_median_per_segment": np.nanmedian(stab["cos"], axis=1).tolist(),
                "shift_px_median_per_segment": np.nanmedian(stab["shift"], axis=1).tolist(),
            }
        A = st.merge_from_stats(A_sp, tot, mask, merge_corr, a.merge_on == "lowpass")
        print(f"  update: {mask.sum()} kept, {A.shape[0]} after merge", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
