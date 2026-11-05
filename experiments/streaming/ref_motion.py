"""Reference outputs of MiniMe's motion stage on one window, for checking the Rust core.

Builds the session template exactly as ``pipeline.run`` does (``pass1.build_template``:
the first ``template_frames`` frames of segment 0), then estimates the shift of every frame
in ``[t0, t1)`` the way ``pass1.Pass1.process`` does, and writes, under
``processed/stream/ref_motion_<t0>_<t1>/``:

* ``inputs/``: ``params.json`` (the motion parameters plus ``H``, ``W``, ``t0``, ``t1``) and
  ``check_frames.npy`` (the frames whose images the Rust core also saves);
* ``ref/``: ``template.npy`` (prepared template image, float32), ``shifts.npy`` ((n, 2)
  float64, before pass 1's float32 cast), ``quality.npy`` (float32), and for the frames
  in ``CHECK_FRAMES`` the registration image ``reg.npy`` (float32) and the prepared,
  windowed image ``prep.npy`` (float32, ``frames_t.npy`` holds their indices).

    python experiments/streaming/ref_motion.py --t0 0 --t1 3000
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import stream as st

from minime import io, motion, pass1, preprocess
from minime.params import Params

CHECK_FRAMES = (0, 1, 199, 200, 1000, 2999)


def window_frames(info: io.RawInfo, t0: int, t1: int) -> Iterator[tuple[int, np.ndarray]]:
    """Raw frames ``[t0, t1)`` as (first index, (n, h, w) uint8) per overlapping segment."""
    starts = st.segment_starts(info)
    for seg, s0, s1 in zip(info.segments, starts[:-1], starts[1:], strict=True):
        if s1 <= t0 or s0 >= t1:
            continue
        fr = io.read_avi(seg, max_frames=int(min(s1, t1) - s0), bgr=info.bgr)
        lo = max(t0 - int(s0), 0)
        yield int(s0) + lo, fr[lo:]


def motion_meta(params: Params, H: int, W: int, t0: int, t1: int) -> dict:
    return {
        "H": H,
        "W": W,
        "t0": t0,
        "t1": t1,
        **params.motion_params(),
        "motion_bandpass": list(params.motion_bandpass),
        # not a sidecar key (the shifts do not depend on it), but the Rust core builds the fill
        "template_source": params.template_source,
        "template_chunks": params.template_chunks,
        "motion_fill": params.motion_fill,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--baseline", default="baseline", help="the batch run whose params to use")
    ap.add_argument("--name", default="ref_motion", help="output folder prefix")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=200, help="frames per estimate call")
    a = ap.parse_args()

    s = st.Session(a.session, baseline=a.baseline)
    params = s.params()
    info = io.probe(s.raw)
    H, W = info.height, info.width
    out = s.out(f"{a.name}_{a.t0}_{a.t1}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)
    (inp / "params.json").write_text(json.dumps(motion_meta(params, H, W, a.t0, a.t1), indent=1))
    want = [t for t in CHECK_FRAMES if a.t0 <= t < a.t1]
    np.save(inp / "check_frames.npy", np.asarray(want, np.int64))

    t = time.perf_counter()
    tmpl = pass1.build_template(info, params)
    t_template = time.perf_counter() - t
    np.save(ref / "template.npy", motion.template_image(tmpl))
    if tmpl.fill is not None:
        np.save(ref / "fill.npy", tmpl.fill)

    shifts, quality, reg_chk, prep_chk, chk_t = [], [], [], [], []
    t = time.perf_counter()
    for f0, frames in window_frames(info, a.t0, a.t1):
        for b0 in range(0, frames.shape[0], a.batch):
            den = preprocess.denoise(frames[b0 : b0 + a.batch], params.median_ksize)
            reg = pass1.registration_image(den, params)
            sh = motion.estimate(reg, tmpl, params.motion_max_shift)
            warped = motion.apply_shifts(reg, sh)
            shifts.append(sh)
            quality.append(motion.frame_quality(warped, tmpl))
            for i in range(reg.shape[0]):
                if f0 + b0 + i in CHECK_FRAMES:
                    chk_t.append(f0 + b0 + i)
                    reg_chk.append(reg[i])
                    prep_chk.append(motion._prep(reg[i : i + 1], tmpl.down)[0])
    seconds = time.perf_counter() - t
    sh = np.concatenate(shifts)
    np.save(ref / "shifts.npy", sh)
    np.save(ref / "quality.npy", np.concatenate(quality))
    np.save(ref / "frames_t.npy", np.asarray(chk_t, np.int64))
    np.save(ref / "reg.npy", np.stack(reg_chk))
    np.save(ref / "prep.npy", np.stack(prep_chk))

    base = s.sidecar().shifts[a.t0 : a.t1]
    summary = {
        "frames": int(sh.shape[0]),
        "template_seconds": t_template,
        "estimate_seconds": seconds,
        "max_abs_shift": float(np.abs(sh).max()),
        "max_diff_vs_baseline_sidecar": float(np.abs(sh.astype(np.float32) - base).max()),
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
