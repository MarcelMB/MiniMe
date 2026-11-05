"""Reference outputs of MiniMe's pass 1 on one window, for checking the Rust core.

Runs ``pass1.run_pass1`` exactly as a ``fresh`` window run does (``stream.window_units_fresh``:
the window's raw segments only, the baseline sidecar's shifts and template, one worker) on
frames ``[t0, t1)`` and writes, under ``processed/stream/ref_pass1_<t0>_<t1>/``:

* ``inputs/``: ``params.json`` (the pass-1 parameters plus ``H``, ``W``, ``t0``, ``t1``),
  ``shifts.npy`` ((n, 2) float32, the sidecar's) and ``template.npy`` (the sidecar's
  prepared template image, float32);
* ``ref/``: ``max_proj.npy``, ``block_max.npy`` (uint8), ``block_edges.npy``,
  ``quality.npy`` (float32), the tile store ``tiles/`` (``tile_NNN.part00``) and
  ``tile_crc.npy`` (CRC-32 of every tile's bytes, so a browser run can be checked without
  copying its tile store out).

    python experiments/streaming/ref_pass1.py --t0 0 --t1 3000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import zlib
from pathlib import Path

import numpy as np
import stream as st

from minime import io, pass1, pipeline
from minime.params import Params
from minime.tiles import TileStore, tile_grid


def pass1_meta(params: Params, H: int, W: int, t0: int, t1: int) -> dict:
    return {
        "H": H,
        "W": W,
        "t0": t0,
        "t1": t1,
        **params.motion_params(),
        "motion_bandpass": list(params.motion_bandpass),
        "tile": params.tile,
        "seeds_wnd": params.seeds_wnd,
        "seeds_stp": params.seeds_stp,
        "template_source": params.template_source,
        "template_chunks": params.template_chunks,
        "motion_fill": params.motion_fill,
    }


def tile_crcs(store: TileStore) -> np.ndarray:
    """CRC-32 of every tile's bytes (all parts in order), grid order."""
    out = []
    for t in store.grid:
        crc = 0
        for p in range(store.n_parts):
            crc = zlib.crc32(store.part_path(t.index, p).read_bytes(), crc)
        out.append(crc)
    return np.asarray(out, np.uint32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=st.DEFAULT_SESSION)
    ap.add_argument("--baseline", default="baseline", help="the batch run whose sidecar to use")
    ap.add_argument("--name", default="ref_pass1", help="output folder prefix")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=3000)
    a = ap.parse_args()

    s = st.Session(a.session, baseline=a.baseline)
    params = s.params()
    sc = s.sidecar()
    full = io.probe(s.raw)
    out = s.out(f"{a.name}_{a.t0}_{a.t1}")
    inp, ref = out / "inputs", out / "ref"
    inp.mkdir(exist_ok=True)
    ref.mkdir(exist_ok=True)
    info = io.probe(st.window_raw_dir(full, a.t0, a.t1, out / "work"))
    H, W = info.height, info.width
    shifts = np.ascontiguousarray(sc.shifts[a.t0 : a.t1], dtype=np.float32)
    (inp / "params.json").write_text(json.dumps(pass1_meta(params, H, W, a.t0, a.t1), indent=1))
    np.save(inp / "shifts.npy", shifts)
    np.save(inp / "template.npy", sc.template.astype(np.float32))
    template = pipeline._template_from_sidecar(sc, info, params)
    if params.motion_fill == "template":  # the session's fill image, as pipeline.run uses it
        if sc.fill is None:
            raise SystemExit("the baseline's sidecar holds no fill image (run it again)")
        np.save(inp / "fill.npy", sc.fill)
        template = dataclasses.replace(template, fill=sc.fill)
    store = TileStore(ref / "tiles", tile_grid(H, W, params.tile), 1)
    store.remove()
    t = time.perf_counter()
    p1 = pass1.run_pass1(info, store, template, params, 1, shifts=shifts)
    seconds = time.perf_counter() - t
    np.save(ref / "max_proj.npy", p1.max_proj)
    np.save(ref / "block_max.npy", p1.block_max)
    np.save(ref / "block_edges.npy", p1.block_edges)
    np.save(ref / "quality.npy", p1.quality)
    np.save(ref / "tile_crc.npy", tile_crcs(store))
    summary = {
        "frames": p1.frames,
        "blocks": int(p1.block_max.shape[0]),
        "tiles": len(store.grid),
        "store_mb": store.nbytes() / 2**20,
        "seconds": seconds,
        "stage_seconds": p1.seconds,
    }
    (ref / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"reference: {out}")


if __name__ == "__main__":
    main()
