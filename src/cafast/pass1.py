"""Pass 1: frame-major preprocessing, rigid motion correction, seed statistics, tile store."""

from __future__ import annotations

import multiprocessing
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cafast import motion, preprocess
from cafast.io import RawInfo, read_avi
from cafast.motion import Template
from cafast.params import Params
from cafast.tiles import Tile, TileStore, pack_tiles, tile_offsets

STAGES = (
    "read_avi",
    "denoise",
    "tophat",
    "motion_estimate",
    "motion_apply",
    "max_proj",
    "block_max",
    "pack",
    "write_tiles",
)
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------------------------
# Block edges and template
# ---------------------------------------------------------------------------


def seed_block_edges(T: int, wnd_size: int, stp_size: int) -> np.ndarray:
    """Sorted unique starts/stops of Minian's rolling seed windows, plus 0 and T.

    Windows are ``[floor(c - hwnd).clip(0), min(ceil(c + hwnd), T))`` for ``ceil(T / stp) + 1``
    centres evenly spaced over ``[0, T - 1]`` and ``hwnd = ceil(wnd / 2)``.
    """
    nstp = int(np.ceil(T / stp_size) + 1)
    hwnd = np.ceil(wnd_size / 2)
    edges = {0, T}
    for c in np.linspace(0, T - 1, nstp):
        edges.add(int(np.floor(c - hwnd).clip(0)))
        edges.add(min(int(np.ceil(c + hwnd)), T))
    return np.array(sorted(edges), dtype=np.int64)


def build_template(info: RawInfo, params: Params) -> Template:
    """Motion template from the first ``template_frames`` frames of segment 0."""
    frames = read_avi(info.segments[0], max_frames=params.template_frames)
    pre = preprocess.preprocess(frames, params.median_ksize, params.tophat_wnd)
    return motion.make_template(pre, params.motion_down, params.motion_max_shift)


# ---------------------------------------------------------------------------
# Frame processor
# ---------------------------------------------------------------------------


class Pass1:
    """Stateful frame-batch processor: running max, block maxima, shifts, quality, timings.

    With ``shifts`` (T, 2) given, frames are warped by ``shifts[absolute frame index]``
    instead of being estimated against the template (apply-only rerun from a sidecar).
    """

    def __init__(
        self,
        template: Template,
        grid: list[Tile],
        params: Params,
        block_edges: np.ndarray,
        t_start: int,
        shifts: np.ndarray | None = None,
    ) -> None:
        self.template = template
        self.given_shifts = shifts
        self.grid = grid
        self.params = params
        self.block_edges = block_edges
        self.t_start = t_start
        h, w = template.shape
        self.max_proj = np.zeros((h, w), dtype=np.uint8)
        self.block_max = np.zeros((len(block_edges) - 1, h, w), dtype=np.uint8)
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.shifts: list[np.ndarray] = []
        self.quality: list[np.ndarray] = []
        self.frames = 0

    def _update_block_max(self, blk: np.ndarray, blk_t0: int) -> None:
        """Fold corrected frames ``blk`` (absolute index ``blk_t0`` onward) into ``block_max``."""
        edges = self.block_edges
        first = int(np.searchsorted(edges, blk_t0, "right")) - 1
        for bi in range(first, len(edges) - 1):
            lo, hi = edges[bi] - blk_t0, edges[bi + 1] - blk_t0
            if lo >= blk.shape[0]:
                break
            part = blk[max(lo, 0) : min(hi, blk.shape[0])]
            if part.size:
                np.maximum(self.block_max[bi], part.max(axis=0), out=self.block_max[bi])

    def process(self, frames: np.ndarray) -> np.ndarray:
        """Corrected, tile-packed uint8 buffer for this batch (see ``pack_tiles``)."""
        p, s = self.params, self.seconds
        out = np.empty_like(frames)
        for b0 in range(0, frames.shape[0], p.sub_batch):
            blk = frames[b0 : b0 + p.sub_batch]
            t0 = time.perf_counter()
            blk = preprocess.denoise(blk, p.median_ksize)
            t1 = time.perf_counter()
            blk = preprocess.tophat(blk, p.tophat_wnd)
            t2 = time.perf_counter()
            if self.given_shifts is None:
                sh = motion.estimate(blk, self.template, p.motion_max_shift)
            else:
                a = self.t_start + self.frames + b0
                sh = self.given_shifts[a : a + blk.shape[0]]
                if sh.shape[0] != blk.shape[0]:
                    raise ValueError(
                        f"frames {a}..{a + blk.shape[0]} decoded, but only "
                        f"{self.given_shifts.shape[0]} shifts given"
                    )
            t3 = time.perf_counter()
            blk = motion.apply_shifts(blk, sh)
            t4 = time.perf_counter()
            np.maximum(self.max_proj, blk.max(axis=0), out=self.max_proj)
            t5 = time.perf_counter()
            self._update_block_max(blk, self.t_start + self.frames + b0)
            t6 = time.perf_counter()
            self.quality.append(motion.frame_quality(blk, self.template))
            out[b0 : b0 + blk.shape[0]] = blk
            self.shifts.append(sh.astype(np.float32))
            stages = ("denoise", "tophat", "motion_estimate", "motion_apply", "max_proj")
            for k, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4), strict=True):
                s[k] += dt
            s["block_max"] += t6 - t5
        t0 = time.perf_counter()
        packed = pack_tiles(out, self.grid)
        s["pack"] += time.perf_counter() - t0
        self.frames += frames.shape[0]
        return packed


# ---------------------------------------------------------------------------
# Worker and driver
# ---------------------------------------------------------------------------


@dataclass
class Pass1Result:
    shifts: np.ndarray  # (T, 2) float32
    quality: np.ndarray  # (T,) float32
    max_proj: np.ndarray  # (H, W) uint8
    block_max: np.ndarray  # (n_blocks, H, W) uint8
    block_edges: np.ndarray  # (n_blocks + 1,) int64
    frames: int  # decoded frames (equals the probed count, else run_pass1 raises)
    seconds: dict[str, float]  # CPU seconds per stage, summed over workers


def _split(n: int, parts: int) -> list[range]:
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [range(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def n_parts_for(info: RawInfo, workers: int) -> int:
    """Number of tile-store parts (non-empty worker ranges) ``run_pass1`` uses."""
    return len(_split(len(info.segments), workers))


def _init_worker() -> None:
    cv2.setNumThreads(1)


def pass1_worker(
    paths: list[str],
    part: int,
    store_root: str,
    grid: list[Tile],
    n_parts: int,
    template: Template,
    params: Params,
    block_edges: np.ndarray,
    t_start: int,
    shifts: np.ndarray | None = None,
) -> dict:
    """Process a contiguous run of segments, writing part ``part`` of every tile file."""
    cv2.setNumThreads(1)
    store = TileStore(Path(store_root), grid, n_parts)
    p1 = Pass1(template, grid, params, block_edges, t_start, shifts)
    t_read = t_write = 0.0
    decoded: list[int] = []
    with ExitStack() as stack:
        files = [stack.enter_context(f) for f in store.open_part(part)]
        for path in paths:
            t0 = time.perf_counter()
            frames = read_avi(Path(path))
            t_read += time.perf_counter() - t0
            decoded.append(int(frames.shape[0]))
            packed = p1.process(frames)
            t0 = time.perf_counter()
            for f, (a, b) in zip(files, tile_offsets(frames.shape[0], grid), strict=True):
                f.write(memoryview(packed[a:b]))
            t_write += time.perf_counter() - t0
    return {
        "seconds": {**p1.seconds, "read_avi": t_read, "write_tiles": t_write},
        "frames": p1.frames,
        "segment_frames": decoded,
        "max_proj": p1.max_proj,
        "block_max": p1.block_max,
        "shifts": np.concatenate(p1.shifts),
        "quality": np.concatenate(p1.quality),
    }


def check_segment_frames(info: RawInfo, decoded: list[int]) -> None:
    """Raise ``ValueError`` naming every segment whose decoded frame count differs from probe.

    Block edges and each worker's first frame index come from the probed counts, so a
    mismatch (a truncated or mis-labelled segment) would misplace frames silently.
    """
    bad = [
        f"{seg.name}: probed {exp}, decoded {got}"
        for seg, exp, got in zip(info.segments, info.segment_frames, decoded, strict=True)
        if exp != got
    ]
    if bad:
        raise ValueError(
            "decoded frame counts differ from the container frame counts "
            f"(truncated or corrupt segment?): {'; '.join(bad)}"
        )


def run_pass1(
    info: RawInfo,
    store: TileStore,
    template: Template,
    params: Params,
    workers: int,
    shifts: np.ndarray | None = None,
) -> Pass1Result:
    """Run pass 1 over all segments with ``workers`` single-threaded processes.

    ``shifts`` (T, 2): apply these instead of estimating motion (``template`` is then used
    only for the per-frame quality).
    """
    if shifts is not None:
        shifts = np.asarray(shifts, dtype=np.float32)
        if shifts.shape != (info.frames, 2):
            raise ValueError(f"shifts must be ({info.frames}, 2), got {shifts.shape}")
    edges = seed_block_edges(info.frames, params.seeds_wnd, params.seeds_stp)
    starts = np.concatenate([[0], np.cumsum(info.segment_frames)]).astype(int)
    ranges = _split(len(info.segments), workers)
    if len(ranges) != store.n_parts:
        raise ValueError(
            f"store has {store.n_parts} parts, pass 1 needs {len(ranges)}; "
            "size it with n_parts_for(info, workers)"
        )
    jobs = [
        (
            [str(info.segments[j]) for j in r],
            k,
            str(store.root),
            store.grid,
            store.n_parts,
            template,
            params,
            edges,
            int(starts[r.start]),
            shifts,
        )
        for k, r in enumerate(ranges)
    ]
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(len(jobs), initializer=_init_worker) as pool:
            res = pool.starmap(pass1_worker, jobs, chunksize=1)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    check_segment_frames(info, [n for r in res for n in r["segment_frames"]])
    return Pass1Result(
        shifts=np.concatenate([r["shifts"] for r in res]).astype(np.float32),
        quality=np.concatenate([r["quality"] for r in res]).astype(np.float32),
        max_proj=np.maximum.reduce([r["max_proj"] for r in res]),
        block_max=np.maximum.reduce([r["block_max"] for r in res]),
        block_edges=edges,
        frames=int(sum(r["frames"] for r in res)),
        seconds={k: float(sum(r["seconds"][k] for r in res)) for k in STAGES},
    )
