"""Pass 1: frame-major preprocessing, rigid motion correction, seed statistics, tile store."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from minime import motion, preprocess
from minime.io import RawInfo, iter_avi, read_avi
from minime.motion import Template
from minime.params import Params
from minime.tiles import Tile, TileStore, pack_tiles, tile_offsets

STAGES = (
    "read_avi",
    "denoise",
    "bandpass",
    "tophat",
    "motion_estimate",
    "motion_apply",
    "max_proj",
    "block_max",
    "pack",
    "write_tiles",
)
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
READ_BATCH = 1000  # frames decoded, processed and written at a time by a pass-1 worker


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


def registration_image(denoised: np.ndarray, params: Params) -> np.ndarray:
    """The frames motion is estimated on: band-passed (before the tophat) or tophat-filtered."""
    if params.motion_before_tophat:
        return preprocess.bandpass(denoised, *params.motion_bandpass)
    return preprocess.tophat(denoised, params.tophat_wnd)


def build_template(info: RawInfo, params: Params) -> Template:
    """Motion template from the first ``template_frames`` frames of segment 0."""
    frames = read_avi(info.segments[0], max_frames=params.template_frames, bgr=info.bgr)
    den = preprocess.denoise(frames, params.median_ksize)
    reg = registration_image(den, params)
    return motion.make_template(reg, params.motion_down, params.motion_max_shift)


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
            t = time.perf_counter()

            def lap(stage: str) -> None:
                nonlocal t
                now = time.perf_counter()
                s[stage] += now - t
                t = now

            blk = preprocess.denoise(blk, p.median_ksize)
            lap("denoise")
            reg = registration_image(blk, p)
            lap("bandpass" if p.motion_before_tophat else "tophat")
            if self.given_shifts is None:
                sh = motion.estimate(reg, self.template, p.motion_max_shift)
            else:
                a = self.t_start + self.frames + b0
                sh = self.given_shifts[a : a + blk.shape[0]]
                if sh.shape[0] != blk.shape[0]:
                    raise ValueError(
                        f"frames {a}..{a + blk.shape[0]} decoded, but only "
                        f"{self.given_shifts.shape[0]} shifts given"
                    )
            lap("motion_estimate")
            if p.motion_before_tophat:
                blk = motion.apply_shifts(blk, sh)
                reg = motion.apply_shifts(reg, sh)
                lap("motion_apply")
                blk = preprocess.tophat(blk, p.tophat_wnd)
                lap("tophat")
            else:
                blk = reg = motion.apply_shifts(reg, sh)
                lap("motion_apply")
            np.maximum(self.max_proj, blk.max(axis=0), out=self.max_proj)
            lap("max_proj")
            self._update_block_max(blk, self.t_start + self.frames + b0)
            lap("block_max")
            self.quality.append(motion.frame_quality(reg, self.template))
            out[b0 : b0 + blk.shape[0]] = blk
            self.shifts.append(sh.astype(np.float32))
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


def run_workers(
    fn: Callable[..., Any], jobs: list[tuple], initializer: Callable[[], None] | None = None
) -> list[Any]:
    """``fn(*job)`` for every job in single-threaded spawned processes; results in job order.

    A worker that dies (killed for memory, say) raises ``BrokenProcessPool``; a
    ``multiprocessing.Pool`` would wait for its result forever.
    """
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(len(jobs), mp_context=ctx, initializer=initializer) as ex:
            return list(ex.map(fn, *zip(*jobs, strict=True)))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _block_range(edges: np.ndarray, t_start: int, frames: int) -> tuple[int, int]:
    """Blocks ``[a, b)`` that frames ``[t_start, t_start + frames)`` fall in."""
    a = int(np.searchsorted(edges, t_start, "right")) - 1
    return a, max(a, int(np.searchsorted(edges, t_start + frames, "left")))


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
    bgr: bool = False,
    read_batch: int = READ_BATCH,
) -> dict:
    """Process a contiguous run of segments, writing part ``part`` of every tile file.

    Each segment is streamed in batches of ``read_batch`` frames (``io.iter_avi``), so a
    worker holds about 3 batches whatever the segment length. Returns the decoded frame
    count of every segment (``segment_frames``) for ``check_segment_frames``. ``block_max``
    holds only the blocks ``block_range`` the worker's frames fall in.
    """
    cv2.setNumThreads(1)
    store = TileStore(Path(store_root), grid, n_parts)
    p1 = Pass1(template, grid, params, block_edges, t_start, shifts)
    t_read = t_write = 0.0
    decoded: list[int] = []
    with ExitStack() as stack:
        files = [stack.enter_context(f) for f in store.open_part(part)]
        for path in paths:
            batches = iter_avi(Path(path), read_batch, bgr)
            n_seg = 0
            while True:
                t0 = time.perf_counter()
                frames = next(batches, None)
                t_read += time.perf_counter() - t0
                if frames is None:
                    break
                n_seg += frames.shape[0]
                packed = p1.process(frames)
                t0 = time.perf_counter()
                for f, (a, b) in zip(files, tile_offsets(frames.shape[0], grid), strict=True):
                    f.write(memoryview(packed[a:b]))
                t_write += time.perf_counter() - t0
            decoded.append(n_seg)
    a, b = _block_range(block_edges, t_start, p1.frames)
    return {
        "seconds": {**p1.seconds, "read_avi": t_read, "write_tiles": t_write},
        "frames": p1.frames,
        "segment_frames": decoded,
        "max_proj": p1.max_proj,
        "block_range": (a, b),
        "block_max": p1.block_max[a:b].copy(),
        # a worker whose segments hold 0 frames (a recording stopped right after a new file)
        "shifts": np.concatenate(p1.shifts) if p1.shifts else np.empty((0, 2), np.float32),
        "quality": np.concatenate(p1.quality) if p1.quality else np.empty(0, np.float32),
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
            info.bgr,
        )
        for k, r in enumerate(ranges)
    ]
    res = run_workers(pass1_worker, jobs, _init_worker)
    check_segment_frames(info, [n for r in res for n in r["segment_frames"]])
    block_max = np.zeros((len(edges) - 1, *template.shape), dtype=np.uint8)
    for r in res:
        a, b = r["block_range"]
        np.maximum(block_max[a:b], r["block_max"], out=block_max[a:b])
    return Pass1Result(
        shifts=np.concatenate([r["shifts"] for r in res]).astype(np.float32),
        quality=np.concatenate([r["quality"] for r in res]).astype(np.float32),
        max_proj=np.maximum.reduce([r["max_proj"] for r in res]),
        block_max=block_max,
        block_edges=edges,
        frames=int(sum(r["frames"] for r in res)),
        seconds={k: float(sum(r["seconds"][k] for r in res)) for k in STAGES},
    )
