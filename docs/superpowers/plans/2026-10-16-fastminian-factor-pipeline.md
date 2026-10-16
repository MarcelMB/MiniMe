# fastminian Factor-First Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `fastminian run <raw_dir>` command that compresses a Miniscope AVI session into per-tile SVD factors (window 1) and runs seeds, init and two CNMF iterations entirely on those factors (window 2), scoring at least as well as Minian against synthetic ground truth.

**Architecture:** Window 1 is a two-pass streaming engine (frame-major pass 1 → uint8 tile store → tile-major pass 2 → `factors/`), a straight port of the measured `pre_rect_down2` prototype. Window 2 holds every tile's factors `W (P_b×r)`, `V (T×r)`, `mean`, `sigma`, `energy` in RAM and expresses each Minian stage as small matrix algebra on them, streaming reconstructed chunks only where a nonlinearity (clip, max, percentile) needs it.

**Tech Stack:** Python 3.11, numpy, scipy, numba, opencv-python-headless, scikit-learn, pandas, typer; pytest + ruff; `uv` venv.

**Spec:** `docs/superpowers/specs/2026-10-16-fastminian-factor-pipeline-design.md`

**Prototype root** (read-only, referenced as `$BENCH` below):
`/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/minian_v2.0/scripts/`.
Minian reference source (read-only, `$MINIAN`):
`/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/minian_v2.0/minian/`.
Test data: `/Users/mbrosch/minian_testing_local/synth_v4_5min/{raw,meta.json}` (9000 frames,
600×600) and, once Task 1 generates it, `synth_v4_10min`.

## Global Constraints

- Python ≥ 3.11; run everything with `.venv/bin/python` / `.venv/bin/pytest` from the repo root.
- Dependencies only: numpy, scipy, numba, opencv-python-headless, scikit-learn, pandas, typer. Minian is never imported by package code.
- Every function signature carries type hints. `ruff format` and `ruff check --fix` clean before each commit (config in `pyproject.toml`: line length 100, rules `E,F,I,UP,B,SIM`).
- Raw data is read-only. Outputs go to `<raw_dir>/../processed/<run_name>/`.
- Each test file runs in under 10 s on tiny synthetic data; nothing in `tests/` touches `~/minian_testing_local`.
- Commit on branch `feat/factor-pipeline`, one sentence, no body, no attribution trailer. Never push.
- Numerics follow the prototypes exactly where a prototype exists (same dtypes, same chunking constants). Deviations are listed in the task.
- Shift convention: `shifts[t] = (dy, dx)` as returned by `$BENCH/bench_browser/candidates/motion.py:estimate`; the warp is `cv2.warpAffine` with `M = [[1,0,-dx],[0,1,-dy]]`.

## Review Focus

1. **Segments with differing frame counts** (the last AVI of a session is shorter): pass 1, the tile store and `block_edges` must use the real per-segment counts, not `segment_frames` from meta. Test in Task 4.
2. **A tile whose every component fails the rank cut** (dark corner, vignetted edge): `compress_tile` must still return rank 1, and window 2 must tolerate `r = 1` tiles. Test in Task 5.
3. **Zero seeds after PNR/GMM, or a seed with empty footprint after `initA`**: the pipeline must stop with a clear error, not a shape exception deep in numba. Tests in Tasks 6 and 8.
4. **A unit with `size_thres` violation in iteration 1 that another unit depends on in `unit_merge`**: masks must be applied consistently to `A`, `C`, `YrA`. Test in Task 10.
5. **Rerun on the same raw folder with the same params**: the sidecar is reused, the store is rebuilt, outputs are overwritten, no stale files from a previous run with more units remain. Test in Task 11.

---

### Task 1: Scaffold, synthetic generator, test data

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`, `src/fastminian/__init__.py`, `src/fastminian/testing/__init__.py`, `src/fastminian/testing/synth.py`, `src/fastminian/cli.py` (synth subcommand only), `tests/conftest.py`, `tests/test_synth.py`

**Interfaces:**
- Produces:
  - `synth.synth_chunks(h: int, w: int, T: int, seed: int = 0, chunk: int = 50) -> Iterator[tuple[int, np.ndarray]]` — identical draws to `$BENCH/bench_browser/heavy_pass.py:synth_chunks`.
  - `synth.synth_movie(h, w, T, seed=0) -> np.ndarray` (T, h, w) uint8.
  - `synth.SynthSetup` dataclass and `synth.synth_setup(h, w, T, seed=0) -> SynthSetup`, `synth.truth_chunks(setup, h, w, chunk=100) -> Iterator[tuple[int, np.ndarray, np.ndarray]]` — port of `$BENCH/bench_browser/synth_truth.py`.
  - `synth.true_centroids(setup) -> np.ndarray` (n_cells, 2) float: `(ys + 7 - margin, xs + 7 - margin)` in frame coordinates of an unshifted frame.
  - `synth.write_session(out_dir: Path, minutes: float, fps: int = 30, segment: int = 1000, seed: int = 0, side: int = 600) -> dict` — writes `N.avi` (fourcc GREY) into `out_dir` and `meta.json` into `out_dir.parent`, as `$BENCH/bench_browser/gen_dataset.py`.
  - `tests/conftest.py` fixture `tiny_session(tmp_path)` → dict with `raw_dir` (3 segments of 100 frames, 64×64, seed 0), `setup`, `T=300`.
  - CLI `fastminian synth --minutes 10 --out ~/minian_testing_local/synth_v4_10min/raw`.

- [ ] **Step 1: Write pyproject, venv, package skeleton**

`pyproject.toml`:
```toml
[project]
name = "fastminian"
version = "0.0.1"
description = "Factor-first CNMF pipeline for one-photon miniscope data"
requires-python = ">=3.11"
dependencies = [
  "numpy>=2.0", "scipy>=1.13", "numba>=0.60", "opencv-python-headless>=4.9",
  "scikit-learn>=1.4", "pandas>=2.2", "typer>=0.12",
]
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
[project.scripts]
fastminian = "fastminian.cli:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/fastminian"]
[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
[tool.pytest.ini_options]
testpaths = ["tests"]
```
Run from the repo root:
```bash
uv venv -p 3.11 && uv pip install -e ".[dev]"
```
`.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, `processed/`.

- [ ] **Step 2: Write the failing generator tests**

`tests/test_synth.py`:
```python
import numpy as np
from fastminian.testing import synth


def test_synth_movie_is_deterministic_uint8():
    a = synth.synth_movie(32, 32, 20, seed=1)
    b = synth.synth_movie(32, 32, 20, seed=1)
    assert a.dtype == np.uint8 and a.shape == (20, 32, 32)
    assert np.array_equal(a, b)


def test_truth_matches_generator_without_noise():
    h, w, T = 48, 48, 60
    setup = synth.synth_setup(h, w, T, seed=0)
    movie = synth.synth_movie(h, w, T, seed=0)
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, h, w, chunk=25)])
    # noise sd is 4 grey levels; clean must be within a few sd everywhere
    diff = movie.astype(float) - clean.astype(float)
    assert abs(diff.mean()) < 0.5 and diff.std() < 5.0
    assert setup.shifts.shape == (T, 2)


def test_true_centroids_inside_frame():
    setup = synth.synth_setup(64, 64, 10, seed=0)
    c = synth.true_centroids(setup)
    assert c.shape[1] == 2
    assert (c > -10).all() and (c < 74).all()


def test_write_session_layout(tmp_path):
    meta = synth.write_session(tmp_path / "raw", minutes=0.05, fps=30, segment=30, seed=0, side=32)
    assert sorted(p.name for p in (tmp_path / "raw").glob("*.avi")) == ["0.avi", "1.avi", "2.avi"]
    assert meta["frames"] == 90 and (tmp_path / "meta.json").exists()
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_synth.py -v` → FAIL (`ImportError`).

- [ ] **Step 4: Port the generator and truth**

Copy `synth_chunks`, `synth_movie`, `_gauss_patch`, `_traces` and the constants `MOTION_MARGIN = 8`, `GEN_CHUNK = 50` from `$BENCH/bench_browser/heavy_pass.py` verbatim (same RNG call order: positions, traces, steps, then per-chunk noise). Copy `SynthSetup`, `synth_setup`, `truth_chunks` from `$BENCH/bench_browser/synth_truth.py`. Add:
```python
def true_centroids(setup: SynthSetup) -> np.ndarray:
    half = setup.patch.shape[0] // 2
    return np.stack([setup.ys + half - setup.margin, setup.xs + half - setup.margin], axis=1).astype(float)
```
`write_session` is `gen_dataset.main` as a function (parameters instead of argparse; `side` parameter instead of `FOV`). `cli.py`:
```python
import typer
app = typer.Typer(add_completion=False)

@app.command()
def synth(out: Path, minutes: float = 10.0, fps: int = 30, segment: int = 1000, seed: int = 0) -> None:
    """Write a synthetic Miniscope session (N.avi, GREY) and meta.json."""
    from fastminian.testing.synth import write_session
    typer.echo(write_session(out, minutes, fps, segment, seed))
```
`tests/conftest.py`:
```python
import pytest
from fastminian.testing import synth

@pytest.fixture
def tiny_session(tmp_path):
    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return {"raw_dir": raw, "setup": synth.synth_setup(64, 64, 300, 0), "T": 300, "h": 64, "w": 64}
```

- [ ] **Step 5: Run tests, verify they pass; lint; commit**

Run: `.venv/bin/pytest tests/test_synth.py -v` → PASS. `.venv/bin/ruff format . && .venv/bin/ruff check --fix .`
```bash
git add -A && git commit -m "Scaffold package with synthetic session generator and truth"
```

- [ ] **Step 6: Generate the 10-min dataset (background, ~1 min)**

```bash
.venv/bin/fastminian synth /Users/mbrosch/minian_testing_local/synth_v4_10min/raw --minutes 10
```
Verify `meta.json` says 18000 frames, 18 segments, and `0.avi` from `synth_v4_5min/raw` and `synth_v4_10min/raw` are byte-identical (same seed, same first segment).

---

### Task 2: io and tiles

**Files:**
- Create: `src/fastminian/io.py`, `src/fastminian/tiles.py`, `tests/test_io_tiles.py`

**Interfaces:**
- Produces (`io.py`):
  - `list_segments(raw_dir: Path) -> list[Path]` sorted by integer stem, `*.avi` only.
  - `read_avi(path: Path) -> np.ndarray` (n, h, w) uint8 (grayscale; take channel 0 if 3-D frames come back).
  - `frame_count(path: Path) -> int` via `cv2.CAP_PROP_FRAME_COUNT`.
  - `@dataclass(frozen=True) RawInfo(height: int, width: int, frames: int, segment_frames: list[int], segments: list[Path], fps: float | None)`; `probe(raw_dir: Path) -> RawInfo` (reads one frame of segment 0 for h, w; counts every segment; fps from `raw_dir.parent / "meta.json"` if present).
  - `hash_files(paths: list[Path], sample_mb: int = 1, n_samples: int = 16) -> str` — port of `$BENCH/bench_motion/motion_sidecar.py:hash_files`.
- Produces (`tiles.py`):
  - `@dataclass(frozen=True) Tile(index, y0, y1, x0, x1)` with `pixels`, `shape` properties; `tile_grid(h: int, w: int, tile: int = 64) -> list[Tile]`; `pack_tiles(frames, grid) -> np.ndarray`; `tile_offsets(n, grid) -> list[tuple[int, int]]` — verbatim from `$BENCH/bench_browser/session.py`.
  - `class TileStore(root: Path, grid: list[Tile], n_parts: int)`: `part_path(tile: int, part: int) -> Path` (`tile_{tile:03d}.part{part:02d}`), `open_part(part) -> list[BinaryIO]` (one file per tile, order of `grid`), `read_tile(tile: int) -> np.ndarray` (T, th, tw) uint8 by concatenating all parts, `nbytes() -> int`, `remove() -> None`.

- [ ] **Step 1: Failing tests**

`tests/test_io_tiles.py`:
```python
import numpy as np
from fastminian import io, tiles


def test_segments_and_probe(tiny_session):
    segs = io.list_segments(tiny_session["raw_dir"])
    assert [s.stem for s in segs] == ["0", "1", "2"]
    info = io.probe(tiny_session["raw_dir"])
    assert (info.height, info.width, info.frames) == (64, 64, 300)
    assert info.segment_frames == [100, 100, 100] and info.fps == 30


def test_read_avi_roundtrip(tiny_session):
    from fastminian.testing import synth
    frames = io.read_avi(io.list_segments(tiny_session["raw_dir"])[0])
    assert frames.dtype == np.uint8 and frames.shape == (100, 64, 64)
    assert np.array_equal(frames, synth.synth_movie(64, 64, 300, 0)[:100])


def test_hash_changes_with_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(bytes(range(256)) * 100)
    b.write_bytes(bytes(range(256)) * 99 + bytes(255 - i for i in range(256)))
    assert io.hash_files([a]) == io.hash_files([a]) != io.hash_files([b])


def test_pack_and_store_roundtrip(tmp_path):
    grid = tiles.tile_grid(70, 100, tile=32)  # ragged edge tiles
    assert len(grid) == 3 * 4 and grid[-1].shape == (6, 4)
    rng = np.random.default_rng(0)
    parts = [rng.integers(0, 255, (7, 70, 100), dtype=np.uint8), rng.integers(0, 255, (5, 70, 100), dtype=np.uint8)]
    store = tiles.TileStore(tmp_path / "tiles", grid, n_parts=2)
    for p, frames in enumerate(parts):
        packed = tiles.pack_tiles(frames, grid)
        files = store.open_part(p)
        for f, (a, b) in zip(files, tiles.tile_offsets(frames.shape[0], grid), strict=True):
            f.write(memoryview(packed[a:b]))
            f.close()
    full = np.concatenate(parts)
    for t in grid:
        assert np.array_equal(store.read_tile(t.index), full[:, t.y0 : t.y1, t.x0 : t.x1])
    assert store.nbytes() == full.size
    store.remove()
    assert not (tmp_path / "tiles").exists()
```

- [ ] **Step 2: Run, verify fail.** `.venv/bin/pytest tests/test_io_tiles.py -v`
- [ ] **Step 3: Implement** by copying the named functions from `session.py` / `run_session_local.py` / `motion_sidecar.py`; `TileStore.read_tile` is `pass2_worker`'s `np.fromfile` + `concatenate` + `reshape(-1, *t.shape)`.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add AVI reader, raw probe, content hash and tile store"`

---

### Task 3: preprocess and motion (with sidecar)

**Files:**
- Create: `src/fastminian/preprocess.py`, `src/fastminian/motion.py`, `tests/test_motion.py`

**Interfaces:**
- Produces (`preprocess.py`): `denoise(frames: np.ndarray, ksize: int = 7) -> np.ndarray`; `tophat(frames: np.ndarray, wnd: int = 15) -> np.ndarray` using `np.ones((2*wnd+1,)*2, np.uint8)` (square kernel, `$BENCH/bench_browser/candidates/preproc.py:tophat_rect`); `preprocess(frames, ksize, wnd) -> np.ndarray` = tophat(denoise(frames)).
- Produces (`motion.py`):
  - `@dataclass(frozen=True) Template(fft: np.ndarray, down: int, shape: tuple[int, int])` (`shape` = full-resolution h, w).
  - `estimate(frames: np.ndarray, tmpl: Template, max_shift: int = 20) -> np.ndarray` (n, 2) float64 — `$BENCH/bench_browser/candidates/motion.py:estimate` with `MAX_SHIFT` a parameter.
  - `make_template(pre_frames: np.ndarray, down: int = 2, max_shift: int = 20) -> Template` (the re-centring step included).
  - `apply_shifts(frames: np.ndarray, shifts: np.ndarray) -> np.ndarray` — `heavy_pass.apply_shifts` (works for uint8 and float32 input).
  - `frame_quality(frames: np.ndarray, tmpl: Template) -> np.ndarray` (n,) float32: Pearson correlation between each `_prep`-ed frame and `irfft2(tmpl.fft)`.
  - `template_image(tmpl: Template) -> np.ndarray` (h/down, w/down) float32 = `irfft2(tmpl.fft)`.
  - `@dataclass MotionSidecar(shifts: np.ndarray, template: np.ndarray, quality: np.ndarray, params: dict, raw_hash: str, meta: dict)`; `save_sidecar(path: Path, sc) -> None` (`meta.json` + `arrays.npz`), `load_sidecar(path) -> MotionSidecar`, `load_if_valid(path, raw_hash: str, params: dict, n_frames: int) -> MotionSidecar | None` — port of `motion_sidecar.py` (`FORMAT_VERSION = 1`, `canonical_params`), minus `warp_block`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from fastminian import motion, preprocess
from fastminian.testing import synth


def _aligned_rmse(est, true):
    best = 1e9
    for sign in (-1, 1):
        d = est - sign * true
        best = min(best, float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())))
    return best


def test_tophat_square_matches_cv2():
    import cv2
    f = synth.synth_movie(64, 64, 3, 0)
    out = preprocess.tophat(f, wnd=15)
    k = np.ones((31, 31), np.uint8)
    assert out.dtype == np.uint8
    assert np.array_equal(out[1], cv2.morphologyEx(f[1], cv2.MORPH_TOPHAT, k))


def test_motion_estimate_accuracy():
    h, w, T = 128, 128, 300
    movie = synth.synth_movie(h, w, T, 0)
    setup = synth.synth_setup(h, w, T, 0)
    pre = preprocess.preprocess(movie, 7, 15)
    tmpl = motion.make_template(pre[:200], down=2)
    sh = motion.estimate(pre, tmpl)
    assert sh.shape == (T, 2)
    assert _aligned_rmse(sh, setup.shifts.astype(float)) < 0.15
    q = motion.frame_quality(motion.apply_shifts(pre, sh), tmpl)
    assert q.shape == (T,) and q.min() > 0.5


def test_sidecar_roundtrip_and_validation(tmp_path):
    sc = motion.MotionSidecar(
        shifts=np.zeros((10, 2), np.float32), template=np.ones((8, 8), np.float32),
        quality=np.ones(10, np.float32), params={"max_shift": 20, "down": 2}, raw_hash="abc", meta={},
    )
    motion.save_sidecar(tmp_path / "motion", sc)
    back = motion.load_sidecar(tmp_path / "motion")
    assert np.array_equal(back.shifts, sc.shifts) and back.params == sc.params
    assert motion.load_if_valid(tmp_path / "motion", "abc", {"down": 2, "max_shift": 20}, 10) is not None
    assert motion.load_if_valid(tmp_path / "motion", "zzz", sc.params, 10) is None
    assert motion.load_if_valid(tmp_path / "motion", "abc", {"max_shift": 10, "down": 2}, 10) is None
    assert motion.load_if_valid(tmp_path / "motion", "abc", sc.params, 11) is None
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** from `candidates/motion.py` (`_down2`, `_hann`, `_prep`, `estimate`, `_mean_template`, `make_template`) and `heavy_pass.py` (`_parabolic`, `apply_shifts`). `frame_quality`: `f = _prep(frames, tmpl.down)`, `t = irfft2(tmpl.fft)`; Pearson over pixels per frame. Sidecar functions from `motion_sidecar.py` lines 95–248 (`canonical_params`, `MotionSidecar`, `save_sidecar`, `load_sidecar`, `validate`, `load_if_valid`); replace `_minian_version()` with `fastminian.__version__`.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add preprocessing, template motion estimate and motion sidecar"`

---

### Task 4: pass 1

**Files:**
- Create: `src/fastminian/pass1.py`, `src/fastminian/params.py`, `tests/test_pass1.py`

**Interfaces:**
- Produces (`params.py`): `@dataclass Params` with the exact fields and defaults of spec §6 (`median_ksize=7, tophat_wnd=15, motion_max_shift=20, motion_down=2, tile=64, svd_k0=40, svd_power_iter=1, rank_rho_min=0.2, rank_z=4.0, seeds_wnd=1000, seeds_stp=500, seeds_max_wnd=15, seeds_diff_thres=3, noise_freq=0.06, pnr_thres: float | str = "auto", merge_dist=10, merge_corr=0.8, inita_corr=0.8, inita_wnd=10, init_merge_corr=0.8, loop_merge_corr=0.9, dl_wnd=10, sparse_penal=0.01, size_thres=(25, None), n_iter=2, workers=1, template_frames=200, sub_batch=100`); `Params.from_json(path)`, `to_json(path)`, `motion_params() -> dict` (the subset that invalidates a sidecar: ksize, wnd, max_shift, down, template_frames), `asdict()`.
- Produces (`pass1.py`):
  - `seed_block_edges(T: int, wnd_size: int, stp_size: int) -> np.ndarray` — `$BENCH/bench_passes/candidates/seeds_blockmax.py:rolling_windows` + `block_edges` (sorted unique window starts/stops, always containing 0 and T).
  - `class Pass1(template: Template, grid: list[Tile], params: Params, block_edges: np.ndarray, t_start: int)`: `process(frames: np.ndarray) -> np.ndarray` packed uint8 (sub-batches of `params.sub_batch`; denoise → tophat → estimate → apply_shifts → running `max_proj` and per-block maxima for frames `t_start + i` → pack); attributes `shifts: list`, `quality: list`, `max_proj`, `block_max (n_blocks, H, W) uint8`, `seconds: dict`, `frames: int`.
  - `@dataclass Pass1Result(shifts, quality, max_proj, block_max, block_edges, frames, seconds)`.
  - `run_pass1(info: RawInfo, store: TileStore, template: Template, params: Params, workers: int) -> Pass1Result` — splits segments into `workers` contiguous ranges (`_split` from `run_session_local.py`), each worker a single-threaded process (`OMP_NUM_THREADS=1`, `cv2.setNumThreads(1)`) that gets its absolute `t_start` from `info.segment_frames`, writes part files, returns shifts/quality/max/block_max; the driver concatenates and `np.maximum.reduce`s.
  - `build_template(info: RawInfo, params: Params) -> Template` from the first `template_frames` frames of segment 0.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from fastminian import io, motion, pass1, preprocess, tiles
from fastminian.params import Params


def test_block_edges_cover_movie():
    e = pass1.seed_block_edges(300, wnd_size=100, stp_size=50)
    assert e[0] == 0 and e[-1] == 300 and np.all(np.diff(e) > 0)


def test_pass1_single_and_parallel_agree(tiny_session, tmp_path):
    info = io.probe(tiny_session["raw_dir"])
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    tmpl = pass1.build_template(info, p)
    grid = tiles.tile_grid(64, 64, 32)
    outs = []
    for workers in (1, 2):
        store = tiles.TileStore(tmp_path / f"s{workers}", grid, n_parts=workers)
        outs.append((store, pass1.run_pass1(info, store, tmpl, p, workers)))
    (s1, r1), (s2, r2) = outs
    assert r1.frames == r2.frames == 300 and r1.shifts.shape == (300, 2)
    assert np.allclose(r1.shifts, r2.shifts) and np.array_equal(r1.max_proj, r2.max_proj)
    assert np.array_equal(r1.block_max, r2.block_max)
    for t in grid:
        assert np.array_equal(s1.read_tile(t.index), s2.read_tile(t.index))
    # block maxima equal the max over the same corrected frames
    full = np.concatenate([s1.read_tile(t.index) for t in grid[:1]])
    e = r1.block_edges
    assert np.array_equal(r1.block_max[0, :32, :32], full[e[0] : e[1]].max(0))


def test_pass1_ragged_last_segment(tmp_path):
    from fastminian.testing import synth
    raw = tmp_path / "raw"
    synth.write_session(raw, minutes=250 / 30 / 60, fps=30, segment=100, seed=0, side=64)  # 100,100,50
    info = io.probe(raw)
    assert info.segment_frames == [100, 100, 50]
    p = Params(seeds_wnd=100, seeds_stp=50, template_frames=50)
    grid = tiles.tile_grid(64, 64, 32)
    store = tiles.TileStore(tmp_path / "s", grid, n_parts=2)
    r = pass1.run_pass1(info, store, pass1.build_template(info, p), p, workers=2)
    assert r.frames == 250 and r.block_edges[-1] == 250
    assert store.read_tile(0).shape[0] == 250
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** `Pass1.process` is `base.Pass1Variant.process` with `pre.denoise/tophat/estimate` replaced by `preprocess.denoise`, `preprocess.tophat`, `motion.estimate`, plus after the warp:
```python
blk_t0 = self.t_start + self.frames + b0
for bi in range(np.searchsorted(self.block_edges, blk_t0, "right") - 1, len(self.block_edges) - 1):
    lo, hi = self.block_edges[bi] - blk_t0, self.block_edges[bi + 1] - blk_t0
    if lo >= blk.shape[0]: break
    part = blk[max(lo, 0) : min(hi, blk.shape[0])]
    if part.size: np.maximum(self.block_max[bi], part.max(axis=0), out=self.block_max[bi])
```
and `self.quality.append(motion.frame_quality(blk, self.template))`. The worker function must be a module-level function (picklable) and set the thread env vars before importing numpy in the child (use `multiprocessing.get_context("spawn")` and an `initializer` that sets `cv2.setNumThreads(1)`; set `OMP_NUM_THREADS=1` etc. in `os.environ` of the parent before creating the pool, restoring afterwards).
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add pass 1: preprocessing, motion, block maxima and tile store writer"`

---

### Task 5: compression, factors container, pass 2

**Files:**
- Create: `src/fastminian/compress.py`, `src/fastminian/factors.py`, `src/fastminian/pass2.py`, `tests/test_compress.py`

**Interfaces:**
- Produces (`factors.py`):
  - `@dataclass TileFactors(W: np.ndarray, V: np.ndarray, mean: np.ndarray, sigma: np.ndarray, energy: np.ndarray)` with `rank` property (`W.shape[1]`), `P`, `T`.
  - `@dataclass Factors(H: int, W_: int, T: int, grid: list[Tile], tiles: list[TileFactors], max_proj: np.ndarray, block_max: np.ndarray, block_edges: np.ndarray, meta: dict)` with:
    - `save(dir: Path) -> None` / `classmethod load(dir: Path) -> Factors` (layout of spec §3: `meta.json`, `tile_NNN.npz`, `max_proj.npy`, `block_max.npy`, `block_edges.npy`).
    - `P` property (`H*W_`), `tile_of: np.ndarray (P,) int32`, `loc_of: np.ndarray (P,) int32` (built in `__post_init__`), `pixel_index(h, w) -> int`.
    - `reconstruct_chunk(b: int, t0: int, t1: int) -> np.ndarray` (P_b, t1−t0) float32 = `W V[t0:t1]ᵀ + mean[:, None]`.
    - `traces(pixels: np.ndarray) -> np.ndarray` (n, T) float32, denoised incl. mean.
    - `sigma_flat() -> np.ndarray (P,)`, `mean_flat()`, `energy_flat()`.
    - `AtY(A: scipy.sparse.csr_matrix) -> np.ndarray` (K, T) float64 = Σ_b (A[:, rows_b] W_b) V_bᵀ + (A[:, rows_b] mean_b) 1ᵀ. (`rows_b = flat pixel indices of tile b`, provided by `tile_rows(b) -> np.ndarray`.)
    - `YCt(Cc: np.ndarray) -> np.ndarray` (P, K) float32 = per tile `W_b (V_bᵀ Ccᵀ)` (used by the spatial update; `Cc` rows must sum to zero, caller's responsibility).
- Produces (`compress.py`): `compress_tile(raw: np.ndarray, k0: int = 40, n_iter: int = 1, rho_min: float = 0.2, z: float = 4.0, seed: int = 0, chunk: int = 4096) -> TileFactors` — `$BENCH/bench_browser/candidates/rank.py:TwoPassRank.compress_tile` with `keep="ac"`, `adaptive=True` (uses `session.TileMatrix`, `base.rsvd_streamed`, `base.autocorr_keep`, `base.factors`, `ADAPT_MARGIN = 10`, `N_OVERSAMPLES = 10`, `MAX_RANK = 100`); additionally accumulates `energy = Σ_t (y − mean)²` in float64 inside `TileMatrix.__init__` (second pass over chunks). `lag1_autocorr(Vt) -> np.ndarray`, `autocorr_keep(Vt, z, rho_min) -> np.ndarray` exported.
- Produces (`pass2.py`): `run_pass2(store: TileStore, grid: list[Tile], params: Params, workers: int) -> list[TileFactors]` (workers over tile ranges, single-threaded BLAS per worker, `seed = tile.index`); `build_factors(info, grid, p1: Pass1Result, tiles: list[TileFactors], params) -> Factors`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import scipy.sparse as sps
from fastminian import compress, factors as fm, tiles
from fastminian.testing import synth


def _tile_movie(T=400, side=32, seed=3):
    m = synth.synth_movie(side, side, T, seed)  # moving; fine for a compression test
    return m


def test_compress_tile_keeps_signal_and_bounds_rank():
    raw = _tile_movie()
    tf = compress.compress_tile(raw, k0=8, n_iter=1, seed=0)
    T, P = raw.shape[0], 32 * 32
    assert tf.W.shape == (P, tf.rank) and tf.V.shape == (T, tf.rank) and 1 <= tf.rank <= 100
    assert tf.energy.shape == (P,) and tf.energy.dtype == np.float64
    Y = raw.reshape(T, -1).T.astype(np.float32)
    rec = tf.W @ tf.V.T + tf.mean[:, None]
    assert np.linalg.norm(Y - rec) / np.linalg.norm(Y - Y.mean(1, keepdims=True)) < 0.8
    assert np.allclose(tf.energy, ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1), rtol=1e-4)


def test_compress_tile_pure_noise_returns_rank_one():
    rng = np.random.default_rng(0)
    raw = np.clip(50 + 4 * rng.standard_normal((300, 16, 16)), 0, 255).astype(np.uint8)
    tf = compress.compress_tile(raw, k0=8)
    assert tf.rank == 1


def test_autocorr_keep():
    T = 1000
    rng = np.random.default_rng(0)
    smooth = np.cumsum(rng.standard_normal(T)); white = rng.standard_normal(T)
    keep = compress.autocorr_keep(np.stack([smooth, white]), z=4.0, rho_min=0.2)
    assert keep.tolist() == [True, False]


def test_factors_save_load_and_algebra(tmp_path):
    H = W = 32; T = 200
    grid = tiles.tile_grid(H, W, 16)
    movie = synth.synth_movie(H, W, T, 1)
    tf = [compress.compress_tile(np.ascontiguousarray(movie[:, t.y0:t.y1, t.x0:t.x1]), k0=8) for t in grid]
    f = fm.Factors(H, W, T, grid, tf, movie.max(0), movie[:100].max(0)[None], np.array([0, 100, 200]), {})
    f.save(tmp_path / "factors"); g = fm.Factors.load(tmp_path / "factors")
    assert g.T == T and len(g.tiles) == 4 and np.array_equal(g.tiles[2].V, tf[2].V)
    # dense reference
    Yd = np.concatenate([(t.W @ t.V.T + t.mean[:, None]) for t in tf])  # tile-major rows
    perm = np.concatenate([g.tile_rows(b) for b in range(4)])
    Yfull = np.empty((H * W, T), np.float32); Yfull[perm] = Yd
    A = sps.random(5, H * W, density=0.05, format="csr", random_state=0, dtype=np.float32)
    assert np.allclose(g.AtY(A), A @ Yfull, rtol=1e-4, atol=1e-2)
    C = np.random.default_rng(0).standard_normal((5, T)).astype(np.float32); Cc = C - C.mean(1, keepdims=True)
    assert np.allclose(g.YCt(Cc), Yfull @ Cc.T, rtol=1e-3, atol=1e-1)
    px = np.array([0, 17, 1000])
    assert np.allclose(g.traces(px), Yfull[px], atol=1e-3)
    assert np.allclose(g.reconstruct_chunk(1, 10, 20), Yd[256:512, 10:20], atol=1e-3)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** Copy `TileMatrix` from `session.py` (add the `energy` accumulation), `rsvd_streamed` / `rsvd_flops` / `lag1_autocorr` / `autocorr_keep` / `factors` from `candidates/base.py`, the adaptive loop from `rank.py:TwoPassRank.compress_tile`. `_qr` and `gemm_blas` from `heavy_pass.py`. `Factors.tile_of/loc_of`: fill from `grid` once. `AtY`: for each tile `Ab = A[:, rows_b]` (CSC column slice of the CSR → convert `A` to CSC once), `(Ab @ W_b) @ V_bᵀ + (Ab @ mean_b)[:, None]`, accumulate float64. `pass2.run_pass2` is `bench_opt._p2_worker` + `run_parallel`'s pool over `_split(len(grid), workers)`.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add streamed adaptive rSVD compression, factors container and pass 2"`

---

### Task 6: seeds, PNR and GMM on factors

**Files:**
- Create: `src/fastminian/seeds.py`, `tests/test_seeds.py`

**Interfaces:**
- Produces:
  - `seeds_init(block_max: np.ndarray, block_edges: np.ndarray, T: int, wnd_size: int, stp_size: int, max_wnd: int, diff_thres: int) -> pd.DataFrame` with columns `height, width, seeds` (count of windows in which the pixel was a local max) — `seeds_blockmax.py`: `rolling_windows`, `window_block_ranges`, `window_projection`, `disk_kernels`, `local_max_union`, `label_medians`, `count_local_maxima`, `seeds_from_count` (median centre only, no env knobs).
  - `ptp_q(X: np.ndarray, q: tuple[float, float]) -> np.ndarray` row-wise.
  - `pnr_refine(f: Factors, seeds: pd.DataFrame, thres: float | str = "auto", q=(0.1, 99.9)) -> pd.DataFrame` adds `pnr` and `mask_pnr`; `pnr = ptp_q(trace) / (6.18 * sigma_p)` with `trace = f.traces(pixel)`; `thres="auto"` → `GaussianMixture(2)` on `log(pnr)`, keep the component with the larger mean (`mask = predict == argmax(means)`); float → `pnr > thres`.
  - `gmm_refine(f: Factors, seeds: pd.DataFrame, q=(0.1, 99.9), n_components=2, valid_components=1, mean_mask=True) -> pd.DataFrame` adds `mask_gmm` — same logic as `$MINIAN/initialization.py:gmm_refine` lines 179–268 on `ptp_q(f.traces(pixels))`.
  - `seed_pixels(f: Factors, seeds: pd.DataFrame) -> np.ndarray` flat pixel indices.
  - Raise `ValueError("no seeds survived <stage>")` when a mask is all False.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
from fastminian import compress, factors as fm, seeds as sd, tiles
from fastminian.testing import synth


def _session(H=64, W=64, T=600, seed=0):
    movie = synth.synth_movie(H, W, T, seed)
    grid = tiles.tile_grid(H, W, 32)
    tf = [compress.compress_tile(np.ascontiguousarray(movie[:, t.y0:t.y1, t.x0:t.x1]), k0=16) for t in grid]
    from fastminian.pass1 import seed_block_edges
    edges = seed_block_edges(T, 300, 150)  # the same edges pass 1 would store
    bm = np.stack([movie[a:b].max(0) for a, b in zip(edges[:-1], edges[1:])])
    return movie, fm.Factors(H, W, T, grid, tf, movie.max(0), bm, edges, {})


def test_seeds_from_block_max_equal_full_projection():
    movie, f = _session()
    s = sd.seeds_init(f.block_max, f.block_edges, f.T, wnd_size=300, stp_size=150, max_wnd=10, diff_thres=2)
    # reference: same windows from the raw movie
    from fastminian.seeds import rolling_windows, local_max_roll
    wins = rolling_windows(f.T, 300, 150)
    ref = sum(local_max_roll(movie[w.start:w.stop].max(0), 2, 10, 2) for w in wins)
    hh, ww = np.nonzero(ref)
    assert set(zip(s.height, s.width)) == set(zip(hh, ww))
    assert len(s) > 5


def test_pnr_and_gmm_masks(tmp_path):
    movie, f = _session()
    setup = synth.synth_setup(64, 64, 600, 0)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 0).all(1) & (cents < 64).all(1)]
    rng = np.random.default_rng(0)
    noise = np.stack([rng.integers(0, 64, 30), rng.integers(0, 64, 30)], 1)
    seeds = pd.DataFrame({"height": np.r_[cents[:, 0], noise[:, 0]], "width": np.r_[cents[:, 1], noise[:, 1]]})
    out = sd.pnr_refine(f, seeds, thres="auto")
    assert out["mask_pnr"].iloc[: len(cents)].mean() > 0.8
    out2 = sd.pnr_refine(f, seeds, thres=2.0)
    assert out2["pnr"].iloc[: len(cents)].median() > out2["pnr"].iloc[len(cents):].median()
    g = sd.gmm_refine(f, seeds)
    assert g["mask_gmm"].iloc[: len(cents)].mean() > 0.8


def test_pnr_raises_when_nothing_survives():
    _, f = _session()
    seeds = pd.DataFrame({"height": [1, 2], "width": [1, 2]})
    import pytest
    with pytest.raises(ValueError, match="no seeds"):
        sd.pnr_refine(f, seeds, thres=1e9)
```
(Expose `rolling_windows(T, wnd_size, stp_size) -> list[Window]` with `start, stop` and `local_max_roll(fm, k0, k1, diff) -> np.ndarray` as public helpers; the reference in the first test uses them on raw frames.)

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** from `seeds_blockmax.py` (functions named above; window projections come from `block_max` via `window_block_ranges` instead of a movie pass) and `$MINIAN/initialization.py` for the GMM logic.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add seeds from block maxima with PNR and GMM filters on factors"`

---

### Task 7: correlations on factors

**Files:**
- Create: `src/fastminian/corr.py`, `tests/test_corr.py`

**Interfaces:**
- Produces:
  - `lowpass(X: np.ndarray, noise_freq: float) -> np.ndarray` along the last axis: rfft, zero bin 0 and bins ≥ `int(noise_freq * T)`, irfft(T) (float64).
  - `class FactorCorr(f: Factors, noise_freq: float)`: precomputes `Vlp[b] = lowpass(V_bᵀ).T` (T, r_b) float64, `Gaa[b]`, `norm2 (P,)` in "resid" mode (`compressed_svd.py:correlate`: `norm2 = w G_aa wᵀ + f_lp * resid`, `resid = energy − ||w||²` clipped at 0, `f_lp = 2(nr−1)/T`). Methods:
    - `pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray` flat pixel indices → correlations in [−1, 1] (grouped by tile pair, `G_ab` computed on demand and cached in a dict).
    - `window(seed: int, wnd: int) -> tuple[np.ndarray, np.ndarray]`: flat indices of all pixels within Euclidean distance `wnd` of the seed (disk, inside the frame) and their correlation to the seed.
    - `traces_corr(C: np.ndarray) -> np.ndarray` (K, K) Pearson of `lowpass(C)` rows (for unit merge).
  - `seeds_merge(fc: FactorCorr, seeds: pd.DataFrame, max_proj: np.ndarray, thres_dist: int, thres_corr: float) -> pd.DataFrame` adds `mask_mrg`: KD-tree pairs within `thres_dist`, `pairs()` corr > `thres_corr` → `scipy.sparse.csgraph.connected_components`; per component keep the seed with the largest `max_proj[h, w]` (ties: first).
  - `merge_labels(adj: scipy.sparse.spmatrix) -> np.ndarray` connected-component labels (shared with `init.unit_merge`).

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
from fastminian import compress, corr, factors as fm, tiles
from fastminian.testing import synth


def _session(H=64, W=64, T=800, seed=0):
    movie = synth.synth_movie(H, W, T, seed)
    setup = synth.synth_setup(H, W, T, seed)
    grid = tiles.tile_grid(H, W, 32)
    tf = [compress.compress_tile(np.ascontiguousarray(movie[:, t.y0:t.y1, t.x0:t.x1]), k0=16) for t in grid]
    f = fm.Factors(H, W, T, grid, tf, movie.max(0), movie.max(0)[None], np.array([0, T]), {})
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, H, W)])
    return movie, clean, setup, f


def test_lowpass_removes_dc_and_high_band():
    T = 500; x = np.ones((2, T)); x[1] = np.sin(np.arange(T) * 2.5)
    y = corr.lowpass(x, 0.06)
    assert abs(y).max() < 1e-6


def test_pair_correlations_track_noise_free_truth():
    movie, clean, setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 2).all(1) & (cents < 62).all(1)][:40]
    src = cents[:, 0] * 64 + cents[:, 1]
    dst = (cents[:, 0] + 1) * 64 + cents[:, 1]  # neighbour pixel of the same cell: high corr
    far = ((cents[:, 0] + 20) % 64) * 64 + ((cents[:, 1] + 20) % 64)
    c_near, c_far = fc.pairs(src, dst), fc.pairs(src, far)
    Y = clean.reshape(clean.shape[0], -1).T.astype(float)
    lp = corr.lowpass(Y[np.r_[src, dst, far]], 0.06)
    n = len(src)
    def pear(a, b): return np.sum(a * b, 1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ref_near, ref_far = pear(lp[:n], lp[n:2 * n]), pear(lp[:n], lp[2 * n:])
    assert np.median(np.abs(c_near - ref_near)) < 0.05
    assert np.median(c_near) > 0.8 and np.median(np.abs(c_far)) < 0.3
    idx, cw = fc.window(int(src[0]), 10)
    assert cw.max() <= 1.0 and idx.size <= 317 and (cw[idx == src[0]] > 0.99).all()


def test_seeds_merge_keeps_brightest_per_group():
    movie, clean, setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 3).all(1) & (cents < 61).all(1)][:10]
    dup = cents + np.array([1, 0])
    s = pd.DataFrame({"height": np.r_[cents[:, 0], dup[:, 0]], "width": np.r_[cents[:, 1], dup[:, 1]]})
    out = corr.seeds_merge(fc, s, f.max_proj, thres_dist=5, thres_corr=0.6)
    assert out["mask_mrg"].sum() <= len(cents) + 1
    kept = out[out["mask_mrg"]]
    # every kept seed is at least as bright as any seed it absorbed within thres_dist
    for h, w in zip(kept.height, kept.width):
        near = (abs(s.height - h) <= 5) & (abs(s.width - w) <= 5)
        assert f.max_proj[h, w] >= f.max_proj[s.height[near], s.width[near]].max()
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** from `$BENCH/bench_corr/candidates/compressed_svd.py` (`lowpass_factors`, `correlate` with `NORM_MODE="resid"`); `window` builds the disk offsets once per `wnd` (`np.mgrid`, `dy²+dx² ≤ wnd²`), clips to the frame, and calls the same tile-pair machinery with `src` repeated. `seeds_merge` and `traces_corr` are plain scipy/numpy.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add band-limited correlations on factors, seed merge"`

---

### Task 8: initA, initC, unit_merge

**Files:**
- Create: `src/fastminian/init.py`, `tests/test_init.py`

**Interfaces:**
- Produces:
  - `initA(fc: FactorCorr, seeds: pd.DataFrame, thres_corr: float, wnd: int) -> scipy.sparse.csr_matrix` (K, P) float32: row k = `window(seed_k, wnd)` correlations with values < `thres_corr` zeroed; rows with no nonzero are dropped; raises `ValueError("initA produced no footprints")` if K = 0.
  - `initC(f: Factors, A: csr_matrix) -> np.ndarray` (K, T) float32 = solve `(AᵀA) C = AᵀY` by Cholesky, `pinv` fallback when `cond > 1e12` (`$BENCH/bench_passes/candidates/initc_normal.py:gram, make_solver`). `AᵀY` from `f.AtY(A)`.
  - `unit_merge(A: csr_matrix, C: np.ndarray, fc_traces: Callable[[np.ndarray], np.ndarray], thres_corr: float, extra: list[np.ndarray] | None = None) -> tuple[csr_matrix, np.ndarray, list[np.ndarray]]`: adjacency = `(A @ Aᵀ > 0)` off-diagonal AND `fc_traces(C) > thres_corr`; `merge_labels`; merged A and C (and each `extra` array, (K, ·)) are the **mean** over each group (Minian `groupby.mean`), group order = first appearance.
  - `centroids(A: csr_matrix, H: int, W: int) -> np.ndarray` (K, 2) intensity-weighted.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
import scipy.sparse as sps
from fastminian import compress, corr, factors as fm, init, tiles
from fastminian.testing import synth


def _session(H=64, W=64, T=800, seed=0):
    movie = synth.synth_movie(H, W, T, seed); setup = synth.synth_setup(H, W, T, seed)
    grid = tiles.tile_grid(H, W, 32)
    tf = [compress.compress_tile(np.ascontiguousarray(movie[:, t.y0:t.y1, t.x0:t.x1]), k0=16) for t in grid]
    return setup, fm.Factors(H, W, T, grid, tf, movie.max(0), movie.max(0)[None], np.array([0, T]), {})


def test_initA_footprints_cover_cells_and_initC_is_least_squares():
    setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int)
    cents = cents[(cents >= 5).all(1) & (cents < 59).all(1)]
    seeds = pd.DataFrame({"height": cents[:, 0], "width": cents[:, 1]})
    A = init.initA(fc, seeds, thres_corr=0.6, wnd=10)
    assert A.shape == (len(cents), 64 * 64)
    sizes = np.diff(A.indptr)
    assert sizes.min() >= 9 and sizes.max() <= 317
    C = init.initC(f, A)
    # reference: dense lstsq on the reconstructed movie
    Y = np.concatenate([f.reconstruct_chunk(b, 0, f.T) for b in range(4)])
    perm = np.concatenate([f.tile_rows(b) for b in range(4)]); Yf = np.empty_like(Y); Yf[perm] = Y
    ref = np.linalg.lstsq(A.toarray().T, Yf, rcond=None)[0]
    assert np.allclose(C, ref, atol=1e-3 * abs(ref).max())
    # traces correlate with truth
    tr = setup.C[(setup.ys + 7 - 8 >= 5) & (setup.ys + 7 - 8 < 59) & (setup.xs + 7 - 8 >= 5) & (setup.xs + 7 - 8 < 59)]
    r = [np.corrcoef(C[k], tr[k])[0, 1] for k in range(len(tr))]
    assert np.median(r) > 0.8


def test_unit_merge_means_duplicates():
    A = sps.csr_matrix(np.array([[1, 1, 0, 0], [1, 0.5, 0, 0], [0, 0, 1, 0]], np.float32))
    C = np.array([[1, 2, 3, 4], [1.1, 2, 3, 3.9], [4, 3, 2, 1]], np.float32)
    A2, C2, extra = init.unit_merge(A, C, lambda X: np.corrcoef(X), thres_corr=0.9, extra=[C * 2])
    assert A2.shape == (2, 4) and np.allclose(A2.toarray()[0], [1, 0.75, 0, 0])
    assert np.allclose(C2[0], (C[0] + C[1]) / 2) and np.allclose(extra[0][1], C[2] * 2)


def test_initA_raises_without_footprints():
    import pytest
    setup, f = _session()
    fc = corr.FactorCorr(f, 0.06)
    with pytest.raises(ValueError, match="no footprints"):
        init.initA(fc, pd.DataFrame({"height": [0], "width": [0]}), thres_corr=1.01, wnd=3)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** `initA`: loop seeds, collect `(k, idx, val)` where `val ≥ thres_corr`, build COO → CSR. `initC`: `G = (A @ A.T).toarray()`, `cho_factor` / `cho_solve`, fallback `np.linalg.pinv(G) @ AtY`. `unit_merge`: `adj = ((A @ A.T) > 0).toarray()`; `np.fill_diagonal(adj, False)`; `adj &= fc_traces(C) > thres_corr`; labels via `corr.merge_labels`; group means with `np.add.at` / sparse indicator matrix `M (K', K)` with entries `1/|group|` → `A2 = M @ A`, `C2 = M @ C`.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add initA, Cholesky initC and unit merge on factors"`

---

### Task 9: spatial update kernel and CNMF stages

**Files:**
- Create: `src/fastminian/spatial_kernel.py`, `src/fastminian/cnmf.py`, `tests/test_spatial_kernel.py`, `tests/test_cnmf.py`

**Interfaces:**
- Produces (`spatial_kernel.py`): `solve_nnqp(G: np.ndarray, B: np.ndarray, lam: np.ndarray, indptr: np.ndarray, indices: np.ndarray, rel_gap_tol: float = 1e-10, max_sweeps: int = 100_000) -> np.ndarray` — for active pixel i with allowed units `indices[indptr[i]:indptr[i+1]]`, minimise `½ wᵀGw − b_iᵀw + lam_i Σw`, `w ≥ 0`; returns `values` aligned with `indices` (so `(values, indices, indptr)` is the CSR of `A_newᵀ` over active pixels). Numba `@njit(parallel=True, nogil=True)`, duality-gap stop — `$BENCH/bench_spatial/candidates/numba_cd.py` kernel (`_solve_pixels` or equivalent; keep its gap bound).
- Produces (`cnmf.py`):
  - `dilate_footprints(A: csr_matrix, H: int, W: int, dl_wnd: int) -> csr_matrix` bool (K, P): per unit `cv2.dilate(A_k.reshape(H, W) > 0, disk(dl_wnd))` (`skimage.morphology.disk` replaced by an explicit `np.mgrid` disk).
  - `update_spatial(f: Factors, A: csr_matrix, C: np.ndarray, sn: np.ndarray, dl_wnd: int, sparse_penal: float, size_thres: tuple[int | None, int | None]) -> tuple[csr_matrix, np.ndarray, np.ndarray]` → `(A_new (K', P) float32 normalised by max, mask (K,) bool, norm_fac (K',))`. Steps: `sub = dilate_footprints`; active pixels = columns of `sub` with any unit; `Cc = C − mean`, `G = Cc Ccᵀ` float64, `B = f.YCt(Cc)[active]` float64, `lam = T * sparse_penal * sn[active]`; `solve_nnqp`; assemble `A_new`; size filter `low < nnz < high` (strict, as Minian); `norm_fac = A_new.max(axis=1)`; divide rows.
  - `update_background(f: Factors, A: csr_matrix, C: np.ndarray, b: np.ndarray | None = None, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]` → `(b (P,) float32, f (T,) float32)`; streamed over tiles × frame chunks: `R = clip(reconstruct_chunk − A_bᵀ C_c, 0)`; pass 1 `b = mean_t R`; pass 2 `f_t = R_tᵀ b / (b·b)` (`bg_streamed.py`).
  - `compute_trace(f: Factors, A: csr_matrix, b: np.ndarray, C: np.ndarray, fb: np.ndarray) -> np.ndarray` YrA (K, T) float32 = `clip(C + A_norm (YtA − (A Aᵀ) C), 0)` with `YtA = f.AtY(A) − (A b) fbᵀ`, `A_norm = diag(1/Σ a_k²)`.
  - `@dataclass CnmfResult(A, C, YrA, b, f, counts: dict)`; `run_cnmf(fac: Factors, A: csr_matrix, C: np.ndarray, params: Params, fc_traces) -> CnmfResult` implementing the spec §5 loop: `b, f = update_background(A, C)`; for `it in range(n_iter)`: `A, mask, nf = update_spatial(...)`; `C = C[mask] * nf[:, None]`; `b, f = update_background(A, C, b=b)`; `YrA = compute_trace(...)`; `C = YrA`; `A, C, _ = init.unit_merge(A, C, fc_traces, params.loop_merge_corr)`; record counts per stage. Raise `ValueError("all units dropped by size_thres")` when `mask.sum() == 0`.

- [ ] **Step 1: Failing tests**

`tests/test_spatial_kernel.py`:
```python
import numpy as np
from scipy.optimize import nnls
from fastminian.spatial_kernel import solve_nnqp


def test_nnqp_matches_nnls_per_pixel():
    rng = np.random.default_rng(0)
    K, T, P = 6, 300, 50
    C = np.abs(rng.standard_normal((K, T))); Cc = C - C.mean(1, keepdims=True)
    G = Cc @ Cc.T
    Wtrue = np.abs(rng.standard_normal((P, K))) * (rng.random((P, K)) < 0.5)
    Y = Wtrue @ C + 0.1 * rng.standard_normal((P, T))
    B = Y @ Cc.T
    lam = np.full(P, 0.5)
    sub = (Wtrue > 0) | (rng.random((P, K)) < 0.2)
    sub[0] = True
    indptr = np.r_[0, np.cumsum(sub.sum(1))]; indices = np.concatenate([np.flatnonzero(s) for s in sub])
    vals = solve_nnqp(G, B, lam, indptr.astype(np.int64), indices.astype(np.int64))
    for i in range(P):
        idx = indices[indptr[i]:indptr[i + 1]]
        # NNLS of the equivalent problem: ||L w - (L^-T)(b - lam)||, L = chol(G[idx][:, idx])
        Gs = G[np.ix_(idx, idx)]; L = np.linalg.cholesky(Gs)
        rhs = np.linalg.solve(L, B[i, idx] - lam[i])
        w_ref, _ = nnls(L.T, rhs)
        w = vals[indptr[i]:indptr[i + 1]]
        f = lambda x: 0.5 * x @ Gs @ x - B[i, idx] @ x + lam[i] * x.sum()
        assert f(w) <= f(w_ref) + 1e-6 * max(1.0, abs(f(w_ref)))
        assert (w >= 0).all()
```
`tests/test_cnmf.py`:
```python
import numpy as np
import scipy.sparse as sps
from fastminian import cnmf, compress, corr, factors as fm, init, tiles
from fastminian.params import Params
from fastminian.testing import synth


def _dense(f):
    Y = np.concatenate([f.reconstruct_chunk(b, 0, f.T) for b in range(len(f.tiles))])
    perm = np.concatenate([f.tile_rows(b) for b in range(len(f.tiles))])
    out = np.empty_like(Y); out[perm] = Y; return out.astype(np.float64)


def _setup(H=48, W=48, T=600):
    movie = synth.synth_movie(H, W, T, 2); setup = synth.synth_setup(H, W, T, 2)
    grid = tiles.tile_grid(H, W, 24)
    tf = [compress.compress_tile(np.ascontiguousarray(movie[:, t.y0:t.y1, t.x0:t.x1]), k0=16) for t in grid]
    f = fm.Factors(H, W, T, grid, tf, movie.max(0), movie.max(0)[None], np.array([0, T]), {})
    fc = corr.FactorCorr(f, 0.06)
    cents = np.rint(synth.true_centroids(setup)).astype(int); cents = cents[(cents >= 4).all(1) & (cents < H - 4).all(1)]
    import pandas as pd
    A = init.initA(fc, pd.DataFrame({"height": cents[:, 0], "width": cents[:, 1]}), 0.6, 8)
    C = init.initC(f, A)
    return f, fc, A, C


def test_background_and_trace_match_dense_reference():
    f, fc, A, C = _setup()
    Y = _dense(f); Ad = A.toarray().astype(np.float64)
    R = np.clip(Y - Ad.T @ C, 0)
    b_ref = R.mean(1); f_ref = R.T @ b_ref / (b_ref @ b_ref)
    b, fb = cnmf.update_background(f, A, C, chunk=128)
    assert np.allclose(b, b_ref, atol=1e-4 * b_ref.max()) and np.allclose(fb, f_ref, rtol=1e-4)
    b2, fb2 = cnmf.update_background(f, A, C, b=b)
    assert np.array_equal(b2, b) and np.allclose(fb2, fb)
    YtA = (Y - np.outer(b_ref, f_ref)).T @ Ad.T
    An = 1.0 / (Ad ** 2).sum(1)
    ref = np.clip(C + ((YtA - C.T @ (Ad @ Ad.T)) * An).T, 0)
    assert np.allclose(cnmf.compute_trace(f, A, b, C, fb), ref, atol=1e-3 * abs(ref).max())


def test_update_spatial_shrinks_outside_cells_and_keeps_units():
    f, fc, A, C = _setup()
    sn = f.sigma_flat()
    A2, mask, nf = cnmf.update_spatial(f, A, C, sn, dl_wnd=5, sparse_penal=0.01, size_thres=(9, None))
    assert mask.sum() >= 0.8 * A.shape[0] and A2.shape[1] == f.P
    assert np.allclose(A2.max(axis=1).toarray().ravel(), 1.0)
    assert (A2.data >= 0).all()


def test_run_cnmf_two_iterations():
    f, fc, A, C = _setup()
    p = Params(dl_wnd=5, size_thres=(9, None), n_iter=2)
    res = cnmf.run_cnmf(f, A, C, p, fc.traces_corr)
    assert res.A.shape[0] == res.C.shape[0] == res.YrA.shape[0] >= 0.7 * A.shape[0]
    assert res.b.shape == (f.P,) and res.f.shape == (f.T,) and res.counts["iter"] == 2
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** Kernel: copy the `@njit(parallel=True)` pixel loop from `numba_cd.py` (cyclic CD over allowed units, gap check; `G` float64, `B` float64, `lam` float64). `update_background`: per tile b, `Ab = A[:, rows_b]` (CSC slice, dense (K, P_b) float32 is fine: K·P_b ≤ 1000·4096), loop `t0` in steps of `chunk`: `R = f.reconstruct_chunk(b, t0, t1) − Abᵀ @ C[:, t0:t1]`, `np.clip(R, 0, None, out=R)`, accumulate. `compute_trace`: `AtY = f.AtY(A)`, `Ab = A @ b`, `YtA = (AtY − np.outer(Ab, fb)).T`.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add numba spatial solver, background, traces and the CNMF loop on factors"`

---

### Task 10: pipeline, report and CLI

**Files:**
- Create: `src/fastminian/pipeline.py`, `src/fastminian/report.py`, `tests/test_pipeline.py`; Modify: `src/fastminian/cli.py`

**Interfaces:**
- Produces:
  - `pipeline.run(raw_dir: Path, params: Params, run_name: str = "fastminian", out_root: Path | None = None, keep_store: bool = False) -> Path` executing: probe → hash → template → sidecar check → pass 1 (or apply-only from sidecar) → sidecar save → pass 2 → `Factors.save` → store removal → seeds_init → pnr_refine → gmm_refine → seeds_merge → initA → initC → unit_merge(init_merge_corr) → run_cnmf → write `cnmf/` files, `units.csv`, `params.json`, `report.json`. Output dir = `out_root or raw_dir.parent / "processed"` / `run_name`; it is removed and recreated at start (Review Focus 5). Window-2 stages write their wall time and counts into `report.json` through `report.StageTimer`.
  - `report.StageTimer`: `with timer("seeds_init"): ...`; `.rows` list of `{"stage", "seconds"}`; `.counts` dict; `to_json(path, extra: dict)`.
  - CLI: `fastminian run RAW_DIR [--params p.json] [--workers N] [--name NAME] [--keep-store]` and the existing `synth`.
  - Apply-only path when a valid sidecar exists: pass 1 still runs its warp and store write but skips `estimate` (uses stored shifts per frame index) and `make_template`.

- [ ] **Step 1: Failing test**

```python
import json
import numpy as np
import scipy.sparse as sps
from fastminian import pipeline
from fastminian.params import Params
from fastminian.testing import synth


def test_pipeline_end_to_end_tiny(tiny_session):
    p = Params(tile=32, seeds_wnd=100, seeds_stp=50, seeds_max_wnd=8, template_frames=50,
               inita_wnd=8, dl_wnd=5, size_thres=(9, None), svd_k0=16)
    out = pipeline.run(tiny_session["raw_dir"], p, run_name="t", workers=2)
    assert (out / "factors" / "meta.json").exists() and not (out / "tiles").exists()
    A = sps.load_npz(out / "cnmf" / "A.npz"); C = np.load(out / "cnmf" / "C.npy")
    rep = json.loads((out / "report.json").read_text())
    assert A.shape[1] == 64 * 64 and C.shape == (A.shape[0], 300)
    assert rep["counts"]["units_final"] == A.shape[0]
    cents = synth.true_centroids(tiny_session["setup"])
    cents = cents[(cents >= 4).all(1) & (cents < 60).all(1)]
    from fastminian.init import centroids
    uc = centroids(A, 64, 64)
    d = np.sqrt(((cents[:, None] - uc[None]) ** 2).sum(-1)).min(1)
    assert (d < 5).mean() >= 0.8
    # second run reuses the sidecar and overwrites
    out2 = pipeline.run(tiny_session["raw_dir"], p, run_name="t", workers=1)
    rep2 = json.loads((out2 / "report.json").read_text())
    assert rep2["motion"]["sidecar_reused"] is True
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** the sequence above; `units.csv` from `init.centroids`, `np.diff(A.indptr)`, `YrA.max(1)`. `report.json` keys: `stages` (list), `counts` (`seeds_init, seeds_pnr, seeds_gmm, seeds_merged, units_initA, units_init_merge, units_final, iter`), `motion` (`sidecar_reused`, `rmse_quality_median`), `bytes` (`raw, store_peak, factors, cnmf`), `peak_rss`, `wall_total`, `workers`, `version`.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add end-to-end pipeline, report and run CLI"`

---

### Task 11: scoring against truth and the Minian reference

**Files:**
- Create: `bench/__init__.py`, `bench/score.py`, `bench/minian_export.py`, `bench/README.md`, `tests/test_score.py`; Modify: `src/fastminian/cli.py` (add `score`), `tests/conftest.py` (insert the repo root into `sys.path` so `import bench` works)

**Interfaces:**
- Produces:
  - `score.match_units(unit_centroids: np.ndarray, true_centroids: np.ndarray, max_dist: float = 5.0) -> tuple[np.ndarray, np.ndarray]` greedy one-to-one by increasing distance; returns `(unit_idx, true_idx)` of matches.
  - `score.score_run(A: csr_matrix, traces: np.ndarray, H: int, W: int, meta_path: Path, shifts: np.ndarray) -> dict` with `recall, precision, f1, n_units, n_true, trace_corr_median, trace_corr_q25, align_offset_px`: truth from `synth.synth_setup(meta)`, true centroids shifted by the fitted motion offset (`bench_opt.motion_fit(est_shifts, setup.shifts)` → `c`, `sign`: template coordinates = frame coordinates + mean correction; derive and document the sign from the warp convention, then verify with the pipeline's own units: the median matched distance must be < 1.5 px, otherwise flip), trace corr = Pearson(`traces[k]`, `setup.C[j] − 20`).
  - CLI `fastminian score RUN_DIR` → prints JSON and writes `RUN_DIR/score.json`; reads `cnmf/A.npz`, `cnmf/YrA.npy`, `motion/arrays.npz`, `factors/meta.json`, `<raw>/../meta.json`.
  - `bench/minian_export.py RUN_OUT_DIR`: opens Minian's zarr intermediates (`minian_intermediate/A_new.zarr/A_new`, `C_new.zarr/C_new`) and `minian/motion.zarr/motion` with `zarr` + `xarray` from the **minian venv** (`/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/minian_v2.0/.venv/bin/python`, this script is run there, not in ours), writes `minian_units.npz` with `A (K, H*W) float32 dense`, `C (K, T)`, `shifts (T, 2)` (Minian's `motion` is (frame, shift_dim) in (height, width) order); then `fastminian score --minian minian_units.npz --raw <raw_dir>` scores it with the same function.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from bench import score  # bench/ has an __init__.py; tests import it via conftest sys.path insert


def test_match_units_greedy_one_to_one():
    true = np.array([[10, 10], [20, 20], [40, 40]], float)
    units = np.array([[10.5, 10], [11, 11], [21, 20], [80, 80]], float)
    ui, ti = score.match_units(units, true, 5.0)
    assert set(zip(ui, ti)) == {(0, 0), (2, 1)}
    s = score.summarise(ui, ti, n_units=4, n_true=3)
    assert s["recall"] == 2 / 3 and s["precision"] == 0.5
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**; then export and score Minian:
```bash
/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/minian_v2.0/.venv/bin/python bench/minian_export.py \
  /Users/mbrosch/minian_testing_local/synth_v4_5min/processed/minian_baseline_keep
.venv/bin/fastminian score --minian /Users/mbrosch/minian_testing_local/synth_v4_5min/processed/minian_baseline_keep/minian_units.npz \
  --raw /Users/mbrosch/minian_testing_local/synth_v4_5min/raw
```
Record Minian's `f1`, `trace_corr_median` in `bench/README.md` together with its 471 s / 24.9 GB from `$BENCH/bench_browser/results/BASELINE_5min.md`. Delete `minian_intermediate/` afterwards (25 GB) once `minian_units.npz` exists.
- [ ] **Step 4: Run, verify pass; lint; commit** `git commit -m "Add ground-truth scoring and Minian reference export"`

---

### Task 12: 5-min and 10-min runs, README, integration fixes

**Files:**
- Modify: `README.md`, `bench/README.md`; any module the runs expose as faulty.

- [ ] **Step 1: 5-min run, 6 workers**
```bash
.venv/bin/fastminian run /Users/mbrosch/minian_testing_local/synth_v4_5min/raw --workers 6 --name fast_5min
.venv/bin/fastminian score /Users/mbrosch/minian_testing_local/synth_v4_5min/processed/fast_5min
```
Expected: finishes; `report.json` wall under 160 s; score F1 ≥ Minian F1 − 0.02 and trace corr ≥ Minian − 0.02. If F1 fails, sweep `inita_corr` ∈ {0.6, 0.7, 0.9} via `--params` files, keep the best, and note it in `bench/README.md`. If a stage crashes, fix it in its module with a regression test, commit, rerun.
- [ ] **Step 2: 10-min run, 6 workers and 1 worker**
```bash
.venv/bin/fastminian run /Users/mbrosch/minian_testing_local/synth_v4_10min/raw --workers 6 --name fast_10min
.venv/bin/fastminian score /Users/mbrosch/minian_testing_local/synth_v4_10min/processed/fast_10min
.venv/bin/fastminian run /Users/mbrosch/minian_testing_local/synth_v4_10min/raw --workers 1 --name fast_10min_w1
```
Expected: both finish unattended; 6-worker wall ≤ 6 min target (report either way).
- [ ] **Step 3: Real-data smoke run**
```bash
.venv/bin/fastminian run /Users/mbrosch/minian_testing_local/WL25_demo --workers 6 --name fast_wl25
```
(Check first that `WL25_demo` holds `N.avi` segments directly; if they sit in a subfolder, pass that folder.) Expected: finishes; `units_final` between 50 and 2000; note the count and wall time.
- [ ] **Step 4: README** — inputs → outputs, the one command, the parameter file, what is out of scope, the measured table (5-min: Minian vs fastminian; 10-min: 1 and 6 workers). Commit `git commit -m "Document runs and results"`.

---

## Self-review notes

- Spec §4 (window 1) → Tasks 3–5; §5 (window 2) → Tasks 6–9; §3 contracts → Tasks 5 (factors), 10 (cnmf, report); §6 params → Task 4; §7 gates → each task's tests plus Task 11/12; §8 process → this plan; §9 risks → Task 12 step 1 (threshold sweep) and Review Focus.
- Names used across tasks: `Factors.{tile_rows, reconstruct_chunk, traces, AtY, YCt, sigma_flat, P, T}`, `FactorCorr.{pairs, window, traces_corr}`, `init.{initA, initC, unit_merge, centroids}`, `cnmf.{update_spatial, update_background, compute_trace, run_cnmf, CnmfResult}`, `Params` fields as listed in Task 4. Any renaming must be propagated to every task that names it.
- `Params.tile` must be passed to `tile_grid` everywhere; tests use 32 or 24 to get several tiles on small frames.
