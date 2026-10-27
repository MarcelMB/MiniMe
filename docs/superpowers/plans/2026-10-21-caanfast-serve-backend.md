# `caanfast serve` back end Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local FastAPI back end (`caanfast serve`) that opens CaAnFast run folders read-only and serves every stage's data as JSON or binary arrays, plus a job runner that starts new runs.

**Architecture:** `src/caanfast/serve/` holds small single-purpose modules (runs, frames, traces, contours, derived, cache, jobs, params metadata) behind two FastAPI routers. Results that older runs never saved are computed once in a background thread into `~/.cache/caanfast/<run id>/`; routes answer `202` until they are ready. The pipeline itself gains three small outputs (saved seeds, initial units, `progress.jsonl`).

**Tech Stack:** Python ≥ 3.11, FastAPI ≥ 0.115, uvicorn ≥ 0.30, httpx (tests), numpy, scipy, OpenCV, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-21-caanfast-serve-design.md`. This is plan 1 of 2; the React front end (plan 2, `2026-10-21-caanfast-serve-frontend.md`) builds on the API this plan produces.

## Global Constraints

- Package `caanfast`, code under `src/caanfast/serve/`, command `caanfast serve`.
- Bind to `127.0.0.1` only; reject requests whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>`; every `/api/` request carries the token (`X-Token` header, or `?token=` for `EventSource`).
- `serve` never writes inside a run folder; derived results go to `$CAANFAST_CACHE` or `~/.cache/caanfast/<run id>/`.
- `{id}` = first 12 hex chars of SHA-1 of the run folder's resolved absolute path.
- Binary arrays: raw little-endian bytes, headers `X-Shape` (comma list, `""` for 0-d) and `X-Dtype` (`uint8` | `int32` | `float32`).
- Low-pass: CaLab's cosine-tapered FFT low-pass, cutoff `f_lp` in Hz, gain 1 below `0.5·f_lp`, `0.5·(1 + cos(π·(f − 0.5·f_lp)/f_lp))` up to `1.5·f_lp`, 0 above; `f_lp` clamped to Nyquist; traces shorter than 8 samples unchanged; windows filtered with a 5 s margin each side.
- Frame contrast: 1st / 99.9th percentile of ≤ 50 frames spread evenly over the recording, cached per run and source.
- Frame rate: `raw_dir/../meta.json` `fps`, else `frameRate` of `metaData.json` / `neural_metaData.json` in `raw_dir`, else the user's value stored in the cache.
- `ruff format` and `ruff check --fix` on every Python file touched; type hints on every function signature; one-sentence commit messages, no attribution trailer.

## Review Focus

- A run whose raw folder moved or whose drive is unplugged: summary says `raw: false`, `/frame?source=raw` returns 404 with a reason, everything else works (test in Task 4).
- A run written before this plan (no `seeds/`, `init/`, `progress.jsonl`): seeds and initial units come from the cache after a `202`, and the run folder is byte-for-byte unchanged (test in Task 7).
- Paths with spaces, like `$SSD/...`: the shared test run lives in a folder whose name contains a space, so every route and the job subprocess are exercised with one (fixture in Task 3, job test in Task 10).
- Boundary indices: frame `T-1` and the last unit work; frame `T`, negative frames and unit `n` return 400, not 500 (tests in Tasks 8 and 9).
- Launching a run whose output folder already holds a run: refused unless `overwrite` is set, so a launch never silently deletes an earlier run (test in Task 10).

## File Structure

| File | Responsibility |
|---|---|
| `src/caanfast/__main__.py` | `python -m caanfast` → the Typer app (used by the job runner) |
| `src/caanfast/report.py` (modify) | `StageTimer.attach_progress`: stage start/end lines in `progress.jsonl` |
| `src/caanfast/pipeline.py` (modify) | save `seeds/{init,pnr,merged}.npy`, `init/A.npz`; attach progress |
| `src/caanfast/serve/__init__.py` | package marker |
| `src/caanfast/serve/arrays.py` | binary array responses + decoder for tests |
| `src/caanfast/serve/cache.py` | `cache_root`, `RunCache`, `Background` (one-shot threads) |
| `src/caanfast/serve/runs.py` | `run_id`, `is_run`, `find_runs`, `fps_from_files`, `RunView`, `Registry` |
| `src/caanfast/serve/frames.py` | raw / preprocessed / reconstruction frames, contrast, LRU, `tile_map` |
| `src/caanfast/serve/traces.py` | `lowpass_gain`, `lowpass`, `window`, `overview`, `zooms` |
| `src/caanfast/serve/contours.py` | footprint outlines as polygons |
| `src/caanfast/serve/derived.py` | seeds / initial units / block residual for runs that lack them |
| `src/caanfast/serve/paramsmeta.py` | parameter groups, help text, the "common" list |
| `src/caanfast/serve/jobs.py` | `Job`, `JobManager` (queue, subprocess, cancel, persistence) |
| `src/caanfast/serve/app.py` | `create_app`: guard middleware, state, static files, router wiring |
| `src/caanfast/serve/routes_runs.py` | `/api/fs`, `/api/runs…`, `/api/params/defaults` |
| `src/caanfast/serve/routes_jobs.py` | `/api/jobs…` incl. server-sent events |
| `src/caanfast/serve/server.py` | free port, token, start URL, uvicorn, browser |
| `src/caanfast/cli.py` (modify) | `caanfast serve` command |
| `tests/conftest.py` (modify) | `TINY_PARAMS`, session-scoped `tiny_run`, `run_copy`, `client` |
| `tests/test_serve_*.py` | one test file per module |

Note: `bench/stage_viewer.py` exists only as an untracked file on `fix/rank-vs-T`, not on this branch. The functions the spec says to move (`raw_frame`, `store_frame`, `recon_frame`, `tile_map`, `block_residual`, `outline`) are written out in full in Tasks 4, 6 and 7.

---

### Task 1: Dependencies, `python -m caanfast`, binary arrays

**Files:**
- Modify: `pyproject.toml`
- Create: `src/caanfast/__main__.py`, `src/caanfast/serve/__init__.py`, `src/caanfast/serve/arrays.py`
- Test: `tests/test_serve_arrays.py`

**Interfaces:**
- Produces: `array_response(a: np.ndarray) -> fastapi.Response`; `decode(body: bytes, shape: str, dtype: str) -> np.ndarray`; `DTYPES = ("uint8", "int32", "float32")`.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, change the two dependency lists to:

```toml
dependencies = [
  "numpy>=2.0", "scipy>=1.13", "numba>=0.60", "opencv-python-headless>=4.9",
  "scikit-learn>=1.4", "pandas>=2.2", "typer>=0.12",
  "fastapi>=0.115", "uvicorn>=0.30",
]
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6", "httpx>=0.27"]
```

Run: `uv venv && uv pip install -e ".[dev,bench]"`
Expected: installs fastapi, uvicorn, httpx without errors.

- [ ] **Step 2: Write the failing test**

`tests/test_serve_arrays.py`:

```python
import numpy as np
import pytest

from caanfast.serve.arrays import array_response, decode


def test_array_roundtrip_keeps_shape_and_dtype():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    r = array_response(a)
    assert r.headers["X-Shape"] == "3,4"
    assert r.headers["X-Dtype"] == "float32"
    assert r.media_type == "application/octet-stream"
    np.testing.assert_array_equal(decode(r.body, "3,4", "float32"), a)


def test_array_response_makes_non_contiguous_input_contiguous():
    a = np.arange(20, dtype=np.uint8).reshape(4, 5)[:, ::2]
    r = array_response(a)
    np.testing.assert_array_equal(decode(r.body, r.headers["X-Shape"], "uint8"), a)


def test_empty_array_has_its_shape():
    r = array_response(np.zeros((0, 2), np.float32))
    assert r.headers["X-Shape"] == "0,2" and r.body == b""


def test_array_response_rejects_float64():
    with pytest.raises(TypeError):
        array_response(np.zeros(3))
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/pytest tests/test_serve_arrays.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve'`.

- [ ] **Step 4: Implement**

`src/caanfast/serve/__init__.py`:

```python
"""``caanfast serve``: run viewer and launcher in the browser (local back end)."""
```

`src/caanfast/serve/arrays.py`:

```python
"""Binary array responses: little-endian bytes plus ``X-Shape`` / ``X-Dtype`` headers."""

from __future__ import annotations

import numpy as np
from fastapi import Response

DTYPES = ("uint8", "int32", "float32")


def array_response(a: np.ndarray) -> Response:
    """``a`` as an ``application/octet-stream`` body the front end decodes with the headers."""
    a = np.asarray(a)
    if a.dtype.name not in DTYPES:
        raise TypeError(f"unsupported dtype {a.dtype} (one of {', '.join(DTYPES)})")
    a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
    return Response(
        a.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Shape": ",".join(str(n) for n in a.shape), "X-Dtype": a.dtype.name},
    )


def decode(body: bytes, shape: str, dtype: str) -> np.ndarray:
    """Inverse of ``array_response`` (tests and scripts)."""
    dims = tuple(int(n) for n in shape.split(",")) if shape else ()
    return np.frombuffer(body, dtype=np.dtype(dtype).newbyteorder("<")).reshape(dims)
```

`src/caanfast/__main__.py`:

```python
from caanfast.cli import app

app()
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_arrays.py -v && .venv/bin/python -m caanfast --help`
Expected: 4 passed; the help lists `run`, `synth`, `score`.

- [ ] **Step 6: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add pyproject.toml src/caanfast/__main__.py src/caanfast/serve tests/test_serve_arrays.py
git commit -m "Add FastAPI and the binary array format for caanfast serve"
```

---

### Task 2: Pipeline saves seeds, initial units and progress

**Files:**
- Modify: `src/caanfast/report.py` (class `StageTimer`)
- Modify: `src/caanfast/pipeline.py` (`_seeds`, `run`, new `_write_seeds`, `_write_init`)
- Test: `tests/test_pipeline.py` (extend `test_pipeline_end_to_end_tiny`, add `test_stage_timer_progress`)

**Interfaces:**
- Produces: `StageTimer.attach_progress(path: Path) -> None`; progress lines `{"stage": str, "event": "start"|"end", "t": float, "seconds"?: float}`.
- Produces: `pipeline._seeds(fac, fc, params, timer, keep: dict[str, np.ndarray] | None = None) -> pd.DataFrame`, filling `keep["init"|"pnr"|"merged"]` with `(n, 2)` float32 `(height, width)` positions.
- Produces: run folder files `seeds/init.npy`, `seeds/pnr.npy`, `seeds/merged.npy`, `init/A.npz`, `progress.jsonl`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
def test_stage_timer_progress(tmp_path):
    from caanfast.report import StageTimer

    timer = StageTimer()
    with timer("probe"):
        pass
    timer.attach_progress(tmp_path / "progress.jsonl")
    with timer("pass1"):
        pass
    lines = [json.loads(s) for s in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [(x["stage"], x["event"]) for x in lines] == [
        ("probe", "end"),
        ("pass1", "start"),
        ("pass1", "end"),
    ]
    assert lines[2]["seconds"] >= 0 and "seconds" not in lines[1]
```

At the end of `test_pipeline_end_to_end_tiny`, right after `rep = json.loads(...)` and the existing asserts on `out`, add:

```python
    for k in ("init", "pnr", "merged"):
        pos = np.load(out / "seeds" / f"{k}.npy")
        assert pos.dtype == np.float32 and pos.shape == (rep["counts"][f"seeds_{k}"], 2)
    A0 = sps.load_npz(out / "init" / "A.npz")
    assert A0.shape == (rep["counts"]["units_init_merge"], 64 * 64)
    events = [json.loads(s) for s in (out / "progress.jsonl").read_text().splitlines()]
    ends = [e["stage"] for e in events if e["event"] == "end"]
    assert ends == [r["stage"] for r in rep["stages"]]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -k "progress or end_to_end_tiny" -v`
Expected: FAIL — `AttributeError: 'StageTimer' object has no attribute 'attach_progress'` and `FileNotFoundError ... seeds/init.npy`.

- [ ] **Step 3: Implement `StageTimer` progress**

Replace the class in `src/caanfast/report.py` with:

```python
class StageTimer:
    """``with timer("name"): ...`` appends ``{"stage", "seconds"}`` (wall) to ``rows``.

    After ``attach_progress(path)`` every stage start and end is also appended to ``path``
    as one JSON line, so another process (the ``caanfast serve`` job runner) can follow.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.progress: Path | None = None

    def attach_progress(self, path: Path) -> None:
        """Start logging to ``path`` (overwritten); stages already finished come first."""
        self.progress = Path(path)
        now = time.time()
        with self.progress.open("w") as f:
            for r in self.rows:
                f.write(json.dumps({"stage": r["stage"], "event": "end", "t": now, **r}) + "\n")

    def _log(self, stage: str, event: str, seconds: float | None = None) -> None:
        if self.progress is None:
            return
        doc: dict[str, Any] = {"stage": stage, "event": event, "t": time.time()}
        if seconds is not None:
            doc["seconds"] = seconds
        with self.progress.open("a") as f:
            f.write(json.dumps(doc) + "\n")

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        self._log(stage, "start")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.rows.append({"stage": stage, "seconds": dt})
            self._log(stage, "end", dt)

    def to_json(self, path: Path, extra: dict[str, Any]) -> None:
        doc = {"stages": self.rows, "counts": self.counts, **extra}
        Path(path).write_text(json.dumps(doc, indent=1, default=_json_default))
```

(`**r` repeats `stage` and adds `seconds`; the key order does not matter.)

- [ ] **Step 4: Implement the pipeline outputs**

In `src/caanfast/pipeline.py`, change `_seeds` to take `keep` and record the positions:

```python
def _seeds(
    fac: Factors,
    fc: corr.FactorCorr,
    params: Params,
    timer: StageTimer,
    keep: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Seeds through init -> PNR (-> GMM) -> merge; ``keep`` gets each step's positions."""
    p, n = params, timer.counts
    keep = {} if keep is None else keep
    with timer("seeds_init"):
        s = seeds.seeds_init(
            fac.block_max,
            fac.block_edges,
            fac.T,
            p.seeds_wnd,
            p.seeds_stp,
            p.seeds_max_wnd,
            p.seeds_diff_thres,
        )
    n["seeds_init"] = len(s)
    keep["init"] = _positions(s)
    with timer("pnr_refine"):
        s = _filter(seeds.pnr_refine(fac, s, thres=p.pnr_thres), "mask_pnr")
    n["seeds_pnr"] = len(s)
    keep["pnr"] = _positions(s)
    if p.gmm_refine:
        with timer("gmm_refine"):
            s = _filter(seeds.gmm_refine(fac, s), "mask_gmm")
    n["seeds_gmm"] = len(s)  # equals seeds_pnr when gmm_refine is off
    with timer("seeds_merge"):
        s = _filter(corr.seeds_merge(fc, s, fac.max_proj, p.merge_dist, p.merge_corr), "mask_mrg")
    n["seeds_merged"] = len(s)
    keep["merged"] = _positions(s)
    return s


def _positions(s: pd.DataFrame) -> np.ndarray:
    return s[["height", "width"]].to_numpy(np.float32).reshape(-1, 2)
```

Add next to `_write_cnmf`:

```python
def _write_seeds(d: Path, kept: dict[str, np.ndarray]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, pos in kept.items():
        np.save(d / f"{name}.npy", pos)


def _write_init(d: Path, A: sps.csr_matrix) -> None:
    d.mkdir(parents=True, exist_ok=True)
    sps.save_npz(d / "A.npz", sps.csr_matrix(A, dtype=np.float32))
```

In `run`, after `_write_params(out, params, raw_dir, raw_hash)` add:

```python
    timer.attach_progress(out / "progress.jsonl")
```

and replace

```python
    s = _seeds(fac, fc, params, timer)
    A, C = _init_units(fac, fc, s, params, timer)
```

with

```python
    kept: dict[str, np.ndarray] = {}
    s = _seeds(fac, fc, params, timer, keep=kept)
    _write_seeds(out / "seeds", kept)
    A, C = _init_units(fac, fc, s, params, timer)
    _write_init(out / "init", A)
```

`test_seeds_stage_runs_gmm_refine_only_when_enabled` monkeypatches the seed functions and calls `_seeds` without `keep`; `keep` defaults to a fresh dict, so it keeps passing.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: all pass (the end-to-end test takes ~30 s).

- [ ] **Step 6: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/report.py src/caanfast/pipeline.py tests/test_pipeline.py
git commit -m "Save each seed step, the initial units and a progress log with every run"
```

---

### Task 3: Test fixtures, the cache and opening runs

**Files:**
- Modify: `tests/conftest.py`
- Create: `src/caanfast/serve/cache.py`, `src/caanfast/serve/runs.py`
- Test: `tests/test_serve_runs.py`

**Interfaces:**
- Consumes: Task 2's run folder layout.
- Produces (`cache.py`): `cache_root() -> Path`; `RunCache(run_id: str, root: Path | None = None)` with `.dir`, `.path(name) -> Path`, `.has(name) -> bool`, `.load_json(name, default=None)`, `.save_json(name, doc) -> None`, `.load_npy(name) -> np.ndarray`, `.save_npy(name, a) -> None`; `Background()` with `.ensure(key: str, fn: Callable[[], None]) -> dict` returning `{"status": "computing", "elapsed_s": float}` or `{"status": "failed", "error": str}`.
- Produces (`runs.py`): `run_id(path) -> str`; `is_run(path) -> bool`; `find_runs(root, max_depth=3) -> list[Path]`; `fps_from_files(raw_dir) -> float | None`; `changed_params(pj: dict) -> dict[str, dict]`; `RunView(path, cache_root=None)` with `.path .id .cache .params .report .H .W .T .ranks .raw_dir`, properties `.has_raw .has_store .has_saved_seeds .has_saved_init .fps`, `.set_fps(fps)`, cached `.factors .sidecar .raw_info .A`, `.traces(kind) -> np.ndarray (mmap)`, `.summary() -> dict`, `.listing() -> dict`; `Registry(cache_root=None)` with `.open(path) -> RunView`, `.get(rid) -> RunView` (raises `KeyError`).
- Produces (`tests/conftest.py`): `TINY_PARAMS`, session fixture `tiny_run -> Path`, fixture `run_copy -> Path` (a writable copy), fixture `cache_dir -> Path`.

- [ ] **Step 1: Add the fixtures**

Append to `tests/conftest.py`:

```python
import shutil

TINY_PARAMS = dict(
    tile=32,
    seeds_wnd=100,
    seeds_stp=50,
    seeds_max_wnd=8,
    template_frames=50,
    inita_wnd=8,
    dl_wnd=5,
    size_thres=(9, None),
    svd_k0=16,
    motion_before_tophat=False,
)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """One finished run (tile store kept) on a 300-frame 64 x 64 synthetic session.

    Shared by the serve tests, which must not modify it; the folder name has a space, like
    ``$SSD``. Use ``run_copy`` to change files.
    """
    from caanfast import pipeline
    from caanfast.params import Params

    root = tmp_path_factory.mktemp("serve data")
    raw = root / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return pipeline.run(raw, Params(**TINY_PARAMS), run_name="t", workers=1, keep_store=True)


@pytest.fixture
def run_copy(tiny_run, tmp_path):
    """A writable copy of ``tiny_run`` (its params.json still points at the same raw folder)."""
    dst = tmp_path / "copy run" / "t"
    shutil.copytree(tiny_run, dst)
    return dst


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_serve_runs.py`:

```python
import json
import shutil
import time

import pytest

from caanfast.serve.cache import Background, RunCache
from caanfast.serve.runs import (
    Registry,
    RunView,
    changed_params,
    find_runs,
    fps_from_files,
    is_run,
    run_id,
)


def test_find_runs_finds_the_run_below_its_session(tiny_run):
    assert is_run(tiny_run) and not is_run(tiny_run.parent)
    assert find_runs(tiny_run.parent.parent) == [tiny_run]
    assert find_runs(tiny_run) == [tiny_run]


def test_run_id_is_stable_and_short(tiny_run):
    assert run_id(tiny_run) == run_id(tiny_run / ".." / "t")
    assert len(run_id(tiny_run)) == 12


def test_run_view_summary(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    s = rv.summary()
    assert (s["H"], s["W"], s["T"]) == (64, 64, 300)
    assert s["fps"] == 30.0
    assert len(s["ranks"]) == 4 and s["tile"] == 32
    assert s["available"] == {"raw": True, "pre": True, "seeds_saved": True, "init_saved": True}
    assert s["counts"]["units_final"] == rv.A.shape[0]
    assert "tile" in s["params_changed"] and "workers" not in s["params_changed"]
    assert s["name"] == "t" and s["id"] == rv.id


def test_run_view_reports_missing_parts(run_copy, cache_dir):
    pj = json.loads((run_copy / "params.json").read_text())
    pj["raw_dir"] = str(run_copy / "gone")
    (run_copy / "params.json").write_text(json.dumps(pj))
    for d in ("tiles", "seeds", "init"):
        shutil.rmtree(run_copy / d)
    rv = RunView(run_copy, cache_dir)
    assert rv.summary()["available"] == {
        "raw": False,
        "pre": False,
        "seeds_saved": False,
        "init_saved": False,
    }
    assert rv.fps is None
    rv.set_fps(20)
    assert RunView(run_copy, cache_dir).fps == 20.0
    with pytest.raises(ValueError):
        rv.set_fps(0)


def test_fps_from_miniscope_metadata(tmp_path):
    raw = tmp_path / "My_V4_Miniscope"
    raw.mkdir()
    assert fps_from_files(raw) is None
    (raw / "metaData.json").write_text(json.dumps({"frameRate": "20FPS"}))
    assert fps_from_files(raw) == 20.0
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30}))
    assert fps_from_files(raw) == 30.0


def test_changed_params_compares_tuples_as_lists():
    assert changed_params({"size_thres": [25, None], "tile": 32}) == {
        "tile": {"value": 32, "default": 64}
    }


def test_registry_reopens_by_id_after_restart(tiny_run, cache_dir):
    rid = Registry(cache_dir).open(tiny_run).id
    assert Registry(cache_dir).get(rid).path == tiny_run.resolve()
    with pytest.raises(KeyError):
        Registry(cache_dir).get("0" * 12)


def test_run_cache_roundtrip(cache_dir):
    import numpy as np

    c = RunCache("abc", cache_dir)
    assert not c.has("x.npy") and c.load_json("a.json", {}) == {}
    c.save_npy("x.npy", np.arange(3, dtype=np.float32))
    c.save_json("a.json", {"k": 1})
    assert c.has("x.npy") and c.load_json("a.json") == {"k": 1}
    np.testing.assert_array_equal(c.load_npy("x.npy"), [0, 1, 2])


def test_background_runs_once_and_reports_failure():
    calls = []
    bg = Background()

    def slow():
        calls.append(1)
        time.sleep(0.2)

    assert bg.ensure("a", slow)["status"] == "computing"
    assert bg.ensure("a", slow)["status"] == "computing"
    time.sleep(0.4)
    assert calls == [1]

    def boom():
        raise RuntimeError("no")

    bg.ensure("b", boom)
    time.sleep(0.1)
    assert bg.ensure("b", boom) == {"status": "failed", "error": "RuntimeError: no"}
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_runs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve.cache'`.

- [ ] **Step 4: Implement `cache.py`**

```python
"""Per-run cache outside the run folder (``$CAANFAST_CACHE`` or ``~/.cache/caanfast``)."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def cache_root() -> Path:
    return Path(os.environ.get("CAANFAST_CACHE", Path.home() / ".cache" / "caanfast"))


class RunCache:
    """Files derived from one run, in ``<root>/<run id>/``; writes are atomic (tmp + rename)."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.dir = Path(root or cache_root()) / run_id

    def path(self, name: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / name

    def has(self, name: str) -> bool:
        return (self.dir / name).exists()

    def load_json(self, name: str, default: Any = None) -> Any:
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else default

    def save_json(self, name: str, doc: Any) -> None:
        tmp = self.path(name + ".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(self.dir / name)

    def load_npy(self, name: str) -> np.ndarray:
        return np.load(self.dir / name)

    def save_npy(self, name: str, a: np.ndarray) -> None:
        tmp = self.path(name + ".tmp.npy")
        np.save(tmp, a)
        tmp.replace(self.dir / name)


class Background:
    """Slow computations run once per key in a daemon thread.

    ``ensure`` starts ``fn`` if it is neither running nor failed and returns the status a
    route sends with ``202``; the route itself checks the cache for the finished result.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, float] = {}
        self._failed: dict[str, str] = {}

    def ensure(self, key: str, fn: Callable[[], None]) -> dict[str, Any]:
        with self._lock:
            if key in self._failed:
                return {"status": "failed", "error": self._failed[key]}
            if key not in self._running:
                self._running[key] = time.time()
                threading.Thread(target=self._run, args=(key, fn), daemon=True).start()
            return {"status": "computing", "elapsed_s": time.time() - self._running[key]}

    def _run(self, key: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - reported to the page, not raised
            with self._lock:
                self._failed[key] = f"{type(e).__name__}: {e}"
        finally:
            with self._lock:
                self._running.pop(key, None)
```

- [ ] **Step 5: Implement `runs.py`**

```python
"""Find run folders and open one read-only (``RunView``)."""

from __future__ import annotations

import hashlib
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps

from caanfast import io, motion
from caanfast.factors import Factors
from caanfast.params import Params
from caanfast.serve.cache import RunCache, cache_root

MARKERS = ("params.json", "report.json")


def run_id(path: Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]


def is_run(path: Path) -> bool:
    return all((Path(path) / m).is_file() for m in MARKERS)


def find_runs(root: Path, max_depth: int = 3) -> list[Path]:
    """Run folders at or below ``root`` (never inside a run), newest report first."""
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if is_run(d):
            found.append(d)
            return
        if depth == max_depth:
            return
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for p in subs:
            walk(p, depth + 1)

    walk(Path(root), 0)
    return sorted(found, key=lambda p: (p / "report.json").stat().st_mtime, reverse=True)


def _leading_number(v: Any) -> float | None:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(v))
    return float(m.group(1)) if m else None


def fps_from_files(raw_dir: Path) -> float | None:
    """``fps`` of ``raw_dir/../meta.json``, else ``frameRate`` of a Miniscope metadata file."""
    try:
        meta = Path(raw_dir).parent / "meta.json"
        if meta.is_file():
            return float(json.loads(meta.read_text())["fps"])
        for name in ("metaData.json", "neural_metaData.json"):
            p = Path(raw_dir) / name
            if p.is_file():
                v = _leading_number(json.loads(p.read_text()).get("frameRate"))
                if v:
                    return v
    except (OSError, ValueError, KeyError):
        return None
    return None


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(v))


def changed_params(pj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parameters of a run's params.json that differ from today's defaults."""
    out = {}
    for k, d in Params().asdict().items():
        if k in pj and _norm(pj[k]) != _norm(d):
            out[k] = {"value": pj[k], "default": _norm(d)}
    return out


class RunView:
    """One run folder, read-only; the large parts load on first use."""

    def __init__(self, path: Path, cache_root: Path | None = None) -> None:
        self.path = Path(path).resolve()
        if not is_run(self.path):
            raise FileNotFoundError(f"not a caanfast run: {path}")
        self.id = run_id(self.path)
        self.cache = RunCache(self.id, cache_root)
        self.params: dict[str, Any] = json.loads((self.path / "params.json").read_text())
        self.report: dict[str, Any] = json.loads((self.path / "report.json").read_text())
        m = json.loads((self.path / "factors" / "meta.json").read_text())
        self.H, self.W, self.T = int(m["H"]), int(m["W"]), int(m["T"])
        self.ranks: list[int] = [int(r) for r in m["ranks"]]
        self.tile: int = int(m["tile"])
        raw = self.params.get("raw_dir")
        self.raw_dir: Path | None = Path(raw) if raw else None

    @property
    def has_raw(self) -> bool:
        return self.raw_dir is not None and self.raw_dir.is_dir() and any(self.raw_dir.glob("*.avi"))

    @property
    def has_store(self) -> bool:
        return (self.path / "tiles" / "tile_000.part00").is_file()

    @property
    def has_saved_seeds(self) -> bool:
        return (self.path / "seeds" / "merged.npy").is_file()

    @property
    def has_saved_init(self) -> bool:
        return (self.path / "init" / "A.npz").is_file()

    @property
    def fps(self) -> float | None:
        if self.raw_dir is not None:
            f = fps_from_files(self.raw_dir)
            if f:
                return f
        user = self.cache.load_json("fps.json")
        return float(user["fps"]) if user else None

    def set_fps(self, fps: float) -> None:
        if not 0 < float(fps) <= 1000:
            raise ValueError(f"frame rate must be in (0, 1000] Hz, got {fps}")
        self.cache.save_json("fps.json", {"fps": float(fps)})

    @cached_property
    def factors(self) -> Factors:
        return Factors.load(self.path / "factors")

    @cached_property
    def sidecar(self) -> motion.MotionSidecar:
        return motion.load_sidecar(self.path / "motion")

    @cached_property
    def raw_info(self) -> io.RawInfo:
        if not self.has_raw:
            raise LookupError(f"raw videos not found in {self.raw_dir}")
        return io.probe(self.raw_dir)

    @cached_property
    def A(self) -> sps.csr_matrix:
        return sps.load_npz(self.path / "cnmf" / "A.npz").tocsr()

    def traces(self, kind: str) -> np.ndarray:
        if kind not in ("C", "YrA"):
            raise ValueError(f"kind must be C or YrA, got {kind!r}")
        return np.load(self.path / "cnmf" / f"{kind}.npy", mmap_mode="r")

    def listing(self) -> dict[str, Any]:
        rep = self.report
        return {
            "id": self.id,
            "name": self.path.name,
            "path": str(self.path),
            "units": rep["counts"].get("units_final"),
            "T": self.T,
            "fps": self.fps,
            "wall_total": rep.get("wall_total"),
            "mtime": (self.path / "report.json").stat().st_mtime,
        }

    def summary(self) -> dict[str, Any]:
        rep = self.report
        return {
            **self.listing(),
            "H": self.H,
            "W": self.W,
            "tile": self.tile,
            "ranks": self.ranks,
            "raw_dir": str(self.raw_dir) if self.raw_dir else None,
            "params": self.params,
            "params_changed": changed_params(self.params),
            "stages": rep["stages"],
            "counts": rep["counts"],
            "peak_rss": rep.get("peak_rss"),
            "peak_rss_children": rep.get("peak_rss_children"),
            "bytes": rep.get("bytes", {}),
            "workers": rep.get("workers"),
            "motion": rep.get("motion", {}),
            "available": {
                "raw": self.has_raw,
                "pre": self.has_store,
                "seeds_saved": self.has_saved_seeds,
                "init_saved": self.has_saved_init,
            },
        }


class Registry:
    """Open runs by id; ids are remembered in ``<cache root>/runs.json`` across restarts."""

    def __init__(self, cache_root_: Path | None = None) -> None:
        self.root = Path(cache_root_ or cache_root())
        self._runs: dict[str, RunView] = {}

    def _known(self) -> dict[str, str]:
        p = self.root / "runs.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def open(self, path: Path) -> RunView:
        rid = run_id(path)
        if rid not in self._runs:
            self._runs[rid] = RunView(path, self.root)
            known = self._known()
            if known.get(rid) != str(self._runs[rid].path):
                known[rid] = str(self._runs[rid].path)
                self.root.mkdir(parents=True, exist_ok=True)
                tmp = self.root / "runs.json.tmp"
                tmp.write_text(json.dumps(known))
                tmp.replace(self.root / "runs.json")
        return self._runs[rid]

    def get(self, rid: str) -> RunView:
        if rid in self._runs:
            return self._runs[rid]
        path = self._known().get(rid)
        if path is None or not is_run(Path(path)):
            raise KeyError(rid)
        return self.open(Path(path))
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_runs.py -v`
Expected: 9 passed (the first test builds `tiny_run`, ~30 s).

- [ ] **Step 7: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add tests/conftest.py src/caanfast/serve/cache.py src/caanfast/serve/runs.py tests/test_serve_runs.py
git commit -m "Open run folders read-only for caanfast serve, with a per-run cache outside them"
```

---

### Task 4: Frames (raw, preprocessed, reconstruction)

**Files:**
- Create: `src/caanfast/serve/frames.py`
- Test: `tests/test_serve_frames.py`

**Interfaces:**
- Consumes: `RunView` (Task 3): `.raw_info`, `.factors`, `.path`, `.has_raw`, `.has_store`, `.T`, `.cache`.
- Produces: `SOURCES = ("raw", "pre", "recon")`; `raw_frame(info, t) -> float32 (H, W)`; `store_frame(store, fac, t) -> float32 (H, W) | None`; `recon_frame(fac, t) -> float32 (H, W)`; `tile_map(fac, values) -> float32 (H, W)`; `FrameSource(run, capacity=256)` with `.available(source) -> bool`, `.has_contrast(source) -> bool`, `.contrast(source) -> tuple[float, float]` (computes and caches), `.frame(source, t) -> uint8 (H, W)`. Errors: `ValueError` unknown source, `IndexError` frame out of range, `LookupError` source not available.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_frames.py`:

```python
import json
import shutil

import numpy as np
import pytest

from caanfast import io
from caanfast.serve.frames import FrameSource, raw_frame, recon_frame, store_frame
from caanfast.serve.runs import RunView


@pytest.mark.parametrize("t", [0, 99, 100, 250, 299])
def test_raw_frame_matches_sequential_decode(tiny_run, cache_dir, t):
    rv = RunView(tiny_run, cache_dir)
    info = rv.raw_info
    seg, k = divmod(t, 100)
    want = io.read_avi(info.segments[seg], max_frames=k + 1, bgr=info.bgr)[k]
    np.testing.assert_array_equal(raw_frame(info, t), want.astype(np.float32))


def test_store_and_recon_frames_have_the_frame_shape(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    pre = store_frame(tiny_run / "tiles", rv.factors, 10)
    rec = recon_frame(rv.factors, 10)
    assert pre.shape == rec.shape == (64, 64)
    # the reconstruction approximates the preprocessed frame
    assert np.corrcoef(pre.ravel(), rec.ravel())[0, 1] > 0.5


def test_frame_is_uint8_with_cached_contrast(tiny_run, cache_dir):
    fs = FrameSource(RunView(tiny_run, cache_dir))
    assert not fs.has_contrast("recon")
    f = fs.frame("recon", 299)
    assert f.dtype == np.uint8 and f.shape == (64, 64)
    doc = json.loads((cache_dir / fs.run.id / "contrast.json").read_text())
    assert doc["recon"][0] < doc["recon"][1]
    assert fs.has_contrast("recon")


def test_frame_errors(run_copy, cache_dir):
    shutil.rmtree(run_copy / "tiles")
    pj = json.loads((run_copy / "params.json").read_text())
    pj["raw_dir"] = str(run_copy / "unplugged")
    (run_copy / "params.json").write_text(json.dumps(pj))
    fs = FrameSource(RunView(run_copy, cache_dir))
    assert not fs.available("pre") and not fs.available("raw") and fs.available("recon")
    with pytest.raises(LookupError):
        fs.frame("pre", 0)
    with pytest.raises(LookupError):
        fs.frame("raw", 0)
    with pytest.raises(IndexError):
        fs.frame("recon", 300)
    with pytest.raises(IndexError):
        fs.frame("recon", -1)
    with pytest.raises(ValueError):
        fs.frame("movie", 0)
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve.frames'`.

- [ ] **Step 3: Implement**

`src/caanfast/serve/frames.py`:

```python
"""Frame ``t`` of a run as raw, preprocessed (tile store) or reconstruction, scaled to uint8."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from caanfast import io
from caanfast.factors import Factors
from caanfast.serve.runs import RunView

SOURCES = ("raw", "pre", "recon")


def raw_frame(info: io.RawInfo, t: int) -> np.ndarray:
    """Frame ``t`` of the raw session, decoded by seeking inside its segment."""
    edges = np.cumsum([0, *info.segment_frames])
    s = int(np.searchsorted(edges, t, side="right") - 1)
    k = t - int(edges[s])
    cap = io._open(info.segments[s], info.bgr)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = cap.read()
    finally:
        cap.release()
    if not ok:
        raise OSError(f"cannot read frame {k} of {info.segments[s]}")
    return io._gray(f, info.bgr).astype(np.float32)


def store_frame(store: Path, fac: Factors, t: int) -> np.ndarray | None:
    """Frame ``t`` of the preprocessed uint8 tile store, or None if the store is gone."""
    parts = sorted(Path(store).glob("tile_000.part*"))
    if not parts:
        return None
    n_per = [p.stat().st_size // fac.grid[0].pixels for p in parts]
    edges = np.cumsum([0, *n_per])
    p = int(np.searchsorted(edges, t, side="right") - 1)
    k = t - int(edges[p])
    out = np.zeros((fac.H, fac.W_), np.float32)
    for tl in fac.grid:
        mm = np.memmap(Path(store) / f"tile_{tl.index:03d}.part{p:02d}", np.uint8, "r")
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = mm.reshape(-1, *tl.shape)[k]
    return out


def recon_frame(fac: Factors, t: int) -> np.ndarray:
    """Frame ``t`` rebuilt from the per-tile low-rank factors."""
    out = np.zeros((fac.H, fac.W_), np.float32)
    for b, tl in enumerate(fac.grid):
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = fac.reconstruct_chunk(b, t, t + 1)[:, 0].reshape(
            tl.shape
        )
    return out


def tile_map(fac: Factors, values: list[float]) -> np.ndarray:
    """(H, W) image with each tile filled with its value (e.g. its rank)."""
    out = np.zeros((fac.H, fac.W_), np.float32)
    for tl, v in zip(fac.grid, values, strict=True):
        out[tl.y0 : tl.y1, tl.x0 : tl.x1] = v
    return out


class FrameSource:
    """uint8 frames with a fixed per-run contrast and a small LRU of recent frames."""

    def __init__(self, run: RunView, capacity: int = 256) -> None:
        self.run = run
        self.capacity = capacity
        self._lru: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def available(self, source: str) -> bool:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r} (one of {', '.join(SOURCES)})")
        return {"raw": self.run.has_raw, "pre": self.run.has_store, "recon": True}[source]

    def _float(self, source: str, t: int) -> np.ndarray:
        if source == "raw":
            return raw_frame(self.run.raw_info, t)
        if source == "pre":
            f = store_frame(self.run.path / "tiles", self.run.factors, t)
            if f is None:
                raise LookupError("the tile store of this run was not kept")
            return f
        return recon_frame(self.run.factors, t)

    def _check(self, source: str, t: int) -> None:
        if not self.available(source):
            why = {
                "raw": f"raw videos not found in {self.run.raw_dir}",
                "pre": "the tile store was not kept (run with --keep-store)",
            }[source]
            raise LookupError(why)
        if not 0 <= t < self.run.T:
            raise IndexError(f"frame {t} out of range 0..{self.run.T - 1}")

    def has_contrast(self, source: str) -> bool:
        return source in self.run.cache.load_json("contrast.json", {})

    def contrast(self, source: str) -> tuple[float, float]:
        """1st / 99.9th percentile of <= 50 frames spread over the run (cached)."""
        doc = self.run.cache.load_json("contrast.json", {})
        if source not in doc:
            self._check(source, 0)
            ts = np.linspace(0, self.run.T - 1, min(50, self.run.T)).astype(int)
            vals = np.stack([self._float(source, int(t)) for t in ts])
            lo, hi = (float(x) for x in np.percentile(vals, [1, 99.9]))
            doc = self.run.cache.load_json("contrast.json", {})
            doc[source] = [lo, hi if hi > lo else lo + 1.0]
            self.run.cache.save_json("contrast.json", doc)
        lo, hi = doc[source]
        return float(lo), float(hi)

    def frame(self, source: str, t: int) -> np.ndarray:
        self._check(source, t)
        key = (source, t)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
        lo, hi = self.contrast(source)
        f = self._float(source, t)
        u8 = np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        with self._lock:
            self._lru[key] = u8
            while len(self._lru) > self.capacity:
                self._lru.popitem(last=False)
        return u8
```

If `test_raw_frame_matches_sequential_decode` fails because OpenCV's seek lands on the wrong frame for some codec, replace the body of `raw_frame` after computing `s, k` with `return io.read_avi(info.segments[s], max_frames=k + 1, bgr=info.bgr)[k].astype(np.float32)` (correct, slower: reads `k + 1` frames) and note it in `bench/README.md`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_frames.py -v`
Expected: 8 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/frames.py tests/test_serve_frames.py
git commit -m "Serve raw, preprocessed and reconstructed frames with a fixed contrast per run"
```

---

### Task 5: Traces: overview, windows, low-pass, zooms

**Files:**
- Create: `src/caanfast/serve/traces.py`
- Test: `tests/test_serve_traces.py`

**Interfaces:**
- Produces: `MIN_LEN = 8`; `lowpass_gain(n: int, fs: float, f_lp: float) -> np.ndarray`; `lowpass(x, fs, f_lp) -> float32`; `window(trace, start, stop, fs=None, lowpass_hz=None, margin_s=5.0) -> float32`; `overview(C, bins) -> float32 (n, 2, bins)`; `zooms(peak: np.ndarray, T: int, fps: float | None, n_units=10, wnd_min=5.0, per_hour=3) -> dict` with `"units": list[int]`, `"windows": list[{"start": int, "stop": int}]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_traces.py`:

```python
import numpy as np
import pytest

from caanfast.serve.traces import lowpass, lowpass_gain, overview, window, zooms


def test_gain_is_one_below_half_cutoff_half_at_cutoff_zero_above():
    g = lowpass_gain(1000, fs=100.0, f_lp=5.0)  # 0.1 Hz bins: 5 Hz is bin 50
    f = np.fft.rfftfreq(1000, 0.01)
    assert np.all(g[f < 2.5] == 1.0)
    assert g[50] == pytest.approx(0.5)
    assert np.all(g[f >= 7.5] == 0.0)
    assert np.all(np.diff(g) <= 1e-12)


def test_lowpass_removes_a_fast_sine_and_keeps_a_slow_one():
    t = np.arange(1000) / 100.0
    slow, fast = np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 12.0 * t)
    y = lowpass(slow + fast, fs=100.0, f_lp=5.0)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, slow, atol=1e-4)


def test_short_traces_are_returned_unchanged():
    x = np.arange(5, dtype=np.float32)
    np.testing.assert_array_equal(lowpass(x, 30.0, 5.0), x)


def test_window_with_margin_matches_the_full_trace_filter():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=6000)).astype(np.float32)
    full = lowpass(x, 30.0, 3.0)
    w = window(x, 2000, 2600, fs=30.0, lowpass_hz=3.0)
    assert w.shape == (600,)
    assert np.max(np.abs(w - full[2000:2600])) < 0.01 * np.std(x)


def test_window_checks_its_arguments():
    x = np.zeros(100, np.float32)
    np.testing.assert_array_equal(window(x, 10, 20), x[10:20])
    with pytest.raises(IndexError):
        window(x, 90, 101)
    with pytest.raises(IndexError):
        window(x, 20, 20)
    with pytest.raises(ValueError):
        window(x, 0, 50, fs=None, lowpass_hz=5.0)


def test_overview_min_max_per_bin():
    C = np.arange(20, dtype=np.float32).reshape(2, 10)
    ov = overview(C, bins=5)
    assert ov.shape == (2, 2, 5)
    np.testing.assert_array_equal(ov[0, 0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(ov[1, 1], [11, 13, 15, 17, 19])
    assert overview(C, bins=50).shape == (2, 2, 10)


def test_zooms_pick_brightest_units_and_spread_windows():
    z = zooms(np.array([1.0, 5.0, 3.0]), T=3 * 3600 * 20, fps=20.0, n_units=2)
    assert z["units"] == [1, 2]
    assert len(z["windows"]) == 9
    assert z["windows"][1] == {"start": 24000, "stop": 30000}
    assert zooms(np.array([1.0]), T=100, fps=None)["windows"] == []
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_traces.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/caanfast/serve/traces.py`:

```python
"""Trace overview, windows and the CaTune-style low-pass (cutoff in Hz)."""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_LEN = 8  # shorter traces are returned unchanged (as CaLab's MIN_SPECTRUM_LEN)


def lowpass_gain(n: int, fs: float, f_lp: float) -> np.ndarray:
    """rfft gain of CaLab's low-pass: 1 below f_lp/2, cosine taper to 0 at 1.5 f_lp.

    Same curve as ``BandpassFilter.build_gain_curve`` in CaLab's ``crates/solver/src/
    filter.rs`` with only the low-pass half on; there the cutoff comes from tau_rise, here
    it is given in Hz. ``f_lp`` is clamped to Nyquist first, as there.
    """
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f_lp = min(float(f_lp), fs / 2.0)
    w = 0.5 * f_lp
    g = np.ones_like(f)
    band = (f >= f_lp - w) & (f < f_lp + w)
    g[band] = 0.5 * (1.0 + np.cos(np.pi * (f[band] - (f_lp - w)) / (2.0 * w)))
    g[f >= f_lp + w] = 0.0
    return g


def lowpass(x: np.ndarray, fs: float, f_lp: float) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.size < MIN_LEN:
        return x.copy()
    if f_lp <= 0 or fs <= 0:
        raise ValueError(f"cutoff and frame rate must be > 0 (got {f_lp} Hz, {fs} Hz)")
    y = np.fft.irfft(np.fft.rfft(x) * lowpass_gain(x.size, fs, f_lp), n=x.size)
    return y.astype(np.float32)


def window(
    trace: np.ndarray,
    start: int,
    stop: int,
    fs: float | None = None,
    lowpass_hz: float | None = None,
    margin_s: float = 5.0,
) -> np.ndarray:
    """``trace[start:stop]``, low-passed with ``margin_s`` of extra trace on each side."""
    T = len(trace)
    if not 0 <= start < stop <= T:
        raise IndexError(f"window {start}:{stop} outside 0:{T}")
    if lowpass_hz is None:
        return np.asarray(trace[start:stop], np.float32)
    if fs is None:
        raise ValueError("the low-pass needs the frame rate of the recording")
    m = int(round(margin_s * fs))
    a, b = max(0, start - m), min(T, stop + m)
    return lowpass(np.asarray(trace[a:b], np.float32), fs, lowpass_hz)[start - a : stop - a]


def overview(C: np.ndarray, bins: int) -> np.ndarray:
    """(n, 2, bins) float32 min / max of each trace over ``bins`` equal spans of frames."""
    n, T = C.shape
    bins = max(1, min(int(bins), T))
    starts = np.linspace(0, T, bins + 1).astype(int)[:-1]
    out = np.empty((n, 2, bins), np.float32)
    for i in range(0, n, 64):
        blk = np.asarray(C[i : i + 64], np.float32)
        out[i : i + 64, 0] = np.minimum.reduceat(blk, starts, axis=1)
        out[i : i + 64, 1] = np.maximum.reduceat(blk, starts, axis=1)
    return out


def zooms(
    peak: np.ndarray,
    T: int,
    fps: float | None,
    n_units: int = 10,
    wnd_min: float = 5.0,
    per_hour: int = 3,
) -> dict[str, Any]:
    """The ``n_units`` brightest units and ``wnd_min``-minute windows, ``per_hour`` per hour."""
    units = [int(k) for k in np.argsort(np.asarray(peak))[::-1][:n_units]]
    if not fps:
        return {"units": units, "windows": []}
    step = max(1, int(3600 / per_hour * fps))
    wnd = int(wnd_min * 60 * fps)
    windows = [{"start": t0, "stop": min(t0 + wnd, T)} for t0 in range(0, T, step)]
    return {"units": units, "windows": windows}
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_traces.py -v`
Expected: 7 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/traces.py tests/test_serve_traces.py
git commit -m "Add trace overviews, windows, zoom suggestions and CaTune's low-pass with a cutoff in Hz"
```

---

### Task 6: Footprint contours

**Files:**
- Create: `src/caanfast/serve/contours.py`
- Test: `tests/test_serve_contours.py`

**Interfaces:**
- Produces: `unit_polygons(A: sps.csr_matrix, H: int, W: int, rel: float = 0.2) -> list[list[list[list[float]]]]` — per unit a list of polygons, each a list of `[x, y]` pixel-centre points (x = column, y = row).

- [ ] **Step 1: Write the failing test**

`tests/test_serve_contours.py`:

```python
import numpy as np
import scipy.sparse as sps

from caanfast.serve.contours import unit_polygons


def _footprints(H, W, masks):
    rows = [m.astype(np.float32).ravel() for m in masks]
    return sps.csr_matrix(np.stack(rows))


def test_square_footprint_gives_one_polygon_on_its_border():
    H, W = 32, 40
    m = np.zeros((H, W))
    m[10:15, 20:25] = 1.0
    m[12, 22] = 3.0  # the peak; everything else is above 0.2 x peak
    (polys,) = unit_polygons(_footprints(H, W, [m]), H, W)
    assert len(polys) == 1
    pts = np.array(polys[0])
    assert (pts[:, 0].min(), pts[:, 0].max()) == (20, 24)
    assert (pts[:, 1].min(), pts[:, 1].max()) == (10, 14)


def test_two_islands_and_an_empty_unit():
    H, W = 20, 20
    a = np.zeros((H, W))
    a[2:5, 2:5] = 1.0
    a[12:15, 12:15] = 1.0
    out = unit_polygons(_footprints(H, W, [a, np.zeros((H, W))]), H, W)
    assert len(out[0]) == 2 and out[1] == []


def test_pixels_below_rel_of_the_peak_are_outside():
    H, W = 20, 20
    a = np.zeros((H, W))
    a[5:10, 5:10] = 0.1
    a[7, 7] = 1.0
    (polys,) = unit_polygons(_footprints(H, W, [a]), H, W, rel=0.2)
    assert np.array(polys[0]).tolist() == [[7.0, 7.0]]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/pytest tests/test_serve_contours.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/caanfast/serve/contours.py`:

```python
"""Footprint outlines as polygons for the browser to draw."""

from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sps


def unit_polygons(
    A: sps.csr_matrix, H: int, W: int, rel: float = 0.2
) -> list[list[list[list[float]]]]:
    """Per unit, the outer outlines of its pixels >= ``rel`` x its peak, as [x, y] points.

    Points are pixel centres (x = column, y = row); a one-pixel island is a one-point
    polygon. The 0.2 threshold matches the stage viewer's outlines.
    """
    A = sps.csr_matrix(A)
    out: list[list[list[list[float]]]] = []
    for k in range(A.shape[0]):
        lo, hi = A.indptr[k], A.indptr[k + 1]
        idx, val = A.indices[lo:hi], A.data[lo:hi]
        if idx.size == 0 or val.max() <= 0:
            out.append([])
            continue
        m = np.zeros(H * W, np.uint8)
        m[idx[val >= rel * val.max()]] = 1
        cs, _ = cv2.findContours(m.reshape(H, W), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.append([c[:, 0, :].astype(float).tolist() for c in cs])
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_contours.py -v`
Expected: 3 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/contours.py tests/test_serve_contours.py
git commit -m "Turn footprints into outline polygons for the browser"
```

---

### Task 7: Derived results for runs that lack them

**Files:**
- Create: `src/caanfast/serve/derived.py`
- Test: `tests/test_serve_derived.py`

**Interfaces:**
- Consumes: `RunView` (Task 3), `pipeline._seeds(..., keep=)` and `pipeline._init_units` (Task 2), `Params.from_json`.
- Produces: `SEED_STAGES = ("init", "pnr", "merged")`; `seeds(run) -> dict[str, np.ndarray] | None`; `compute_seeds_and_init(run) -> None`; `init_A(run) -> sps.csr_matrix | None`; `block_residual(fac, chunk=32) -> tuple[np.ndarray, np.ndarray]`; `residual(run) -> np.ndarray | None` (float32 `(n_chunks_total, 3)`: dy, dx, corr); `compute_residual(run) -> None`; `traces_overview(run, kind, bins) -> np.ndarray | None`; `compute_overview(run, kind, bins) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_derived.py`:

```python
import shutil

import numpy as np
import scipy.sparse as sps

from caanfast.serve import derived
from caanfast.serve.runs import RunView


def _snapshot(d):
    return {
        p.relative_to(d): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def test_saved_seeds_and_init_are_read_from_the_run(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    s = derived.seeds(rv)
    assert set(s) == {"init", "pnr", "merged"}
    assert derived.init_A(rv).shape[0] == rv.report["counts"]["units_init_merge"]


def test_older_run_gets_seeds_and_init_computed_into_the_cache(tiny_run, run_copy, cache_dir):
    shutil.rmtree(run_copy / "seeds")
    shutil.rmtree(run_copy / "init")
    before = _snapshot(run_copy)
    rv = RunView(run_copy, cache_dir)
    assert derived.seeds(rv) is None and derived.init_A(rv) is None
    derived.compute_seeds_and_init(rv)
    got = derived.seeds(rv)
    for k in derived.SEED_STAGES:
        np.testing.assert_array_equal(got[k], np.load(tiny_run / "seeds" / f"{k}.npy"))
    want = sps.load_npz(tiny_run / "init" / "A.npz")
    assert (derived.init_A(rv) != want).nnz == 0
    assert _snapshot(run_copy) == before


def test_residual_and_overview(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    assert derived.residual(rv) is None
    derived.compute_residual(rv)
    r = derived.residual(rv)
    assert r.dtype == np.float32 and r.shape == (len(rv.factors.block_max), 3)
    assert derived.traces_overview(rv, "C", 100) is None
    derived.compute_overview(rv, "C", 100)
    ov = derived.traces_overview(rv, "C", 100)
    assert ov.shape == (rv.A.shape[0], 2, 100)
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_derived.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/caanfast/serve/derived.py`:

```python
"""Results the stage pages need that a run did not save: computed once into the cache."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from caanfast import corr, motion, pipeline
from caanfast.factors import Factors
from caanfast.params import Params
from caanfast.report import StageTimer
from caanfast.serve.runs import RunView
from caanfast.serve.traces import overview

SEED_STAGES = ("init", "pnr", "merged")


def seeds(run: RunView) -> dict[str, np.ndarray] | None:
    """Seed positions (n, 2) per step: from the run if it saved them, else from the cache."""
    if run.has_saved_seeds:
        return {k: np.load(run.path / "seeds" / f"{k}.npy") for k in SEED_STAGES}
    if all(run.cache.has(f"seeds_{k}.npy") for k in SEED_STAGES):
        return {k: run.cache.load_npy(f"seeds_{k}.npy") for k in SEED_STAGES}
    return None


def init_A(run: RunView) -> sps.csr_matrix | None:
    if run.has_saved_init:
        return sps.load_npz(run.path / "init" / "A.npz").tocsr()
    if run.cache.has("init_A.npz"):
        return sps.load_npz(run.cache.dir / "init_A.npz").tocsr()
    return None


def compute_seeds_and_init(run: RunView) -> None:
    """Rerun the seed and init stages on the saved factors (~2.5 min on 3 h) into the cache."""
    fac = run.factors
    p = Params.from_json(run.path / "params.json")
    fc = corr.FactorCorr(fac, p.noise_freq)
    timer = StageTimer()
    kept: dict[str, np.ndarray] = {}
    s = pipeline._seeds(fac, fc, p, timer, keep=kept)
    A0, _ = pipeline._init_units(fac, fc, s, p, timer)
    tmp = run.cache.path("init_A.tmp.npz")
    sps.save_npz(tmp, sps.csr_matrix(A0, dtype=np.float32))
    tmp.replace(run.cache.dir / "init_A.npz")
    for k in SEED_STAGES:  # seeds last: their presence marks the computation done
        run.cache.save_npy(f"seeds_{k}.npy", kept[k])


def block_residual(fac: Factors, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Residual (dy, dx) and correlation of each corrected ``block_max`` vs the whole-run max.

    A consistent registration gives ~0 px for every block; a block locked to a slightly
    wrong offset shows up as a step here even when its per-frame quality looks normal.
    """
    ref = motion._mean_template(fac.max_proj[None].astype(np.float32), down=1)
    shifts, cor = [], []
    for i in range(0, len(fac.block_max), chunk):
        blk = fac.block_max[i : i + chunk].astype(np.float32)
        shifts.append(motion.estimate(blk, ref))
        cor.append(motion.frame_quality(blk, ref))
    return np.concatenate(shifts), np.concatenate(cor)


def residual(run: RunView) -> np.ndarray | None:
    return run.cache.load_npy("residual.npy") if run.cache.has("residual.npy") else None


def compute_residual(run: RunView) -> None:
    sh, cor = block_residual(run.factors)
    run.cache.save_npy("residual.npy", np.column_stack([sh, cor]).astype(np.float32))


def _overview_name(kind: str, bins: int) -> str:
    return f"overview_{kind}_{int(bins)}.npy"


def traces_overview(run: RunView, kind: str, bins: int) -> np.ndarray | None:
    name = _overview_name(kind, bins)
    return run.cache.load_npy(name) if run.cache.has(name) else None


def compute_overview(run: RunView, kind: str, bins: int) -> None:
    run.cache.save_npy(_overview_name(kind, bins), overview(run.traces(kind), bins))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_derived.py -v`
Expected: 3 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/derived.py tests/test_serve_derived.py
git commit -m "Compute seeds, initial units, block residuals and trace overviews into the cache when a run lacks them"
```

---

### Task 8: App, access guard, folder, run, image and frame routes

**Files:**
- Create: `src/caanfast/serve/app.py`, `src/caanfast/serve/routes_runs.py`
- Modify: `tests/conftest.py` (add `client`, `wait_ok`)
- Test: `tests/test_serve_app.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 7.
- Produces: `create_app(token: str, port: int | None = None, cache_root: Path | None = None, static_dir: Path | None = None, jobs: JobManager | None = None) -> FastAPI` with `app.state.registry: Registry`, `app.state.frames: dict[str, FrameSource]`, `app.state.background: Background`, `app.state.jobs`; `STATIC_DIR = Path(__file__).with_name("static")`.
- Produces routes: `GET /api/fs`, `GET /api/runs`, `GET /api/runs/{rid}/summary`, `POST /api/runs/{rid}/fps` (`{"fps": float}`), `GET /api/runs/{rid}/image/{name}`, `GET /api/runs/{rid}/frame/{t}?source=`; helper `computing(request, key, fn) -> JSONResponse` (status 202, body from `Background.ensure`, `500` if it failed).
- Produces (tests): fixture `client` (TestClient with `X-Token: tok`, app cache in `cache_dir`), helper `wait_ok(client, url, timeout=120) -> Response`.

- [ ] **Step 1: Add the test helpers**

Append to `tests/conftest.py`:

```python
import time as _time


@pytest.fixture
def client(cache_dir, tmp_path):
    from fastapi.testclient import TestClient

    from caanfast.serve.app import create_app

    app = create_app("tok", cache_root=cache_dir, static_dir=tmp_path / "no-static")
    c = TestClient(app)
    c.headers["X-Token"] = "tok"
    return c


def wait_ok(client, url, timeout=120):
    """GET ``url`` until it stops answering 202 (a background computation)."""
    t0 = _time.time()
    while True:
        r = client.get(url)
        if r.status_code != 202:
            return r
        assert _time.time() - t0 < timeout, f"{url} still computing"
        _time.sleep(0.2)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_serve_app.py`:

```python
import json

import numpy as np
from fastapi.testclient import TestClient

from caanfast.serve.app import create_app
from caanfast.serve.arrays import decode
from tests.conftest import wait_ok


def _arr(r):
    return decode(r.content, r.headers["X-Shape"], r.headers["X-Dtype"])


def _open(client, run):
    r = client.get("/api/runs", params={"root": str(run)})
    assert r.status_code == 200
    (item,) = r.json()["runs"]
    return item["id"]


def test_api_needs_the_token(cache_dir, tmp_path):
    c = TestClient(create_app("tok", cache_root=cache_dir, static_dir=tmp_path / "s"))
    assert c.get("/api/fs").status_code == 401
    assert c.get("/api/fs", headers={"X-Token": "nope"}).status_code == 401
    assert c.get("/api/fs", headers={"X-Token": "tok"}).status_code == 200
    assert c.get("/api/fs", params={"token": "tok"}).status_code == 200


def test_wrong_host_is_rejected(cache_dir, tmp_path):
    app = create_app("tok", port=8123, cache_root=cache_dir, static_dir=tmp_path / "s")
    c = TestClient(app, base_url="http://127.0.0.1:8123")
    h = {"X-Token": "tok"}
    assert c.get("/api/fs", headers=h).status_code == 200
    assert c.get("/api/fs", headers={**h, "Host": "localhost:8123"}).status_code == 200
    assert c.get("/api/fs", headers={**h, "Host": "evil.example:8123"}).status_code == 403


def test_static_page_and_missing_build(cache_dir, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>caanfast</html>")
    c = TestClient(create_app("tok", cache_root=cache_dir, static_dir=static))
    assert "caanfast" in c.get("/").text
    c2 = TestClient(create_app("tok", cache_root=cache_dir, static_dir=tmp_path / "none"))
    assert "npm run build" in c2.get("/").text


def test_fs_lists_folders_and_flags_runs(client, tiny_run):
    r = client.get("/api/fs", params={"path": str(tiny_run.parent)}).json()
    assert r["path"] == str(tiny_run.parent.resolve())
    names = {e["name"]: e for e in r["entries"]}
    assert names["t"]["is_run"] is True
    up = client.get("/api/fs", params={"path": str(tiny_run.parent.parent)}).json()
    assert {e["name"]: e for e in up["entries"]}["raw"]["has_avi"] is True
    assert client.get("/api/fs", params={"path": str(tiny_run / "nope")}).status_code == 404


def test_summary_image_and_frames(client, tiny_run):
    rid = _open(client, tiny_run)
    s = client.get(f"/api/runs/{rid}/summary").json()
    assert s["T"] == 300 and s["available"]["pre"] is True
    for name in ("max_proj", "background", "rank", "noise"):
        a = _arr(client.get(f"/api/runs/{rid}/image/{name}"))
        assert a.shape == (64, 64) and a.dtype == np.float32
    assert client.get(f"/api/runs/{rid}/image/nope").status_code == 404
    for source in ("raw", "pre", "recon"):
        r = wait_ok(client, f"/api/runs/{rid}/frame/299?source={source}")
        assert r.status_code == 200
        assert _arr(r).shape == (64, 64) and r.headers["X-Dtype"] == "uint8"
    assert client.get(f"/api/runs/{rid}/frame/300?source=recon").status_code == 400
    assert client.get(f"/api/runs/{rid}/frame/-1?source=recon").status_code == 400
    assert client.get(f"/api/runs/{rid}/frame/0?source=movie").status_code == 400
    assert client.get("/api/runs/000000000000/summary").status_code == 404


def test_unplugged_raw_and_user_fps(client, run_copy):
    pj = json.loads((run_copy / "params.json").read_text())
    pj["raw_dir"] = str(run_copy / "unplugged")
    (run_copy / "params.json").write_text(json.dumps(pj))
    rid = _open(client, run_copy)
    r = client.get(f"/api/runs/{rid}/frame/0?source=raw")
    assert r.status_code == 404 and "raw videos not found" in r.json()["detail"]
    assert client.get(f"/api/runs/{rid}/summary").json()["fps"] is None
    assert client.post(f"/api/runs/{rid}/fps", json={"fps": 20}).status_code == 200
    assert client.get(f"/api/runs/{rid}/summary").json()["fps"] == 20.0
    assert client.post(f"/api/runs/{rid}/fps", json={"fps": -1}).status_code == 400
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve.app'`.

- [ ] **Step 4: Implement `app.py`**

```python
"""``create_app``: the FastAPI app of ``caanfast serve`` (guard, state, static files)."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from caanfast.serve.cache import Background
from caanfast.serve.runs import Registry

if TYPE_CHECKING:
    from caanfast.serve.jobs import JobManager

STATIC_DIR = Path(__file__).with_name("static")

NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>CaAnFast</title>
<p>The CaAnFast front end is not built. In the repository run
<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, then restart
<code>caanfast serve</code>.</p>"""


def create_app(
    token: str,
    port: int | None = None,
    cache_root: Path | None = None,
    static_dir: Path | None = None,
    jobs: JobManager | None = None,
) -> FastAPI:
    """The app; ``port`` enables the Host check (None in tests, where Host is testserver)."""
    from caanfast.serve.routes_jobs import router as jobs_router
    from caanfast.serve.routes_runs import router as runs_router

    app = FastAPI(title="CaAnFast", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = Registry(cache_root)
    app.state.frames = {}
    app.state.background = Background()
    app.state.jobs = jobs
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"} if port else None

    @app.middleware("http")
    async def guard(request: Request, call_next):  # noqa: ANN202
        if hosts is not None and request.headers.get("host", "") not in hosts:
            return PlainTextResponse("wrong host", status_code=403)
        if request.url.path.startswith("/api/"):
            given = request.headers.get("x-token") or request.query_params.get("token") or ""
            if not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    app.include_router(runs_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")

    static = Path(static_dir or STATIC_DIR)
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        def not_built() -> str:
            return NOT_BUILT

    return app
```

- [ ] **Step 5: Implement `routes_runs.py` (this task's routes)**

```python
"""Routes for folders, runs and their stage data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from caanfast.serve.arrays import array_response
from caanfast.serve.frames import SOURCES, FrameSource, tile_map
from caanfast.serve.runs import RunView, find_runs, is_run

router = APIRouter()

IMAGES = ("max_proj", "background", "rank", "noise")


def _run(request: Request, rid: str) -> RunView:
    try:
        return request.app.state.registry.get(rid)
    except KeyError:
        raise HTTPException(404, f"unknown run {rid}") from None


def _frames(request: Request, run: RunView) -> FrameSource:
    fr = request.app.state.frames
    if run.id not in fr:
        fr[run.id] = FrameSource(run)
    return fr[run.id]


def computing(request: Request, key: str, fn: Callable[[], None]) -> JSONResponse:
    """202 while ``fn`` runs in the background (500 if it failed)."""
    st = request.app.state.background.ensure(key, fn)
    return JSONResponse(st, status_code=500 if st["status"] == "failed" else 202)


@router.get("/fs")
def fs(path: str | None = None) -> dict[str, Any]:
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        raise HTTPException(404, f"not a folder: {p}")
    p = p.resolve()
    entries = []
    try:
        children = sorted(c for c in p.iterdir() if c.is_dir() and not c.name.startswith("."))
    except OSError as e:
        raise HTTPException(403, f"cannot list {p}: {e}") from None
    for c in children:
        try:
            has_avi = any(c.glob("*.avi"))
        except OSError:
            continue
        entries.append({"name": c.name, "path": str(c), "has_avi": has_avi, "is_run": is_run(c)})
    return {"path": str(p), "parent": str(p.parent), "entries": entries}


@router.get("/runs")
def runs(request: Request, root: str | None = None) -> dict[str, Any]:
    base = Path(root).expanduser() if root else Path.home()
    if not base.is_dir():
        raise HTTPException(404, f"not a folder: {base}")
    reg = request.app.state.registry
    out = []
    for p in find_runs(base):
        try:
            out.append(reg.open(p).listing())
        except (OSError, ValueError, KeyError) as e:
            out.append({"path": str(p), "name": p.name, "error": f"{type(e).__name__}: {e}"})
    return {"root": str(base.resolve()), "runs": out}


@router.get("/runs/{rid}/summary")
def summary(request: Request, rid: str) -> dict[str, Any]:
    return _run(request, rid).summary()


class FpsBody(BaseModel):
    fps: float


@router.post("/runs/{rid}/fps")
def set_fps(request: Request, rid: str, body: FpsBody) -> dict[str, Any]:
    run = _run(request, rid)
    try:
        run.set_fps(body.fps)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"fps": run.fps}


@router.get("/runs/{rid}/image/{name}")
def image(request: Request, rid: str, name: str) -> Response:
    run = _run(request, rid)
    if name not in IMAGES:
        raise HTTPException(404, f"unknown image {name!r} (one of {', '.join(IMAGES)})")
    if name == "max_proj":
        a = np.load(run.path / "factors" / "max_proj.npy")
    elif name == "background":
        a = np.load(run.path / "cnmf" / "b.npy")
    elif name == "rank":
        a = tile_map(run.factors, run.ranks)
    else:
        a = run.factors.sigma_flat().reshape(run.H, run.W)
    return array_response(np.asarray(a, np.float32))


@router.get("/runs/{rid}/frame/{t}", response_model=None)
def frame(request: Request, rid: str, t: int, source: str = "recon") -> Response:
    run = _run(request, rid)
    if source not in SOURCES:
        raise HTTPException(400, f"unknown source {source!r} (one of {', '.join(SOURCES)})")
    fs_ = _frames(request, run)
    if not 0 <= t < run.T:
        raise HTTPException(400, f"frame {t} out of range 0..{run.T - 1}")
    if not fs_.available(source):
        try:
            fs_.frame(source, t)
        except LookupError as e:
            raise HTTPException(404, str(e)) from None
    if not fs_.has_contrast(source):
        return computing(request, f"{rid}:contrast:{source}", lambda: fs_.contrast(source))
    return array_response(fs_.frame(source, t))
```

Create a placeholder-free jobs router so the import works (its routes come in Task 10) — `src/caanfast/serve/routes_jobs.py`:

```python
"""Routes for starting and following runs (filled in by the job runner)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_app.py -v`
Expected: 7 passed.

- [ ] **Step 7: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/app.py src/caanfast/serve/routes_runs.py src/caanfast/serve/routes_jobs.py tests/conftest.py tests/test_serve_app.py
git commit -m "Add the guarded API app with folder, run, image and frame routes"
```

---

### Task 9: Stage routes and parameter metadata

**Files:**
- Create: `src/caanfast/serve/paramsmeta.py`
- Modify: `src/caanfast/serve/routes_runs.py` (append routes)
- Test: `tests/test_serve_stages.py`

**Interfaces:**
- Consumes: Tasks 5, 6, 7, 8 (`_run`, `computing`, `derived.*`, `unit_polygons`, `zooms`, `window`).
- Produces (`paramsmeta.py`): `GROUPS: dict[str, list[str]]`, `COMMON: list[str]`, `HELP: dict[str, str]`, `defaults() -> list[dict]` (keys `name, group, type, default, help, common`).
- Produces routes: `GET /api/runs/{rid}/motion` → float32 `(T, 3)` = dy, dx, quality; `GET /api/runs/{rid}/motion/residual` → float32 `(n, 3)` = dy, dx, corr (202 first); `GET /api/runs/{rid}/seeds` → JSON `{"init"|"pnr"|"merged": [[h, w], …]}` (202 for older runs); `GET /api/runs/{rid}/units?stage=final|init` → JSON `{"stage", "units": [{"id", "centroid": [h, w], "size_px", "peak" | null, "polygons"}]}`; `GET /api/runs/{rid}/traces/overview?kind=C&bins=2000` → float32 `(n, 2, bins)`; `GET /api/runs/{rid}/traces/{unit}?start=&stop=&kind=C&lowpass_hz=` → float32 `(stop-start,)`; `GET /api/runs/{rid}/zooms` → JSON; `GET /api/params/defaults` → `{"groups": [...], "params": [...]}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_stages.py`:

```python
import shutil

import numpy as np

from caanfast.params import Params
from caanfast.serve.arrays import decode
from caanfast.serve.paramsmeta import COMMON, GROUPS, HELP, defaults
from tests.conftest import wait_ok


def _arr(r):
    return decode(r.content, r.headers["X-Shape"], r.headers["X-Dtype"])


def _open(client, run):
    return client.get("/api/runs", params={"root": str(run)}).json()["runs"][0]["id"]


def test_params_metadata_covers_every_parameter():
    names = [n for g in GROUPS.values() for n in g]
    assert sorted(names) == sorted(Params().asdict()) and len(names) == len(set(names))
    assert set(HELP) == set(names) and set(COMMON) <= set(names)
    d = {p["name"]: p for p in defaults()}
    assert d["size_thres"]["default"] == [25, None] and d["workers"]["common"] is True


def test_params_defaults_route(client):
    body = client.get("/api/params/defaults").json()
    assert [g for g in body["groups"]] == list(GROUPS)
    assert len(body["params"]) == len(Params().asdict())


def test_motion_and_residual(client, tiny_run):
    rid = _open(client, tiny_run)
    m = _arr(client.get(f"/api/runs/{rid}/motion"))
    assert m.shape == (300, 3)
    r = wait_ok(client, f"/api/runs/{rid}/motion/residual")
    assert r.status_code == 200 and _arr(r).shape[1] == 3


def test_seeds_and_units(client, tiny_run):
    rid = _open(client, tiny_run)
    s = client.get(f"/api/runs/{rid}/seeds").json()
    assert set(s) == {"init", "pnr", "merged"} and len(s["merged"][0]) == 2
    fin = client.get(f"/api/runs/{rid}/units").json()
    n = len(fin["units"])
    assert fin["stage"] == "final" and n > 0
    u = fin["units"][0]
    assert set(u) == {"id", "centroid", "size_px", "peak", "polygons"}
    ini = client.get(f"/api/runs/{rid}/units?stage=init").json()
    assert ini["units"][0]["peak"] is None
    assert client.get(f"/api/runs/{rid}/units?stage=nope").status_code == 400


def test_seeds_of_an_older_run_answer_202_then_200(client, run_copy):
    shutil.rmtree(run_copy / "seeds")
    shutil.rmtree(run_copy / "init")
    rid = _open(client, run_copy)
    assert client.get(f"/api/runs/{rid}/seeds").status_code == 202
    assert wait_ok(client, f"/api/runs/{rid}/seeds").status_code == 200
    assert wait_ok(client, f"/api/runs/{rid}/units?stage=init").status_code == 200


def test_traces(client, tiny_run):
    rid = _open(client, tiny_run)
    n = len(client.get(f"/api/runs/{rid}/units").json()["units"])
    ov = wait_ok(client, f"/api/runs/{rid}/traces/overview?kind=C&bins=50")
    assert _arr(ov).shape == (n, 2, 50)
    full = _arr(client.get(f"/api/runs/{rid}/traces/{n - 1}"))
    assert full.shape == (300,)
    part = _arr(client.get(f"/api/runs/{rid}/traces/0?start=100&stop=200&lowpass_hz=5"))
    assert part.shape == (100,) and np.isfinite(part).all()
    yra = _arr(client.get(f"/api/runs/{rid}/traces/0?kind=YrA"))
    assert yra.shape == (300,)
    assert client.get(f"/api/runs/{rid}/traces/{n}").status_code == 400
    assert client.get(f"/api/runs/{rid}/traces/0?start=0&stop=301").status_code == 400
    assert client.get(f"/api/runs/{rid}/traces/0?kind=S").status_code == 400
    z = wait_ok(client, f"/api/runs/{rid}/zooms").json()
    assert len(z["units"]) == min(10, n) and z["windows"][0]["start"] == 0
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve.paramsmeta'`.

- [ ] **Step 3: Implement `paramsmeta.py`**

```python
"""Parameter groups and help text for the launcher (one entry per ``Params`` field)."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

from caanfast.params import Params

GROUPS: dict[str, list[str]] = {
    "Preprocessing + motion": [
        "median_ksize",
        "tophat_wnd",
        "motion_max_shift",
        "motion_down",
        "template_frames",
        "motion_before_tophat",
        "motion_bandpass",
    ],
    "Tiling + compression": ["tile", "svd_k0", "svd_power_iter", "rank_rho_min", "rank_z"],
    "Seeds": [
        "seeds_wnd",
        "seeds_stp",
        "seeds_max_wnd",
        "seeds_diff_thres",
        "noise_freq",
        "pnr_thres",
        "gmm_refine",
        "merge_dist",
        "merge_corr",
    ],
    "Initialization": ["inita_corr", "inita_wnd", "init_merge_corr", "loop_merge_corr"],
    "CNMF": ["dl_wnd", "sparse_snr", "size_thres", "n_iter", "background_clip"],
    "Execution": ["workers", "sub_batch"],
}

COMMON = ["workers", "tophat_wnd", "merge_dist", "pnr_thres", "sparse_snr"]

HELP: dict[str, str] = {
    "median_ksize": "Median filter size in px before motion correction (<= 1 skips it).",
    "tophat_wnd": "Tophat window in px that removes the background glow; about one cell diameter.",
    "motion_max_shift": "Largest motion correction shift in px.",
    "motion_down": "Downsampling factor of the frames used to estimate motion.",
    "template_frames": "Frames averaged into the motion template.",
    "motion_before_tophat": "Estimate motion before the tophat, on band-passed frames.",
    "motion_bandpass": "Difference-of-Gaussians sigmas (px) of the band-pass for motion.",
    "tile": "Tile side in px for the low-rank compression.",
    "svd_k0": "Starting rank of each tile's randomized SVD.",
    "svd_power_iter": "Power iterations of the randomized SVD.",
    "rank_rho_min": "Smallest correlation a component needs to be kept.",
    "rank_z": "z-score a component's energy needs over noise to be kept.",
    "seeds_wnd": "Frames per window when picking seed pixels.",
    "seeds_stp": "Step in frames between seed windows.",
    "seeds_max_wnd": "Window in px of the local-maximum search for seeds.",
    "seeds_diff_thres": "Smallest brightness difference that makes a local maximum a seed.",
    "noise_freq": "Frequency (fraction of the frame rate) above which a trace counts as noise.",
    "pnr_thres": "Peak-to-noise threshold for seeds; 'auto' picks it from the data.",
    "gmm_refine": "Also filter seeds with a 2-component Gaussian mixture (off by default).",
    "merge_dist": "Seeds closer than this (px) can merge.",
    "merge_corr": "Seeds whose traces correlate above this merge.",
    "inita_corr": "Correlation a pixel needs with its seed to join the initial footprint.",
    "inita_wnd": "Window in px around a seed for its initial footprint.",
    "init_merge_corr": "Initial units whose traces correlate above this merge.",
    "loop_merge_corr": "Units whose traces correlate above this merge during CNMF.",
    "dl_wnd": "Dilation window in px that bounds where a footprint may grow.",
    "sparse_snr": "Per-pixel signal-to-noise a pixel needs to stay in a footprint.",
    "size_thres": "Smallest and largest footprint size in px (empty = no limit).",
    "n_iter": "CNMF iterations (spatial + temporal update).",
    "background_clip": "Background of the clipped residual as Minian (about 2x slower).",
    "workers": "Worker processes; about the number of CPU cores minus one or two.",
    "sub_batch": "Units per batch in the temporal update.",
}


def defaults() -> list[dict[str, Any]]:
    d = Params().asdict()
    types = {f.name: str(f.type) for f in fields(Params)}
    return [
        {
            "name": name,
            "group": group,
            "type": types[name],
            "default": json.loads(json.dumps(d[name])),
            "help": HELP[name],
            "common": name in COMMON,
        }
        for group, names in GROUPS.items()
        for name in names
    ]
```

- [ ] **Step 4: Append the stage routes to `routes_runs.py`**

Add these imports at the top of `routes_runs.py`:

```python
import pandas as pd

from caanfast import init
from caanfast.serve import derived
from caanfast.serve.contours import unit_polygons
from caanfast.serve.paramsmeta import GROUPS, defaults
from caanfast.serve.traces import window, zooms
```

and append:

```python
@router.get("/params/defaults")
def params_defaults() -> dict[str, Any]:
    return {"groups": list(GROUPS), "params": defaults()}


@router.get("/runs/{rid}/motion")
def motion_(request: Request, rid: str) -> Response:
    sc = _run(request, rid).sidecar
    q = np.asarray(sc.quality, np.float32).reshape(-1, 1)
    return array_response(np.hstack([np.asarray(sc.shifts, np.float32), q]))


@router.get("/runs/{rid}/motion/residual", response_model=None)
def motion_residual(request: Request, rid: str) -> Response:
    run = _run(request, rid)
    r = derived.residual(run)
    if r is None:
        return computing(request, f"{rid}:residual", lambda: derived.compute_residual(run))
    return array_response(r)


@router.get("/runs/{rid}/seeds", response_model=None)
def seeds_(request: Request, rid: str) -> Response | dict[str, Any]:
    run = _run(request, rid)
    s = derived.seeds(run)
    if s is None:
        return computing(request, f"{rid}:seeds", lambda: derived.compute_seeds_and_init(run))
    return {k: np.round(v, 1).tolist() for k, v in s.items()}


@router.get("/runs/{rid}/units", response_model=None)
def units(request: Request, rid: str, stage: str = "final") -> Response | dict[str, Any]:
    run = _run(request, rid)
    if stage == "final":
        A = run.A
        peak = pd.read_csv(run.path / "cnmf" / "units.csv")["peak"].to_numpy()
    elif stage == "init":
        A = derived.init_A(run)
        if A is None:
            return computing(request, f"{rid}:seeds", lambda: derived.compute_seeds_and_init(run))
        peak = None
    else:
        raise HTTPException(400, f"stage must be final or init, got {stage!r}")
    cent = init.centroids(A, run.H, run.W) if A.shape[0] else np.zeros((0, 2))
    polys = unit_polygons(A, run.H, run.W)
    sizes = np.diff(A.indptr)
    return {
        "stage": stage,
        "units": [
            {
                "id": k,
                "centroid": [round(float(cent[k, 0]), 2), round(float(cent[k, 1]), 2)],
                "size_px": int(sizes[k]),
                "peak": None if peak is None else float(peak[k]),
                "polygons": polys[k],
            }
            for k in range(A.shape[0])
        ],
    }


def _kind(kind: str) -> str:
    if kind not in ("C", "YrA"):
        raise HTTPException(400, f"kind must be C or YrA, got {kind!r}")
    return kind


@router.get("/runs/{rid}/traces/overview", response_model=None)
def traces_overview(request: Request, rid: str, kind: str = "C", bins: int = 2000) -> Response:
    run = _run(request, rid)
    kind = _kind(kind)
    if not 1 <= bins <= 20000:
        raise HTTPException(400, "bins must be in 1..20000")
    ov = derived.traces_overview(run, kind, bins)
    if ov is None:
        key = f"{rid}:overview:{kind}:{bins}"
        return computing(request, key, lambda: derived.compute_overview(run, kind, bins))
    return array_response(ov)


@router.get("/runs/{rid}/traces/{unit}")
def trace(
    request: Request,
    rid: str,
    unit: int,
    start: int = 0,
    stop: int | None = None,
    kind: str = "C",
    lowpass_hz: float | None = None,
) -> Response:
    run = _run(request, rid)
    C = run.traces(_kind(kind))
    if not 0 <= unit < C.shape[0]:
        raise HTTPException(400, f"unit {unit} out of range 0..{C.shape[0] - 1}")
    stop = run.T if stop is None else stop
    try:
        y = window(C[unit], start, stop, fs=run.fps, lowpass_hz=lowpass_hz)
    except (IndexError, ValueError) as e:
        raise HTTPException(400, str(e)) from None
    return array_response(y)


@router.get("/runs/{rid}/zooms", response_model=None)
def zooms_(request: Request, rid: str) -> Response | dict[str, Any]:
    run = _run(request, rid)
    ov = derived.traces_overview(run, "C", 2000)
    if ov is None:
        key = f"{rid}:overview:C:2000"
        return computing(request, key, lambda: derived.compute_overview(run, "C", 2000))
    return zooms(ov[:, 1].max(axis=1), run.T, run.fps)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_stages.py tests/test_serve_app.py -v`
Expected: all pass.

- [ ] **Step 6: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/paramsmeta.py src/caanfast/serve/routes_runs.py tests/test_serve_stages.py
git commit -m "Serve motion, residuals, seeds, units, traces, zooms and the parameter list"
```

---

### Task 10: Job runner and job routes

**Files:**
- Create: `src/caanfast/serve/jobs.py`
- Modify: `src/caanfast/serve/routes_jobs.py`
- Test: `tests/test_serve_jobs.py`

**Interfaces:**
- Consumes: `pipeline.out_dir_for`, `pipeline.check_run_name`, `Params.from_json`, `runs.is_run`, `python -m caanfast run` (Task 1), `progress.jsonl` (Task 2).
- Produces: `FINAL = {"done", "failed", "cancelled", "interrupted"}`; `Job` dataclass (`id, raw_dir, name, params, keep_store, status, out_dir, created, started, ended, error`); `JobManager(root: Path | None = None, command: Callable[[Job, Path], list[str]] | None = None)` with `.submit(raw_dir: str, name: str, params: dict, keep_store=False, overwrite=False) -> Job` (raises `ValueError`), `.get(jid) -> Job` (raises `KeyError`), `.list() -> list[Job]`, `.cancel(jid) -> Job`, `.progress(job) -> list[dict]`, `.log_lines(jid) -> list[str]`, `.wait(jid, timeout) -> Job`.
- Produces route `GET /api/raw/info?path=` → `{"path", "segments", "frames", "height", "width", "fps", "bytes"}` (404 when the folder has no `.avi`), for the launcher's first step.
- Produces routes: `GET /api/jobs`, `POST /api/jobs` (body `{"raw_dir", "name", "params", "keep_store", "overwrite"}` → job dict; 400 on `ValueError`), `GET /api/jobs/{jid}`, `POST /api/jobs/{jid}/cancel`, `GET /api/jobs/{jid}/events` (server-sent events `stage`, `log`, `status`).

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_jobs.py`:

```python
import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from caanfast.serve.app import create_app
from caanfast.serve.jobs import JobManager
from caanfast.testing import synth
from tests.conftest import TINY_PARAMS


def _sleeper(job, params_file):
    return [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "my session" / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    return raw


def test_submit_rejects_bad_input(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    with pytest.raises(ValueError, match="no .avi"):
        jm.submit(str(tmp_path), "x", {})
    with pytest.raises(ValueError, match="unknown parameter"):
        jm.submit(str(raw), "x", {"bogus": 1})
    with pytest.raises(ValueError):
        jm.submit(str(raw), "../escape", {})


def test_a_real_run_end_to_end_and_overwrite_guard(tmp_path, raw):
    jm = JobManager(tmp_path / "cache")
    job = jm.submit(str(raw), "t", {**TINY_PARAMS, "size_thres": [9, None]})
    done = jm.wait(job.id, timeout=300)
    assert done.status == "done", jm.log_lines(job.id)[-20:]
    ends = [e["stage"] for e in jm.progress(done) if e["event"] == "end"]
    assert "cnmf" in ends
    assert (raw.parent / "processed" / "t" / "report.json").is_file()
    with pytest.raises(ValueError, match="already holds a run"):
        jm.submit(str(raw), "t", {})
    again = jm.submit(str(raw), "t", {**TINY_PARAMS, "size_thres": [9, None]}, overwrite=True)
    assert jm.wait(again.id, timeout=300).status == "done"


def test_cancel_and_queue(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    a = jm.submit(str(raw), "a", {})
    b = jm.submit(str(raw), "b", {})
    t0 = time.time()
    while jm.get(a.id).status != "running":
        assert time.time() - t0 < 10
        time.sleep(0.05)
    assert jm.get(b.id).status == "queued"
    assert jm.cancel(b.id).status == "cancelled"
    jm.cancel(a.id)
    assert jm.wait(a.id, timeout=10).status == "cancelled"


def test_running_jobs_are_interrupted_after_a_restart(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "j1",
                    "raw_dir": "/r",
                    "name": "n",
                    "params": {},
                    "keep_store": False,
                    "status": "running",
                    "out_dir": "/o",
                    "created": 0.0,
                    "started": 1.0,
                    "ended": None,
                    "error": "",
                }
            ]
        )
    )
    assert JobManager(root, command=_sleeper).get("j1").status == "interrupted"


def test_job_routes_and_events(tmp_path, raw):
    jm = JobManager(tmp_path / "cache", command=_sleeper)
    c = TestClient(create_app("tok", cache_root=tmp_path / "cache", jobs=jm))
    c.headers["X-Token"] = "tok"
    r = c.post("/api/jobs", json={"raw_dir": str(raw), "name": "a", "params": {}})
    assert r.status_code == 200
    jid = r.json()["id"]
    assert c.post("/api/jobs", json={"raw_dir": "/nope", "name": "a", "params": {}}).status_code == 400
    assert [j["id"] for j in c.get("/api/jobs").json()["jobs"]] == [jid]
    assert c.post(f"/api/jobs/{jid}/cancel").json()["status"] in ("cancelled", "running")
    jm.wait(jid, timeout=10)
    with c.stream("GET", f"/api/jobs/{jid}/events") as s:
        text = "".join(s.iter_text())
    assert "event: status" in text and '"cancelled"' in text
    assert c.get("/api/jobs/nope").status_code == 404


def test_raw_info(tmp_path, raw):
    c = TestClient(create_app("tok", cache_root=tmp_path / "cache"))
    c.headers["X-Token"] = "tok"
    info = c.get("/api/raw/info", params={"path": str(raw)}).json()
    assert (info["segments"], info["frames"], info["height"], info["width"]) == (3, 300, 64, 64)
    assert info["fps"] == 30.0 and info["bytes"] > 0
    assert c.get("/api/raw/info", params={"path": str(tmp_path)}).status_code == 404
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve.jobs'`.

- [ ] **Step 3: Implement `jobs.py`**

```python
"""Run queue of ``caanfast serve``: one ``caanfast run`` subprocess at a time."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from caanfast import pipeline
from caanfast.params import Params
from caanfast.serve.cache import cache_root
from caanfast.serve.runs import is_run

FINAL = {"done", "failed", "cancelled", "interrupted"}


@dataclass
class Job:
    id: str
    raw_dir: str
    name: str
    params: dict[str, Any]
    keep_store: bool
    status: str = "queued"  # queued | running | done | failed | cancelled | interrupted
    out_dir: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    error: str = ""


def default_command(job: Job, params_file: Path) -> list[str]:
    cmd = [sys.executable, "-m", "caanfast", "run", job.raw_dir]
    cmd += ["--params", str(params_file), "--name", job.name]
    return cmd + (["--keep-store"] if job.keep_store else [])


class JobManager:
    """Queue persisted in ``<root>/jobs.json``; logs and params files in ``<root>/jobs/``."""

    def __init__(
        self,
        root: Path | None = None,
        command: Callable[[Job, Path], list[str]] | None = None,
    ) -> None:
        self.root = Path(root or cache_root())
        self.dir = self.root / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.command = command or default_command
        self._cv = threading.Condition()
        self._proc: subprocess.Popen | None = None
        self._cancel: set[str] = set()
        self.jobs: dict[str, Job] = {}
        f = self.root / "jobs.json"
        if f.exists():
            for d in json.loads(f.read_text()):
                job = Job(**d)
                if job.status == "running":
                    job.status, job.error = "interrupted", "the server stopped during the run"
                self.jobs[job.id] = job
        self._save()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- public ---------------------------------------------------------------

    def submit(
        self,
        raw_dir: str,
        name: str,
        params: dict[str, Any],
        keep_store: bool = False,
        overwrite: bool = False,
    ) -> Job:
        raw = Path(raw_dir).expanduser()
        if not raw.is_dir() or not any(raw.glob("*.avi")):
            raise ValueError(f"no .avi files in {raw}")
        pipeline.check_run_name(name)
        jid = uuid.uuid4().hex[:10]
        pfile = self.dir / f"{jid}.params.json"
        pfile.write_text(json.dumps(params))
        Params.from_json(pfile)  # raises ValueError on unknown or invalid parameters
        out = pipeline.out_dir_for(raw, name)
        if is_run(out) and not overwrite:
            raise ValueError(f"{out} already holds a run; choose another name or overwrite")
        job = Job(jid, str(raw), name, params, keep_store, out_dir=str(out))
        with self._cv:
            self.jobs[jid] = job
            self._save()
            self._cv.notify_all()
        return job

    def get(self, jid: str) -> Job:
        return self.jobs[jid]

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, jid: str) -> Job:
        with self._cv:
            job = self.jobs[jid]
            if job.status == "queued":
                job.status, job.ended = "cancelled", time.time()
                self._save()
            elif job.status == "running" and self._proc is not None:
                self._cancel.add(jid)
                try:
                    os.killpg(self._proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self._cv.notify_all()
            return job

    def wait(self, jid: str, timeout: float) -> Job:
        t0 = time.time()
        with self._cv:
            while self.jobs[jid].status not in FINAL:
                left = timeout - (time.time() - t0)
                if left <= 0:
                    raise TimeoutError(f"job {jid} still {self.jobs[jid].status}")
                self._cv.wait(min(left, 0.5))
            return self.jobs[jid]

    def log_path(self, jid: str) -> Path:
        return self.dir / f"{jid}.log"

    def log_lines(self, jid: str) -> list[str]:
        p = self.log_path(jid)
        return p.read_text(errors="replace").splitlines() if p.exists() else []

    def progress(self, job: Job) -> list[dict[str, Any]]:
        """Stage events of this job's run (older lines of a previous run are skipped)."""
        p = Path(job.out_dir) / "progress.jsonl"
        if job.started is None or not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written
            if e.get("t", 0) >= job.started - 1:
                out.append(e)
        return out

    # -- worker -----------------------------------------------------------------

    def _save(self) -> None:
        tmp = self.root / "jobs.json.tmp"
        tmp.write_text(json.dumps([asdict(j) for j in self.jobs.values()], indent=1))
        tmp.replace(self.root / "jobs.json")

    def _next(self) -> Job:
        with self._cv:
            while True:
                queued = [j for j in self.jobs.values() if j.status == "queued"]
                if queued:
                    job = min(queued, key=lambda j: j.created)
                    job.status, job.started = "running", time.time()
                    self._save()
                    return job
                self._cv.wait()

    def _worker(self) -> None:
        while True:
            job = self._next()
            pfile = self.dir / f"{job.id}.params.json"
            with self.log_path(job.id).open("wb") as log:
                proc = subprocess.Popen(
                    self.command(job, pfile),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with self._cv:
                    self._proc = proc
                rc = proc.wait()
            with self._cv:
                self._proc = None
                job.ended = time.time()
                if job.id in self._cancel:
                    job.status = "cancelled"
                elif rc == 0:
                    job.status = "done"
                else:
                    job.status = "failed"
                    job.error = "\n".join(self.log_lines(job.id)[-20:])
                self._save()
                self._cv.notify_all()
```

- [ ] **Step 4: Implement the job routes**

Replace `src/caanfast/serve/routes_jobs.py` with:

```python
"""Routes for starting and following runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from caanfast import io
from caanfast.serve.jobs import FINAL, Job, JobManager
from caanfast.serve.runs import fps_from_files

router = APIRouter()


def _jobs(request: Request) -> JobManager:
    jm = request.app.state.jobs
    if jm is None:
        raise HTTPException(503, "the job runner is not enabled")
    return jm


def _job(request: Request, jid: str) -> Job:
    try:
        return _jobs(request).get(jid)
    except KeyError:
        raise HTTPException(404, f"unknown job {jid}") from None


class JobBody(BaseModel):
    raw_dir: str
    name: str
    params: dict[str, Any] = {}
    keep_store: bool = False
    overwrite: bool = False


@router.get("/raw/info")
def raw_info(path: str) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_dir() or not any(raw.glob("*.avi")):
        raise HTTPException(404, f"no .avi files in {raw}")
    try:
        info = io.probe(raw)
    except OSError as e:
        raise HTTPException(400, f"cannot read the videos in {raw}: {e}") from None
    return {
        "path": str(raw.resolve()),
        "segments": len(info.segments),
        "frames": info.frames,
        "height": info.height,
        "width": info.width,
        "fps": fps_from_files(raw),
        "bytes": int(sum(p.stat().st_size for p in info.segments)),
    }


@router.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    return {"jobs": [asdict(j) for j in _jobs(request).list()]}


@router.post("/jobs")
def submit(request: Request, body: JobBody) -> dict[str, Any]:
    try:
        job = _jobs(request).submit(
            body.raw_dir, body.name, body.params, body.keep_store, body.overwrite
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return asdict(job)


@router.get("/jobs/{jid}")
def get_job(request: Request, jid: str) -> dict[str, Any]:
    return asdict(_job(request, jid))


@router.post("/jobs/{jid}/cancel")
def cancel(request: Request, jid: str) -> dict[str, Any]:
    _job(request, jid)
    return asdict(_jobs(request).cancel(jid))


@router.get("/jobs/{jid}/events")
async def events(request: Request, jid: str) -> StreamingResponse:
    jm = _jobs(request)
    _job(request, jid)

    async def gen():  # noqa: ANN202
        n_prog = n_log = 0
        last = None
        while True:
            job = jm.get(jid)
            prog = jm.progress(job)
            for e in prog[n_prog:]:
                yield f"event: stage\ndata: {json.dumps(e)}\n\n"
            n_prog = len(prog)
            lines = jm.log_lines(jid)
            for line in lines[n_log:]:
                yield f"event: log\ndata: {json.dumps(line)}\n\n"
            n_log = len(lines)
            if job.status != last:
                last = job.status
                yield f"event: status\ndata: {json.dumps(asdict(job))}\n\n"
            if job.status in FINAL or await request.is_disconnected():
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_serve_jobs.py -v`
Expected: 6 passed (the end-to-end test runs two real tiny runs, ~1 min).

- [ ] **Step 6: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/jobs.py src/caanfast/serve/routes_jobs.py tests/test_serve_jobs.py
git commit -m "Add the run queue with cancel, restart recovery, an overwrite guard and live job events"
```

---

### Task 11: `caanfast serve`

**Files:**
- Create: `src/caanfast/serve/server.py`
- Modify: `src/caanfast/cli.py` (new command), `README.md` (section "Browser viewer")
- Test: `tests/test_serve_server.py`

**Interfaces:**
- Consumes: `create_app` (Task 8), `JobManager` (Task 10).
- Produces: `free_port() -> int`; `start_url(port: int, token: str, folder: Path | None) -> str`; `serve_forever(folder: Path | None, port: int = 0, open_browser: bool = True, token: str | None = None) -> None`; CLI `caanfast serve [FOLDER] [--port N] [--no-browser] [--token T]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_server.py`:

```python
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from typer.testing import CliRunner

from caanfast.cli import app
from caanfast.serve.server import free_port, start_url


def test_start_url_carries_token_and_folder():
    u = urlparse(start_url(8123, "abc", Path("$SSD/x")))
    assert (u.hostname, u.port) == ("127.0.0.1", 8123)
    q = parse_qs(u.query)
    assert q["token"] == ["abc"] and q["root"] == ["$SSD/x"]
    assert "root" not in parse_qs(urlparse(start_url(1, "t", None)).query)


def test_free_port_is_usable():
    assert 1024 < free_port() < 65536


def test_cli_serve_wires_options(monkeypatch, tmp_path):
    seen = {}

    def fake(folder, port=0, open_browser=True, token=None):
        seen.update(folder=folder, port=port, open_browser=open_browser, token=token)

    monkeypatch.setattr("caanfast.serve.server.serve_forever", fake)
    r = CliRunner().invoke(app, ["serve", str(tmp_path), "--port", "9001", "--no-browser"])
    assert r.exit_code == 0, r.output
    assert seen == {"folder": tmp_path, "port": 9001, "open_browser": False, "token": None}
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/pytest tests/test_serve_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caanfast.serve.server'`.

- [ ] **Step 3: Implement `server.py`**

```python
"""Start ``caanfast serve``: free port, random token, uvicorn on 127.0.0.1, open the browser."""

from __future__ import annotations

import secrets
import socket
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_url(port: int, token: str, folder: Path | None) -> str:
    q = {"token": token}
    if folder is not None:
        q["root"] = str(folder)
    return f"http://127.0.0.1:{port}/?{urlencode(q)}"


def serve_forever(
    folder: Path | None, port: int = 0, open_browser: bool = True, token: str | None = None
) -> None:
    import uvicorn

    from caanfast.serve.app import create_app
    from caanfast.serve.jobs import JobManager

    port = port or free_port()
    token = token or secrets.token_urlsafe(16)
    app = create_app(token, port=port, jobs=JobManager())
    url = start_url(port, token, Path(folder).resolve() if folder else None)
    print(f"CaAnFast is running at {url}\nPress Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
```

- [ ] **Step 4: Add the CLI command**

Append to `src/caanfast/cli.py`:

```python
@app.command()
def serve(
    folder: Annotated[
        Path | None, typer.Argument(help="Folder to look for runs in (default: home).")
    ] = None,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 (0 = a free one).")] = 0,
    no_browser: Annotated[bool, typer.Option(help="Do not open the browser.")] = False,
    token: Annotated[str | None, typer.Option(hidden=True)] = None,
) -> None:
    """Open the run viewer and launcher in the browser (local only)."""
    from caanfast.serve import server

    server.serve_forever(folder, port=port, open_browser=not no_browser, token=token)
```

- [ ] **Step 5: Document it**

Add to `README.md` after the section that explains `caanfast run`:

```markdown
## Browser viewer

    caanfast serve [FOLDER]

starts a local server on 127.0.0.1 and opens the browser. It lists the runs under FOLDER
(default: your home folder), shows every stage of a run, and starts new runs. Runs are only
read; results it has to compute for older runs go to `~/.cache/caanfast/` (or
`$CAANFAST_CACHE`). On a remote machine, start it with `--no-browser --port 8765` and open
an SSH tunnel: `ssh -L 8765:127.0.0.1:8765 host`, then the printed URL.
```

- [ ] **Step 6: Run the whole suite and try it by hand**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

Run (manual check, with any run folder, for example the tiny run kept by a test session or `processed/` of a synthetic run):

```bash
.venv/bin/caanfast synth /tmp/caanfast-demo/raw --minutes 0.2
.venv/bin/caanfast run /tmp/caanfast-demo/raw --name demo
.venv/bin/caanfast serve /tmp/caanfast-demo --no-browser --port 8765
```

In a second terminal, with the token the first one printed:

```bash
curl -s -H "X-Token: <token>" "http://127.0.0.1:8765/api/runs?root=/tmp/caanfast-demo"
```

Expected: JSON with one run named `demo`; `http://127.0.0.1:8765/` shows the "front end is not built" page.

- [ ] **Step 7: Format, lint, commit**

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests
git add src/caanfast/serve/server.py src/caanfast/cli.py README.md tests/test_serve_server.py
git commit -m "Add the caanfast serve command"
```
