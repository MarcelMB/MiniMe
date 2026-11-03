"""Find run folders and open one read-only (``RunView``)."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps

from minime import io, motion, pipeline
from minime.factors import Factors
from minime.params import Params
from minime.serve.cache import RunCache, cache_root
from minime.window2 import RERUNS

MARKERS = ("params.json", "report.json")
SKIP_DIRS = {"Library"}  # macOS ~/Library: huge, and never holds runs
KEEP_LOADED = 2  # runs whose factors, A and motion sidecar stay in memory


def run_id(path: Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]


def is_run(path: Path) -> bool:
    return all((Path(path) / m).is_file() for m in MARKERS)


def find_runs(root: Path, max_depth: int = 4) -> list[Path]:
    """Run folders at or below ``root`` (never inside a run), newest report first.

    Depth 4 reaches ``root/collection/session/processed/run``.
    """
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if is_run(d):
            found.append(d)
            return
        if depth == max_depth:
            return
        try:
            subs = sorted(
                p
                for p in d.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_DIRS
            )
        except OSError:
            return
        for p in subs:
            walk(p, depth + 1)

    walk(Path(root), 0)
    return sorted(found, key=lambda p: (p / "report.json").stat().st_mtime, reverse=True)


def report_stamp(path: Path) -> list[int]:
    st = (Path(path) / "report.json").stat()
    return [st.st_mtime_ns, st.st_size]


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
            raise FileNotFoundError(f"not a minime run: {path}")
        self.id = run_id(self.path)
        self.cache = RunCache(self.id, cache_root)
        self._stamp = report_stamp(self.path)
        self._heavy = threading.Lock()
        self._factors: Factors | None = None
        self._sidecar: motion.MotionSidecar | None = None
        self._A: sps.csr_matrix | None = None
        self._invalidate_stale_cache()
        self.params: dict[str, Any] = json.loads(
            (self.path / "params.json").read_text(encoding="utf-8")
        )
        self.report: dict[str, Any] = json.loads(
            (self.path / "report.json").read_text(encoding="utf-8")
        )
        m = json.loads((self.path / "factors" / "meta.json").read_text(encoding="utf-8"))
        self.H, self.W, self.T = int(m["H"]), int(m["W"]), int(m["T"])
        self.ranks: list[int] = [int(r) for r in m["ranks"]]
        self.tile: int = int(m["tile"])
        raw = self.params.get("raw_dir")
        self.raw_dir: Path | None = Path(raw) if raw else None

    def _invalidate_stale_cache(self) -> None:
        """Drop the cache of a run folder that was rewritten (keeps the user's frame rate)."""
        old = self.cache.load_json("stamp.json")
        if old is not None and old != self._stamp:
            fps = self.cache.load_json("fps.json")
            if self.cache.dir.is_dir():
                shutil.rmtree(self.cache.dir)
            if fps is not None:
                self.cache.save_json("fps.json", fps)
        if old != self._stamp:
            self.cache.save_json("stamp.json", self._stamp)

    @property
    def stamp(self) -> str:
        """Version of the run folder (``report.json`` mtime and size), for cache keys."""
        return "-".join(map(str, self._stamp))

    def is_current(self) -> bool:
        """False once ``report.json`` changed on disk since this view was opened."""
        try:
            return report_stamp(self.path) == self._stamp
        except OSError:
            return False

    @property
    def has_raw(self) -> bool:
        return (
            self.raw_dir is not None and self.raw_dir.is_dir() and any(self.raw_dir.glob("*.avi"))
        )

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
    def has_stages(self) -> bool:
        """A window-2 rerun: seeds and footprints of every step in ``stages/``."""
        return (self.path / "stages" / "index.json").is_file()

    def rerun_dir(self, name: str) -> Path:
        """``reruns/name`` of this run; ``KeyError`` unless it holds a finished rerun."""
        pipeline.check_run_name(name)
        d = self.path / RERUNS / name
        if not is_run(d):
            raise KeyError(name)
        return d

    @property
    def fps(self) -> float | None:
        """Frame rate given when the run started, else from the raw folder's files, else
        the one set in the viewer."""
        if self.params.get("fps"):
            return float(self.params["fps"])
        if self.raw_dir is not None:
            f = io.fps_from_files(self.raw_dir)
            if f:
                return f
        user = self.cache.load_json("fps.json")
        return float(user["fps"]) if user else None

    def set_fps(self, fps: float) -> None:
        self.cache.save_json("fps.json", {"fps": io.check_fps(fps)})

    # The heavy parts load once (also under concurrent requests) and ``release`` drops them.

    @property
    def factors(self) -> Factors:
        with self._heavy:
            if self._factors is None:
                self._factors = Factors.load(self.path / "factors")
            return self._factors

    @property
    def sidecar(self) -> motion.MotionSidecar:
        with self._heavy:
            if self._sidecar is None:
                self._sidecar = motion.load_sidecar(self.path / "motion")
            return self._sidecar

    @property
    def A(self) -> sps.csr_matrix:
        with self._heavy:
            if self._A is None:
                self._A = sps.load_npz(self.path / "cnmf" / "A.npz").tocsr()
            return self._A

    def release(self) -> None:
        """Drop the loaded factors, A and motion sidecar; they load again on next use."""
        with self._heavy:
            self._factors = self._sidecar = self._A = None

    @cached_property
    def raw_info(self) -> io.RawInfo:
        if not self.has_raw:
            raise LookupError(f"raw videos not found in {self.raw_dir}")
        return io.probe(self.raw_dir)

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
    """Open runs by id; ids are remembered in ``<cache root>/runs.json`` across restarts.

    Only the ``KEEP_LOADED`` most recently used runs keep their heavy parts in memory.
    """

    def __init__(self, cache_root_: Path | None = None) -> None:
        self.root = Path(cache_root_ or cache_root())
        self._runs: dict[str, RunView] = {}
        self._recent: list[str] = []  # most recently used first
        self._lock = threading.RLock()

    def recent(self) -> list[str]:
        """Ids of the runs that keep their heavy parts, most recent first."""
        with self._lock:
            return list(self._recent)

    def _touch(self, rid: str) -> None:
        self._recent = [rid] + [r for r in self._recent if r != rid]
        for old in self._recent[KEEP_LOADED:]:
            if old in self._runs:
                self._runs[old].release()
        del self._recent[KEEP_LOADED:]

    def _known(self) -> dict[str, str]:
        p = self.root / "runs.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def open(self, path: Path, touch: bool = True) -> RunView:
        """The view of ``path``; ``touch=False`` (listings) leaves the recent runs alone."""
        with self._lock:
            rid = run_id(path)
            if rid not in self._runs or not self._runs[rid].is_current():
                self._runs[rid] = RunView(path, self.root)
                known = self._known()
                if known.get(rid) != str(self._runs[rid].path):
                    known[rid] = str(self._runs[rid].path)
                    self.root.mkdir(parents=True, exist_ok=True)
                    tmp = self.root / "runs.json.tmp"
                    tmp.write_text(json.dumps(known), encoding="utf-8")
                    tmp.replace(self.root / "runs.json")
            if touch:
                self._touch(rid)
            return self._runs[rid]

    def get(self, rid: str) -> RunView:
        with self._lock:
            run = self._runs.get(rid)
            if run is not None and run.is_current():
                self._touch(rid)
                return run
            path = run.path if run is not None else self._known().get(rid)
            if path is None or not is_run(Path(path)):
                raise KeyError(rid)
            return self.open(Path(path))
