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
