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
