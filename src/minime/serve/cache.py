"""Per-run cache outside the run folder (``$MINIME_CACHE`` or ``~/.cache/minime``)."""

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
    return Path(os.environ.get("MINIME_CACHE") or Path.home() / ".cache" / "minime")


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

    def _tmp(self, name: str, suffix: str = "") -> Path:
        """A tmp name of its own per process and thread, so concurrent writes never collide."""
        return self.path(f"{name}.{os.getpid()}-{threading.get_ident()}.tmp{suffix}")

    def save_json(self, name: str, doc: Any) -> None:
        tmp = self._tmp(name)
        tmp.write_text(json.dumps(doc))
        tmp.replace(self.dir / name)

    def load_npy(self, name: str) -> np.ndarray:
        return np.load(self.dir / name)

    def save_npy(self, name: str, a: np.ndarray) -> None:
        tmp = self._tmp(name, ".npy")
        np.save(tmp, a)
        tmp.replace(self.dir / name)


class Background:
    """Slow computations run once per key in a daemon thread.

    ``ensure`` starts ``fn`` if it is neither running nor failed and returns the status a
    route sends with ``202``; the route itself checks the cache for the finished result.
    A failure is reported for ``failed_ttl`` seconds, then the next ``ensure`` retries.
    """

    def __init__(self, failed_ttl: float = 60.0) -> None:
        self.failed_ttl = failed_ttl
        self._lock = threading.Lock()
        self._running: dict[str, float] = {}
        self._failed: dict[str, tuple[float, str]] = {}

    def ensure(self, key: str, fn: Callable[[], None]) -> dict[str, Any]:
        with self._lock:
            if key in self._failed:
                t, error = self._failed[key]
                if time.time() - t < self.failed_ttl:
                    return {"status": "failed", "error": error}
                del self._failed[key]
            if key not in self._running:
                self._running[key] = time.time()
                threading.Thread(target=self._run, args=(key, fn), daemon=True).start()
            return {"status": "computing", "elapsed_s": time.time() - self._running[key]}

    def _run(self, key: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - reported to the page, not raised
            with self._lock:
                self._failed[key] = (time.time(), f"{type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._running.pop(key, None)
