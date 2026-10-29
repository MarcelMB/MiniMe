"""Run report: wall time per stage, counts, and the ``report.json`` writer."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class StageTimer:
    """``with timer("name"): ...`` appends ``{"stage", "seconds"}`` (wall) to ``rows``.
    ``on_stage(name)``, if given, is called as each stage starts (progress display).
    After ``attach_progress(path)`` every stage start and end (``error`` if the stage
    raised) is also appended to ``path`` as one JSON line, so another process (the
    ``minime serve`` job runner) can follow.
    """

    def __init__(self, on_stage: Callable[[str], None] | None = None) -> None:
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.on_stage = on_stage
        self.progress: Path | None = None

    def attach_progress(self, path: Path) -> None:
        """Start logging to ``path`` (overwritten); stages already finished come first."""
        self.progress = Path(path)
        now = time.time()
        with self.progress.open("w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps({"stage": r["stage"], "event": "end", "t": now, **r}) + "\n")

    def _log(self, stage: str, event: str, seconds: float | None = None) -> None:
        if self.progress is None:
            return
        doc: dict[str, Any] = {"stage": stage, "event": event, "t": time.time()}
        if seconds is not None:
            doc["seconds"] = seconds
        with self.progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if self.on_stage is not None:
            self.on_stage(stage)
        self._log(stage, "start")
        t0 = time.perf_counter()
        try:
            yield
        except BaseException:
            dt = time.perf_counter() - t0
            self.rows.append({"stage": stage, "seconds": dt})
            self._log(stage, "error", dt)
            raise
        dt = time.perf_counter() - t0
        self.rows.append({"stage": stage, "seconds": dt})
        self._log(stage, "end", dt)

    def to_json(self, path: Path, extra: dict[str, Any]) -> None:
        doc = {"stages": self.rows, "counts": self.counts, **extra}
        Path(path).write_text(json.dumps(doc, indent=1, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    return str(o)


def peak_rss(children: bool = False) -> int | None:
    """Peak resident set size in bytes of this process, or of its largest finished child.

    ``ru_maxrss`` is bytes on macOS and KiB on Linux. Windows has no ``resource``: the
    peak working set of this process, and None for the children.
    """
    if sys.platform == "win32":
        return None if children else _peak_working_set()
    import resource

    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    rss = resource.getrusage(who).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _peak_working_set() -> int | None:
    """``PeakWorkingSetSize`` of this process (Windows), None if the call fails."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t)
            for n in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    c = Counters(cb=ctypes.sizeof(Counters))
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return int(c.PeakWorkingSetSize) if ok else None


def dir_bytes(path: Path) -> int:
    """Total size of the regular files under ``path``."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
