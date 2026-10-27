"""Run report: wall time per stage, counts, and the ``report.json`` writer."""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class StageTimer:
    """``with timer("name"): ...`` appends ``{"stage", "seconds"}`` (wall) to ``rows``.

    ``on_stage(name)``, if given, is called as each stage starts (progress display).
    """

    def __init__(self, on_stage: Callable[[str], None] | None = None) -> None:
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.on_stage = on_stage

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if self.on_stage is not None:
            self.on_stage(stage)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append({"stage": stage, "seconds": time.perf_counter() - t0})

    def to_json(self, path: Path, extra: dict[str, Any]) -> None:
        doc = {"stages": self.rows, "counts": self.counts, **extra}
        Path(path).write_text(json.dumps(doc, indent=1, default=_json_default))


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    return str(o)


def peak_rss(children: bool = False) -> int:
    """Peak resident set size in bytes of this process, or of its largest finished child.

    ``ru_maxrss`` is bytes on macOS and KiB on Linux.
    """
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    rss = resource.getrusage(who).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def dir_bytes(path: Path) -> int:
    """Total size of the regular files under ``path``."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
