"""Folders MiniMe has written to (the demo download, sessions with runs), newest first.

Kept in ``$MINIME_CACHE/folders.json`` (default ``~/.cache/minime``) so ``minime serve`` can
open where the data is instead of the home folder. Losing the file only loses the shortcuts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

MAX_PLACES = 20


def places_file() -> Path:
    root = Path(os.environ.get("MINIME_CACHE") or Path.home() / ".cache" / "minime")
    return root / "folders.json"


def _load() -> list[dict[str, Any]]:
    try:
        data = json.loads(places_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return (
        [p for p in data if isinstance(p, dict) and "path" in p] if isinstance(data, list) else []
    )


def remember(folder: Path, kind: str, raw: Path | None = None) -> None:
    """Put ``folder`` first (``kind``: "demo" or "run"; ``raw``: its video folder).

    Never raises: a run or download must not fail over a shortcut list.
    """
    entry = {"path": os.path.abspath(folder), "kind": kind, "time": time.time()}
    if raw is not None:
        entry["raw"] = os.path.abspath(raw)
    rest = [p for p in _load() if p["path"] != entry["path"]]
    dest = places_file()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps([entry, *rest][:MAX_PLACES], indent=1), encoding="utf-8")
        tmp.replace(dest)
    except OSError:
        pass


def known() -> list[dict[str, Any]]:
    """Remembered folders that still exist, newest first (``raw`` dropped if it is gone)."""
    out = []
    for p in _load():
        if not Path(p["path"]).is_dir():
            continue
        if "raw" in p and not Path(p["raw"]).is_dir():
            p = {k: v for k, v in p.items() if k != "raw"}
        out.append(p)
    return out
