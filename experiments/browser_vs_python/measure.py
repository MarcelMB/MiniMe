"""Run a command and sample the resident memory of its process tree.

    python measure.py --out mem.json [--match REGEX] -- CMD ...

Every second the RSS of the command and all its descendants is summed (only processes whose
command name matches REGEX when given, e.g. the Chromium processes of a Playwright runner).
Writes wall time, exit code, peak tree RSS and the sampled series to ``--out``. RSS counts
memory shared between processes once per process, so for Chromium it is an upper bound.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def tree_rss(root: int, match: re.Pattern[str] | None) -> tuple[int, int]:
    """Summed RSS in bytes of ``root`` and its descendants, and how many processes counted."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,comm="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows: dict[int, tuple[int, int, str]] = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        rows[int(parts[0])] = (int(parts[1]), int(parts[2]) * 1024, parts[3])
    children: dict[int, list[int]] = {}
    for pid, (ppid, _, _) in rows.items():
        children.setdefault(ppid, []).append(pid)
    total, n, stack = 0, 0, [root]
    while stack:
        pid = stack.pop()
        stack.extend(children.get(pid, []))
        if pid in rows and (match is None or match.search(rows[pid][2])):
            total += rows[pid][1]
            n += 1
    return total, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", default=None)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    match = re.compile(a.match) if a.match else None
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    series: list[tuple[float, int]] = []
    peak, peak_n = 0, 0
    while proc.poll() is None:
        rss, n = tree_rss(proc.pid, match)
        series.append((round(time.time() - t0, 1), rss))
        if rss > peak:
            peak, peak_n = rss, n
        time.sleep(a.interval)
    wall = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "match": a.match,
                "exit_code": proc.returncode,
                "wall_s": wall,
                "peak_rss_tree_bytes": peak,
                "processes_at_peak": peak_n,
                "series": series,
            }
        )
    )
    print(
        f"exit {proc.returncode}, wall {wall:.0f} s, peak tree RSS {peak / 2**30:.2f} GiB"
    )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
