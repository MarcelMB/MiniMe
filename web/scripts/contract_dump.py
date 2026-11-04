"""Ask main's ``minime serve`` routes every run-viewer request on one run folder and save the
answers, so the browser router can be checked against them (``testing/contract.ts``).

    python web/scripts/contract_dump.py RUN_DIR OUT_DIR [--raw RAW_DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from minime.serve.app import create_app
from minime.serve.arrays import decode

TOKEN = "contract"


def requests(
    rid: str, raw: Path | None, n_units: int, T: int
) -> list[tuple[str, str, dict | None]]:
    r = f"/runs/{rid}"
    out = [
        ("GET", f"{r}/summary", None),
        ("GET", f"{r}/reruns", None),
        ("GET", f"{r}/image/max_proj", None),
        ("GET", f"{r}/image/background", None),
        ("GET", f"{r}/image/rank", None),
        ("GET", f"{r}/image/noise", None),
        ("GET", f"{r}/motion", None),
        ("GET", f"{r}/motion/residual", None),
        ("GET", f"{r}/seeds", None),
        ("GET", f"{r}/units", None),
        ("GET", f"{r}/units?stage=init", None),
        ("GET", f"{r}/traces/overview?kind=C&bins=2000", None),
        ("GET", f"{r}/traces/overview?kind=C&bins=37", None),
        ("GET", f"{r}/zooms", None),
        ("GET", f"{r}/frame/0?source=raw", None),
        ("GET", f"{r}/frame/{T // 2}?source=raw", None),
        ("GET", f"{r}/frame/{T - 1}?source=recon", None),
        ("GET", f"{r}/frame/{T // 3}?source=pre", None),
        ("GET", f"{r}/frame/{T}?source=raw", None),
        ("GET", "/params/defaults", None),
        ("GET", "/params/derived?diameter=7.5", None),
        ("GET", "/params/derived?diameter=12", None),
    ]
    for u in sorted({0, n_units // 2, max(n_units - 1, 0)}) if n_units else []:
        out += [
            ("GET", f"{r}/traces/{u}", None),
            ("GET", f"{r}/traces/{u}?start=10&stop={T - 10}&lowpass_hz=2", None),
        ]
    if raw is not None:
        out += [
            ("GET", f"/raw/info?path={raw}", None),
            ("POST", "/raw/preview", {"raw_dir": str(raw), "t": T // 2, "params": {}}),
            ("POST", "/raw/motion", {"raw_dir": str(raw), "t": T // 2, "params": {}}),
        ]
    return out


def ask(c: TestClient, method: str, path: str, body: dict | None):
    for _ in range(600):  # 202 = computing in the background: poll
        res = c.request(method, f"/api{path}", json=body, headers={"x-token": TOKEN})
        if res.status_code != 202:
            return res
        time.sleep(0.2)
    raise TimeoutError(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--raw", type=Path, default=None)
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    with (
        tempfile.TemporaryDirectory() as cache,
        # a server error is an answer (500), not an exception: a browser run has no factors in
        # Python's format, so main fails on its recon, pre, noise and residual requests
        TestClient(create_app(TOKEN, cache_root=Path(cache)), raise_server_exceptions=False) as c,
    ):
        root = a.run.resolve().parent
        runs = c.get("/api/runs", params={"root": str(root)}, headers={"x-token": TOKEN}).json()[
            "runs"
        ]
        run = next(x for x in runs if Path(x["path"]) == a.run.resolve())
        summary = c.get(f"/api/runs/{run['id']}/summary", headers={"x-token": TOKEN}).json()
        n_units = int(summary["counts"].get("units_final") or 0)
        manifest = []
        for i, (method, path, body) in enumerate(
            requests(run["id"], a.raw, n_units, int(summary["T"]))
        ):
            res = ask(c, method, path, body)
            name = f"{i:04d}"
            if res.headers.get("x-shape") is not None:
                arr = decode(res.content, res.headers["x-shape"], res.headers["x-dtype"])
                np.save(a.out / f"{name}.npy", arr)
                file = f"{name}.npy"
            else:
                try:
                    body_json = res.json()
                except ValueError:  # a 500 answers in plain text
                    body_json = {"detail": res.text}
                (a.out / f"{name}.json").write_text(json.dumps(body_json, indent=1))
                file = f"{name}.json"
            manifest.append(
                {
                    "request": f"{method} {path.replace(run['id'], '{rid}')}",
                    "body": body,
                    "status": res.status_code,
                    "file": file,
                }
            )
        (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} answers -> {a.out}")


if __name__ == "__main__":
    main()
