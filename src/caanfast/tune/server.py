"""``caanfast tune``: a localhost page to rerun window 2 with new parameters and compare stages.

Standard library only (``http.server``). One worker thread runs window 2 on the factors held in
memory (``Window2Session``) and draws the figures; requests only read. Reruns go to
``RUN_DIR/reruns/NAME/`` (as ``caanfast cnmf``), shared figures, the pass-1 preview and saved
parameter files to ``RUN_DIR/tune/``.

Routes: ``GET /`` (``?run=NAME&cmp=NAME``), ``GET /api/status``, ``GET /fig/RUN/FILE.png``
(``RUN`` = ``_shared`` for the shared figures and the preview), ``GET /unit/RUN/K.png``,
``GET /thumb/RUN/K.png`` (a small crop of one unit, for the flagged gallery),
``GET /maxproj.png`` (the max projection at native size, to measure a cell),
``POST /api/run``, ``POST /api/save``, ``POST /api/preview``, ``POST /api/autotune``
(``diameter``: sweep, write the proposal as the rerun ``auto``).
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
import urllib.parse
from dataclasses import fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from caanfast import pipeline, window2
from caanfast.autotune import run as at
from caanfast.autotune.score import Checks
from caanfast.autotune.sweep import Trial
from caanfast.params import Params
from caanfast.tune import autofig, figures, minian, page, preview

SHARED = "_shared"
PREVIEW_PNG = "preview.png"
_PNG = re.compile(r"^[a-z0-9_]+\.png$")
_TYPES = {f.name: str(f.type) for f in fields(Params)}

# ---------------------------------------------------------------------------
# Form <-> Params
# ---------------------------------------------------------------------------


def params_to_form(p: Params, keys: tuple[str, ...] = window2.WINDOW2_PARAMS) -> dict[str, str]:
    return {k: page.fmt_value(getattr(p, k)) for k in keys}


def _number(s: str) -> int | float:
    v = float(s)
    return int(v) if v.is_integer() and "." not in s and "e" not in s.lower() else v


def _parse(key: str, s: str) -> Any:
    s = s.strip()
    t = _TYPES[key]
    if t == "bool":
        if s.lower() not in ("true", "false", "1", "0", "on", "off"):
            raise ValueError(f"{key}: expected true or false, got {s!r}")
        return s.lower() in ("true", "1", "on")
    if t == "int":
        return int(s)
    if t == "float":
        return float(s)
    if key == "pnr_thres":
        return "auto" if s == "auto" else float(s)
    if t.startswith("tuple"):
        parts = [x.strip() for x in s.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{key}: expected two comma-separated values, got {s!r}")
        if key == "size_thres":
            return tuple(None if x.lower() in ("", "none") else _number(x) for x in parts)
        return tuple(float(x) for x in parts)
    raise ValueError(f"{key}: unsupported type {t}")


def form_to_params(form: dict[str, str], base: Params) -> Params:
    """``base`` with the parameters named in ``form`` replaced; ``ValueError`` names a bad key."""
    d = {}
    for k, s in form.items():
        if k not in _TYPES:
            continue
        try:
            d[k] = _parse(k, s)
        except ValueError as e:
            raise ValueError(f"{k}: {e}" if not str(e).startswith(k) else str(e)) from None
    return Params(**{**base.asdict(), **d})


def changed(p: Params, base: Params) -> dict[str, tuple[Any, Any]]:
    """Window-2 parameters of ``p`` that differ from ``base``: ``{key: (base, new)}``."""
    a, b = base.asdict(), p.asdict()
    return {
        k: (a[k], b[k])
        for k in window2.WINDOW2_PARAMS
        if window2._norm(a[k]) != window2._norm(b[k])
    }


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class Busy(RuntimeError):
    pass


class TuneApp:
    """The run being tuned, its reruns, and the single worker that makes new ones."""

    def __init__(self, run_dir: Path, minian_npz: Path | None = None, start: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.session = window2.Window2Session(self.run_dir)
        self.minian_npz = Path(minian_npz) if minian_npz is not None else None
        self.tune_dir = self.run_dir / "tune"
        self.tune_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, figures.RunData]] = {}
        self.status: dict[str, Any] = {"state": "idle"}
        if start:
            self._start(self._startup, "base")

    # ---- reruns on disk ------------------------------------------------------------

    @property
    def reruns_dir(self) -> Path:
        return self.run_dir / window2.RERUNS

    def runs(self) -> list[dict[str, Any]]:
        """Every rerun, oldest first: its summary, or ``error`` if it failed."""
        out = []
        dirs = [d for d in self.reruns_dir.glob("*") if (d / "params.json").exists()]
        for d in sorted(dirs, key=lambda d: (d / "params.json").stat().st_mtime):
            row: dict[str, Any] = {"name": d.name}
            if (d / "summary.json").exists():
                row |= json.loads((d / "summary.json").read_text())
            elif (d / "error.txt").exists():
                row["error"] = (d / "error.txt").read_text().strip().splitlines()[-1]
            else:
                row["pending"] = True
            try:
                row["changed"] = changed(Params.from_json(d / "params.json"), self.session.base)
            except ValueError as e:
                row["changed"] = {"?": (str(e), "")}
            out.append(row)
        return out

    def names(self) -> list[str]:
        return [r["name"] for r in self.runs()]

    def next_name(self) -> str:
        taken = set(self.names())
        i = 1
        while f"t{i:03d}" in taken:
            i += 1
        return f"t{i:03d}"

    def run_data(self, name: str) -> figures.RunData:
        d = self.reruns_dir / name
        mtime = (d / "report.json").stat().st_mtime
        with self._lock:
            hit = self._cache.get(name)
            if hit is None or hit[0] != mtime:
                if len(self._cache) >= 4:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[name] = (mtime, figures.load_run(d))
            return self._cache[name][1]

    def overlay(self, run: figures.RunData) -> minian.MinianOverlay | None:
        return minian.load(self.minian_npz, run) if self.minian_npz is not None else None

    def params_of(self, name: str | None) -> Params:
        if name and (self.reruns_dir / name / "params.json").exists():
            return Params.from_json(self.reruns_dir / name / "params.json")
        return self.session.base

    # ---- worker --------------------------------------------------------------------

    def _set(self, **kw: Any) -> None:
        with self._lock:
            self.status = {**self.status, **kw}

    def _start(self, target: Any, name: str, *args: Any) -> None:
        with self._lock:
            if self.status.get("state") == "running":
                raise Busy(f"busy with {self.status.get('name')}")
            self.status = {"state": "running", "name": name, "stage": "", "t0": time.time()}
        threading.Thread(target=self._guard, args=(target, name, *args), daemon=True).start()

    def _guard(self, target: Any, name: str, *args: Any) -> None:
        try:
            target(name, *args)
        except Exception as e:  # reported on the page
            traceback.print_exc()
            self._set(state="error", error=f"{type(e).__name__}: {e}")
        else:
            self._set(state="done", error=None)
        finally:
            self._set(elapsed=round(time.time() - self.status["t0"], 1))

    def _figures(self, name: str) -> None:
        self._set(stage="figures")
        run = self.run_data(name)
        figures.make_run_figures(run, self.overlay(run), self.checks())

    def _rerun(self, name: str, params: Params) -> None:
        self.session.run(params, name, on_stage=lambda s: self._set(stage=s))
        self._figures(name)

    def checks(self) -> Checks | None:
        """The unit checks of the last auto-tune (``reruns/auto/autotune.json``), if any."""
        doc = at.load_doc(self.reruns_dir / at.AUTO)
        return at.checks_of(doc) if doc is not None else None

    def _autotune(self, name: str, diameter: float) -> None:
        n = at.DEFAULT_GRIDS.n_trials()
        done: list[Trial] = []

        def progress(t: Trial) -> None:
            done.append(t)
            self._set(stage=f"{t.step}: setting {len(done)} of ~{n}")

        at.autotune(self.run_dir, diameter, name=name, session=self.session, on_trial=progress)
        self._figures(name)
        self._rescore()

    def _rescore(self) -> None:
        """Rewrite every run's summary with the current checks (good / flagged columns)."""
        self._set(stage="rescoring runs")
        checks = self.checks()
        for d in sorted(self.reruns_dir.glob("*")):
            if (d / "report.json").exists() and (d / "stages" / "index.json").exists():
                run = self.run_data(d.name)
                figures.write_summary(run, self.overlay(run), checks)

    def _startup(self, name: str) -> None:
        self._set(stage="shared figures")
        figures.make_shared_figures(self.run_dir, self.tune_dir, self.session.fac)
        if not (self.reruns_dir / name / "report.json").exists():
            self._rerun(name, self.session.base)
        for d in sorted(self.reruns_dir.glob("*")):
            s = d / "summary.json"
            text = s.read_text() if s.exists() else ""
            stale = s.exists() and (
                (self.minian_npz and "minian_matched" not in text)
                or (self.checks() is not None and '"good"' not in text)
            )
            if (d / "stages" / "index.json").exists() and (not s.exists() or stale):
                self._figures(d.name)

    def start_run(self, params: Params, name: str | None = None) -> str:
        name = name or self.next_name()
        self.session.out_dir(name)  # validates the name
        at.check_not_reserved(name)
        window2.check_factor_params(self.session.fac, params)
        self._start(self._rerun, name, params)
        return name

    def start_autotune(self, diameter: float) -> str:
        fac = self.session.fac
        at.check_diameter(diameter, fac.H, fac.W_)  # ValueError -> HTTP 400 before the worker
        self._start(self._autotune, at.AUTO, diameter)
        return at.AUTO

    def save_params(self, name: str, file: str) -> Path:
        pipeline.check_run_name(file)
        if not file.endswith(".json"):
            file += ".json"
        path = self.tune_dir / file
        Params.from_json(self.reruns_dir / name / "params.json").to_json(path)
        return path

    def make_preview(self, params: Params) -> Path:
        raw = window2.raw_dir(self.run_dir)
        if raw is None or not raw.exists():
            raise ValueError(f"raw folder {raw} not found")
        return preview.make_preview(raw, params, self.tune_dir / PREVIEW_PNG)

    def figure_path(self, run: str, file: str) -> Path | None:
        if not _PNG.match(file):
            return None
        if run == SHARED:
            p = self.tune_dir / file
        elif run in self.names():
            p = self.reruns_dir / run / "figures" / file
        else:
            return None
        return p if p.is_file() else None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    app: TuneApp  # set by make_server

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(obj, default=str).encode())

    def _not_found(self) -> None:
        self._send(404, "text/plain", b"not found")

    def _form(self) -> dict[str, str]:
        n = int(self.headers.get("Content-Length") or 0)
        q = urllib.parse.parse_qs(self.rfile.read(n).decode(), keep_blank_values=True)
        return {k: v[-1] for k, v in q.items()}

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.strip("/").split("/")]
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        app = self.app
        if url.path == "/":
            html = page.render(app, q.get("run"), q.get("cmp"))
            self._send(200, "text/html; charset=utf-8", html.encode())
        elif url.path == "/maxproj.png":
            self._send(200, "image/png", autofig.max_proj_png(app.session.fac.max_proj))
        elif url.path == "/api/status":
            st = dict(app.status)
            if st.get("state") == "running":
                st["elapsed"] = round(time.time() - st["t0"], 1)
            self._json(st)
        elif len(parts) == 3 and parts[0] == "fig":
            p = app.figure_path(parts[1], parts[2])
            self._send(200, "image/png", p.read_bytes()) if p else self._not_found()
        elif len(parts) == 3 and parts[0] in ("unit", "thumb") and parts[1] in app.names():
            m = re.fullmatch(r"(\d+)\.png", parts[2])
            run = app.run_data(parts[1])
            if m is None or int(m[1]) >= run.A.shape[0]:
                return self._not_found()
            k = int(m[1])
            png = (
                figures.unit_png(run, k, app.overlay(run))
                if parts[0] == "unit"
                else autofig.thumb_png(run, k)
            )
            self._send(200, "image/png", png)
        else:
            self._not_found()

    def do_POST(self) -> None:
        app, form = self.app, self._form()
        try:
            if self.path == "/api/run":
                base = app.params_of(form.pop("from", None))
                name = form.pop("name", "").strip() or None
                self._json({"name": app.start_run(form_to_params(form, base), name)})
            elif self.path == "/api/save":
                self._json({"path": str(app.save_params(form["run"], form["file"]))})
            elif self.path == "/api/preview":
                prm = form_to_params(form, app.params_of(form.pop("from", None)))
                self._json({"path": str(app.make_preview(prm))})
            elif self.path == "/api/autotune":
                try:
                    d = float(form.get("diameter", ""))
                except ValueError:
                    raise ValueError("diameter: expected a number of px") from None
                self._json({"name": app.start_autotune(d)})
            else:
                self._not_found()
        except Busy as e:
            self._json({"error": str(e)}, HTTPStatus.CONFLICT)
        except (ValueError, KeyError, FileNotFoundError) as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)


def make_server(app: TuneApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("TuneHandler", (Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd
