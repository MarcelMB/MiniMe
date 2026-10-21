"""``create_app``: the FastAPI app of ``caanfast serve`` (guard, state, static files)."""

from __future__ import annotations

import secrets
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from caanfast.serve.cache import Background
from caanfast.serve.runs import Registry

if TYPE_CHECKING:
    from caanfast.serve.jobs import JobManager

STATIC_DIR = Path(__file__).with_name("static")

NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>CaAnFast</title>
<p>The CaAnFast front end is not built. In the repository run
<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, then restart
<code>caanfast serve</code>.</p>"""

LOCAL_HOSTS = ("127.0.0.1", "localhost")


def host_ok(host: str) -> bool:
    """A Host header naming this machine by loopback name, with any port or none.

    The check stops DNS rebinding; the port is free so an SSH tunnel to another local
    port (``ssh -L 9000:127.0.0.1:8765``) still works.
    """
    name, sep, port = host.rpartition(":")
    if not sep:
        name, port = host, ""
    return name in LOCAL_HOSTS and (port == "" or port.isdigit())


def create_app(
    token: str,
    port: int | None = None,
    cache_root: Path | None = None,
    static_dir: Path | None = None,
    jobs: JobManager | None = None,
) -> FastAPI:
    """The app; ``port`` enables the Host check (None in tests, where Host is testserver)."""
    from caanfast.serve.routes_jobs import router as jobs_router
    from caanfast.serve.routes_runs import router as runs_router

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if jobs is not None:
            jobs.shutdown()  # a running job must not outlive the server

    app = FastAPI(
        title="CaAnFast", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.registry = Registry(cache_root)
    app.state.frames = {}
    app.state.frames_lock = threading.Lock()
    app.state.background = Background()
    app.state.jobs = jobs

    @app.exception_handler(FileNotFoundError)
    async def file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        name = Path(str(exc.filename)).name if exc.filename else "unknown"
        return JSONResponse({"detail": f"file not found: {name}"}, status_code=404)

    @app.middleware("http")
    async def guard(request: Request, call_next):  # noqa: ANN202
        if port and not host_ok(request.headers.get("host", "")):
            return PlainTextResponse("wrong host", status_code=403)
        if request.url.path.startswith("/api/"):
            given = request.headers.get("x-token") or request.query_params.get("token") or ""
            if not secrets.compare_digest(given.encode(), token.encode()):
                return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    app.include_router(runs_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")

    static = Path(static_dir or STATIC_DIR)
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        def not_built() -> str:
            return NOT_BUILT

    return app
