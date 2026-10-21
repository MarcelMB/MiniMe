"""Start ``caanfast serve``: free port, random token, uvicorn on 127.0.0.1, open the browser."""

from __future__ import annotations

import os
import secrets
import signal
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_url(port: int, token: str, folder: Path | None) -> str:
    q = {"token": token}
    if folder is not None:
        q["root"] = str(folder)
    return f"http://127.0.0.1:{port}/?{urlencode(q)}"


def serve_forever(
    folder: Path | None, port: int = 0, open_browser: bool = True, token: str | None = None
) -> None:
    import uvicorn

    from caanfast.serve.app import create_app
    from caanfast.serve.jobs import JobManager

    port = port or free_port()
    token = token or secrets.token_urlsafe(16)
    try:
        jobs = JobManager()
    except RuntimeError as e:
        print(f"{e}; stop it first or use that one.", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    app = create_app(token, port=port, jobs=jobs)
    url = start_url(port, token, Path(folder).resolve() if folder else None)
    print(f"CaAnFast is running at {url}\nPress Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # closing the terminal (SIGHUP) stops like Ctrl+C, so the running job is stopped too
    signal.signal(signal.SIGHUP, lambda *_: os.kill(os.getpid(), signal.SIGTERM))
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=2,
    )
