"""Start ``caanfast serve``: free port, random token, uvicorn on 127.0.0.1, open the browser."""

from __future__ import annotations

import secrets
import socket
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
    app = create_app(token, port=port, jobs=JobManager())
    url = start_url(port, token, Path(folder).resolve() if folder else None)
    print(f"CaAnFast is running at {url}\nPress Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
