from pathlib import Path
from urllib.parse import parse_qs, urlparse

from typer.testing import CliRunner

from caanfast.cli import app
from caanfast.serve.server import free_port, start_url


def test_start_url_carries_token_and_folder():
    u = urlparse(start_url(8123, "abc", Path("/Volumes/Extreme SSD/x")))
    assert (u.hostname, u.port) == ("127.0.0.1", 8123)
    q = parse_qs(u.query)
    assert q["token"] == ["abc"] and q["root"] == ["/Volumes/Extreme SSD/x"]
    assert "root" not in parse_qs(urlparse(start_url(1, "t", None)).query)


def test_free_port_is_usable():
    assert 1024 < free_port() < 65536


def test_cli_serve_wires_options(monkeypatch, tmp_path):
    seen = {}

    def fake(folder, port=0, open_browser=True, token=None):
        seen.update(folder=folder, port=port, open_browser=open_browser, token=token)

    monkeypatch.setattr("caanfast.serve.server.serve_forever", fake)
    r = CliRunner().invoke(app, ["serve", str(tmp_path), "--port", "9001", "--no-browser"])
    assert r.exit_code == 0, r.output
    assert seen == {"folder": tmp_path, "port": 9001, "open_browser": False, "token": None}


def test_serve_forever_runs_uvicorn_quietly_on_loopback(monkeypatch):
    import uvicorn

    from caanfast.serve.server import serve_forever

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    serve_forever(None, port=8123, open_browser=False, token="t")
    assert seen == {
        "host": "127.0.0.1",
        "port": 8123,
        "log_level": "warning",
        "access_log": False,
    }
