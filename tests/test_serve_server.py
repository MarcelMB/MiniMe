from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

from minime.cli import app
from minime.serve.server import free_port, start_url


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

    monkeypatch.setattr("minime.serve.server.serve_forever", fake)
    r = CliRunner().invoke(app, ["serve", str(tmp_path), "--port", "9001", "--no-browser"])
    assert r.exit_code == 0, r.output
    assert seen == {"folder": tmp_path, "port": 9001, "open_browser": False, "token": None}


def test_serve_forever_runs_uvicorn_quietly_on_loopback(monkeypatch, tmp_path):
    import signal

    import uvicorn

    from minime.serve.server import serve_forever

    monkeypatch.setenv("MINIME_CACHE", str(tmp_path / "cache"))
    seen, handlers = {}, {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(signal, "signal", lambda sig, h: handlers.update({sig: h}))
    serve_forever(None, port=8123, open_browser=False, token="t")
    assert seen == {
        "host": "127.0.0.1",
        "port": 8123,
        "log_level": "warning",
        "access_log": False,
        "timeout_graceful_shutdown": 2,
    }
    assert signal.SIGHUP in handlers


def test_a_second_server_on_the_same_cache_exits_with_status_1(monkeypatch, tmp_path, capsys):
    import uvicorn

    from minime.serve.jobs import JobManager
    from minime.serve.server import serve_forever

    monkeypatch.setenv("MINIME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: pytest.fail("must not start"))
    first = JobManager()
    with pytest.raises(SystemExit) as e:
        serve_forever(None, port=8123, open_browser=False, token="t")
    assert e.value.code == 1
    assert "already running" in capsys.readouterr().err
    first.shutdown()
