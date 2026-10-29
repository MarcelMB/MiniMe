import json

import numpy as np
from fastapi.testclient import TestClient

from minime.serve.app import create_app
from minime.serve.arrays import decode
from tests.conftest import wait_ok


def _arr(r):
    return decode(r.content, r.headers["X-Shape"], r.headers["X-Dtype"])


def _open(client, run):
    r = client.get("/api/runs", params={"root": str(run)})
    assert r.status_code == 200
    (item,) = r.json()["runs"]
    return item["id"]


def test_api_needs_the_token(cache_dir, tmp_path):
    c = TestClient(create_app("tok", cache_root=cache_dir, static_dir=tmp_path / "s"))
    assert c.get("/api/fs").status_code == 401
    assert c.get("/api/fs", headers={"X-Token": "nope"}).status_code == 401
    assert c.get("/api/fs", headers={"X-Token": "tok"}).status_code == 200
    assert c.get("/api/fs", params={"token": "tok"}).status_code == 200


def test_wrong_host_is_rejected(cache_dir, tmp_path):
    app = create_app("tok", port=8123, cache_root=cache_dir, static_dir=tmp_path / "s")
    c = TestClient(app, base_url="http://127.0.0.1:8123")
    h = {"X-Token": "tok"}
    assert c.get("/api/fs", headers=h).status_code == 200
    assert c.get("/api/fs", headers={**h, "Host": "localhost:8123"}).status_code == 200
    assert c.get("/api/fs", headers={**h, "Host": "evil.example:8123"}).status_code == 403
    # an SSH tunnel to another local port keeps a loopback name with another port
    assert c.get("/api/fs", headers={**h, "Host": "127.0.0.1:9000"}).status_code == 200
    assert c.get("/api/fs", headers={**h, "Host": "localhost"}).status_code == 200
    assert c.get("/api/fs", headers={**h, "Host": "127.0.0.1.evil:8123"}).status_code == 403
    assert c.get("/api/fs", headers={**h, "Host": "localhost:x"}).status_code == 403


def test_static_page_and_missing_build(cache_dir, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>minime</html>", encoding="utf-8")
    c = TestClient(create_app("tok", cache_root=cache_dir, static_dir=static))
    assert "minime" in c.get("/").text
    c2 = TestClient(create_app("tok", cache_root=cache_dir, static_dir=tmp_path / "none"))
    assert "npm run build" in c2.get("/").text


def test_fs_lists_folders_and_flags_runs(client, tiny_run):
    r = client.get("/api/fs", params={"path": str(tiny_run.parent)}).json()
    assert r["path"] == str(tiny_run.parent.resolve())
    names = {e["name"]: e for e in r["entries"]}
    assert names["t"]["is_run"] is True
    up = client.get("/api/fs", params={"path": str(tiny_run.parent.parent)}).json()
    assert {e["name"]: e for e in up["entries"]}["raw"]["has_avi"] is True
    assert client.get("/api/fs", params={"path": str(tiny_run / "nope")}).status_code == 404


def test_summary_image_and_frames(client, tiny_run):
    rid = _open(client, tiny_run)
    s = client.get(f"/api/runs/{rid}/summary").json()
    assert s["T"] == 300 and s["available"]["pre"] is True
    for name in ("max_proj", "background", "rank", "noise"):
        a = _arr(client.get(f"/api/runs/{rid}/image/{name}"))
        assert a.shape == (64, 64) and a.dtype == np.float32
    assert client.get(f"/api/runs/{rid}/image/nope").status_code == 404
    for source in ("raw", "pre", "recon"):
        r = wait_ok(client, f"/api/runs/{rid}/frame/299?source={source}")
        assert r.status_code == 200
        assert _arr(r).shape == (64, 64) and r.headers["X-Dtype"] == "uint8"
    assert client.get(f"/api/runs/{rid}/frame/300?source=recon").status_code == 400
    assert client.get(f"/api/runs/{rid}/frame/-1?source=recon").status_code == 400
    assert client.get(f"/api/runs/{rid}/frame/0?source=movie").status_code == 400
    assert client.get("/api/runs/000000000000/summary").status_code == 404


def test_unplugged_raw_and_user_fps(client, run_copy):
    pj = json.loads((run_copy / "params.json").read_text(encoding="utf-8"))
    pj["raw_dir"] = str(run_copy / "unplugged")
    (run_copy / "params.json").write_text(json.dumps(pj), encoding="utf-8")
    rid = _open(client, run_copy)
    r = client.get(f"/api/runs/{rid}/frame/0?source=raw")
    assert r.status_code == 404 and "raw videos not found" in r.json()["detail"]
    assert client.get(f"/api/runs/{rid}/summary").json()["fps"] is None
    assert client.post(f"/api/runs/{rid}/fps", json={"fps": 20}).status_code == 200
    assert client.get(f"/api/runs/{rid}/summary").json()["fps"] == 20.0
    assert client.post(f"/api/runs/{rid}/fps", json={"fps": -1}).status_code == 400


def test_non_ascii_token_is_401(client):
    r = client.get("/api/params/defaults", headers={"X-Token": ""}, params={"token": "é"})
    assert r.status_code == 401


def test_a_missing_file_is_404_without_the_full_path(client, run_copy):
    import shutil

    shutil.rmtree(run_copy / "motion")
    rid = _open(client, run_copy)
    r = client.get(f"/api/runs/{rid}/motion")
    assert r.status_code == 404
    assert r.json()["detail"].startswith("file not found: ")
    assert str(run_copy) not in r.json()["detail"]


def test_cache_root_ignores_an_empty_env_var(monkeypatch):
    from pathlib import Path

    from minime.serve.cache import cache_root

    monkeypatch.setenv("MINIME_CACHE", "")
    assert cache_root() == Path.home() / ".cache" / "minime"
