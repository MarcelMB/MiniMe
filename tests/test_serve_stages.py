import shutil

import numpy as np

from minime.params import Params
from minime.serve.arrays import decode
from minime.serve.paramsmeta import COMMON, GROUPS, HELP, defaults
from tests.conftest import wait_ok


def _arr(r):
    return decode(r.content, r.headers["X-Shape"], r.headers["X-Dtype"])


def _open(client, run):
    return client.get("/api/runs", params={"root": str(run)}).json()["runs"][0]["id"]


def test_params_metadata_covers_every_parameter():
    names = [n for g in GROUPS.values() for n in g]
    assert sorted(names) == sorted(Params().asdict()) and len(names) == len(set(names))
    assert set(HELP) == set(names) and set(COMMON) <= set(names)
    d = {p["name"]: p for p in defaults()}
    assert d["size_thres"]["default"] == [25, None] and d["workers"]["common"] is True


def test_params_defaults_route(client):
    body = client.get("/api/params/defaults").json()
    assert [g for g in body["groups"]] == list(GROUPS)
    assert len(body["params"]) == len(Params().asdict())


def test_motion_and_residual(client, tiny_run):
    rid = _open(client, tiny_run)
    m = _arr(client.get(f"/api/runs/{rid}/motion"))
    assert m.shape == (300, 3)
    r = wait_ok(client, f"/api/runs/{rid}/motion/residual")
    assert r.status_code == 200 and _arr(r).shape[1] == 3


def test_seeds_and_units(client, tiny_run):
    rid = _open(client, tiny_run)
    s = client.get(f"/api/runs/{rid}/seeds").json()
    assert set(s) == {"init", "pnr", "merged"} and len(s["merged"][0]) == 2
    fin = client.get(f"/api/runs/{rid}/units").json()
    n = len(fin["units"])
    assert fin["stage"] == "final" and n > 0
    u = fin["units"][0]
    assert set(u) == {"id", "centroid", "size_px", "peak", "polygons"}
    ini = client.get(f"/api/runs/{rid}/units?stage=init").json()
    assert ini["units"][0]["peak"] is None
    assert client.get(f"/api/runs/{rid}/units?stage=nope").status_code == 400


def test_seeds_of_an_older_run_answer_202_then_200(client, run_copy):
    shutil.rmtree(run_copy / "seeds")
    shutil.rmtree(run_copy / "init")
    rid = _open(client, run_copy)
    assert client.get(f"/api/runs/{rid}/seeds").status_code == 202
    assert wait_ok(client, f"/api/runs/{rid}/seeds").status_code == 200
    assert wait_ok(client, f"/api/runs/{rid}/units?stage=init").status_code == 200


def test_traces(client, tiny_run):
    rid = _open(client, tiny_run)
    n = len(client.get(f"/api/runs/{rid}/units").json()["units"])
    ov = wait_ok(client, f"/api/runs/{rid}/traces/overview?kind=C&bins=50")
    assert _arr(ov).shape == (n, 2, 50)
    full = _arr(client.get(f"/api/runs/{rid}/traces/{n - 1}"))
    assert full.shape == (300,)
    part = _arr(client.get(f"/api/runs/{rid}/traces/0?start=100&stop=200&lowpass_hz=5"))
    assert part.shape == (100,) and np.isfinite(part).all()
    yra = _arr(client.get(f"/api/runs/{rid}/traces/0?kind=YrA"))
    assert yra.shape == (300,)
    assert client.get(f"/api/runs/{rid}/traces/{n}").status_code == 400
    assert client.get(f"/api/runs/{rid}/traces/0?start=0&stop=301").status_code == 400
    assert client.get(f"/api/runs/{rid}/traces/0?kind=S").status_code == 400
    z = wait_ok(client, f"/api/runs/{rid}/zooms").json()
    assert len(z["units"]) == min(10, n) and z["windows"][0]["start"] == 0


def test_missing_background_file_is_404(client, run_copy):
    (run_copy / "cnmf" / "b.npy").unlink()
    rid = _open(client, run_copy)
    assert client.get(f"/api/runs/{rid}/image/background").status_code == 404


def test_lowpass_without_a_frame_rate_is_400(client, run_copy):
    import json

    pj = run_copy / "params.json"
    pj.write_text(json.dumps({**json.loads(pj.read_text()), "raw_dir": "/nowhere/raw"}))
    rid = _open(client, run_copy)
    assert client.get(f"/api/runs/{rid}/summary").json()["fps"] is None
    assert client.get(f"/api/runs/{rid}/traces/0").status_code == 200
    assert client.get(f"/api/runs/{rid}/traces/0?lowpass_hz=5").status_code == 400


def test_rank_image_does_not_load_the_factors(client, run_copy):
    rid = _open(client, run_copy)
    r = client.get(f"/api/runs/{rid}/image/rank")
    assert r.status_code == 200
    run = client.app.state.registry.get(rid)
    assert _arr(r).shape == (run.H, run.W)
    assert set(np.unique(_arr(r))) <= set(map(float, run.ranks))
    assert run._factors is None
