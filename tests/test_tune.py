import json
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import pytest
import scipy.sparse as sps

from cafast import window2
from cafast.params import Params

pytest.importorskip("matplotlib")

from cafast.tune import figures, minian, preview, server  # noqa: E402

# ---------------------------------------------------------------------------
# Footprint metrics
# ---------------------------------------------------------------------------


def _footprints(H: int, W: int, blobs: list[list[tuple[int, int, int]]]) -> sps.csr_matrix:
    """One row per unit: the union of square blobs ``(row, col, half-side)``, peak 1."""
    rows = []
    for unit in blobs:
        img = np.zeros((H, W), np.float32)
        for r, c, s in unit:
            img[r - s : r + s + 1, c - s : c + s + 1] = 1.0
        img[unit[0][0], unit[0][1]] = 1.0
        rows.append(img.ravel())
    return sps.csr_matrix(np.array(rows))


def test_size_above_and_parts():
    A = _footprints(30, 30, [[(5, 5, 1)], [(15, 5, 2), (15, 20, 1)]])
    A = sps.csr_matrix(A.multiply(np.linspace(0.5, 1.0, 900)[None]))  # non-flat values
    assert figures.size_above(A, 0.0).tolist() == [9, 34]
    assert figures.n_parts(A, 30, 30).tolist() == [1, 2]


def test_outlines_trace_each_part():
    A = _footprints(30, 30, [[(15, 5, 2), (15, 20, 2)]])
    parts = figures.outlines(A, 30, 30)[0]
    assert len(parts) == 2
    xs = sorted(float(p[:, 0].mean()) for p in parts)
    assert xs == pytest.approx([5, 20], abs=0.6)
    assert all(np.array_equal(p[0], p[-1]) for p in parts)  # closed


# ---------------------------------------------------------------------------
# Figures and the Minian overlay
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_rerun(tiny_run):
    return window2.rerun(tiny_run, name="figs")


@pytest.fixture(scope="module")
def self_minian(tiny_run, tmp_path_factory):
    """A 'Minian' result that is the cafast run itself, shifted by (2, -1) px."""
    A = sps.load_npz(tiny_run / "cnmf" / "A.npz").toarray().reshape(-1, 64, 64)
    mp = np.load(tiny_run / "factors" / "max_proj.npy").astype(np.float32)
    path = tmp_path_factory.mktemp("mn") / "minian_units.npz"
    np.savez(
        path,
        A=np.roll(A, (2, -1), axis=(1, 2)).reshape(len(A), -1),
        C=np.load(tiny_run / "cnmf" / "YrA.npy"),
        max_proj=np.roll(mp, (2, -1), axis=(0, 1)),
    )
    return path


def test_minian_overlay_aligns_and_matches(tiny_rerun, self_minian):
    run = figures.load_run(tiny_rerun)
    mn = minian.load(self_minian, run)
    assert mn.offset == pytest.approx([-2, 1], abs=0.3)
    n = run.A.shape[0]
    assert len(mn.ui) >= 0.8 * n
    assert np.nanmedian(mn.corr) > 0.99


def test_run_figures_and_summary(tiny_rerun, self_minian):
    run = figures.load_run(tiny_rerun)
    summ = figures.make_run_figures(run, minian.load(self_minian, run))
    names = {p.name for p in (tiny_rerun / "figures").iterdir()}
    assert set(figures.RUN_FIGURES) <= names
    assert "minian.png" in names
    for n in names:
        assert (tiny_rerun / "figures" / n).read_bytes()[:4] == b"\x89PNG"
    assert summ["units_final"] == run.A.shape[0]
    assert summ["minian_matched"] >= 0.8 * run.A.shape[0]
    assert json.loads((tiny_rerun / "summary.json").read_text()) == summ
    table = figures.unit_table(run, None)
    assert len(table) == run.A.shape[0]
    assert {"size_px", "size10", "parts", "pnr"} <= set(table.columns)


def test_shared_figures(tiny_run, tmp_path):
    figures.make_shared_figures(tiny_run, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} >= set(figures.SHARED_FIGURES)


def test_unit_figure(tiny_rerun):
    run = figures.load_run(tiny_rerun)
    png = figures.unit_png(run, 0, None)
    assert png[:4] == b"\x89PNG"


def test_pass1_preview(tiny_run, tmp_path):
    prm = window2.run_params(tiny_run)
    out = preview.make_preview(window2.raw_dir(tiny_run), prm, tmp_path / "pv.png", n_frames=60)
    assert out.read_bytes()[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tune_server(tiny_run):
    app = server.TuneApp(tiny_run, minian_npz=None)
    httpd = server.make_server(app, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers["Content-Type"], r.read()


def _post(url: str, data: dict) -> dict:
    body = urllib.parse.urlencode(data).encode()
    with urllib.request.urlopen(urllib.request.Request(url, body)) as r:
        return json.loads(r.read())


def _wait(base: str, timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = json.loads(_get(base + "/api/status")[2])
        if st["state"] != "running":
            return st
        time.sleep(0.2)
    raise TimeoutError


def test_server_rerun_roundtrip(tune_server):
    app, base = tune_server
    _wait(base)  # the initial run with the run's own params
    status, ctype, body = _get(base + "/")
    assert status == 200 and ctype.startswith("text/html")
    page = body.decode()
    for k in window2.WINDOW2_PARAMS:
        assert f'name="{k}"' in page, k
    form = server.params_to_form(app.session.base)
    form.update(sparse_penal="0.05", name="sp05")
    assert _post(base + "/api/run", form)["name"] == "sp05"
    st = _wait(base)
    assert st["state"] == "done", st
    assert Params.from_json(app.run_dir / "reruns" / "sp05" / "params.json").sparse_penal == 0.05
    page = _get(base + "/?run=sp05&cmp=base")[2].decode()
    assert "sp05" in page and "sparse_penal" in page
    status, ctype, png = _get(base + "/fig/sp05/cnmf.png")
    assert status == 200 and ctype == "image/png" and png[:4] == b"\x89PNG"
    saved = _post(base + "/api/save", {"run": "sp05", "file": "tuned.json"})
    assert Params.from_json(saved["path"]) == Params.from_json(
        app.run_dir / "reruns" / "sp05" / "params.json"
    )


def test_page_with_minian_overlay(tiny_run, self_minian):
    app = server.TuneApp(tiny_run, minian_npz=self_minian, start=False)
    app._startup("base")  # runs base if missing, redraws summaries without Minian keys
    page_html = server.page.render(app, "base", None)
    assert "minian.png" in page_html and "Minian matched" in page_html
    assert "minian_unit" in page_html


def test_server_reports_bad_params(tune_server):
    app, base = tune_server
    _wait(base)
    form = server.params_to_form(app.session.base)
    form.update(size_thres="100000, ", name="toolarge")
    _post(base + "/api/run", form)
    st = _wait(base)
    assert st["state"] == "error" and "size_thres" in st["error"]


@pytest.mark.parametrize("path", ["/fig/../params.json", "/fig/base/../../params.json"])
def test_server_serves_only_figures(tune_server, path):
    _, base = tune_server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + path)
    assert e.value.code == 404


def test_params_form_roundtrip(tiny_run):
    prm = window2.run_params(tiny_run)
    assert server.form_to_params(server.params_to_form(prm), prm) == prm
    p1 = Params(**{**prm.asdict(), "seeds_diff_thres": 8.0, "pnr_thres": 2.0})
    assert server.params_to_form(p1)["seeds_diff_thres"] == "8"
    assert server.form_to_params(server.params_to_form(p1), prm) == p1
    p2 = server.form_to_params({"pnr_thres": "auto", "size_thres": "25, 300"}, prm)
    assert p2.pnr_thres == "auto" and p2.size_thres == (25, 300)
    with pytest.raises(ValueError, match="dl_wnd"):
        server.form_to_params({"dl_wnd": "abc"}, prm)
