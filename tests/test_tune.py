import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from minime import window2
from minime.autotune import run as at
from minime.autotune import score, sweep
from minime.params import Params

pytest.importorskip("matplotlib")

from minime.tune import figures, minian, preview, server  # noqa: E402

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
def tiny_rerun(tiny_w2_run):
    return window2.rerun(tiny_w2_run, name="figs")


@pytest.fixture(scope="module")
def self_minian(tiny_w2_run, tmp_path_factory):
    """A 'Minian' result that is the minime run itself, shifted by (2, -1) px."""
    A = sps.load_npz(tiny_w2_run / "cnmf" / "A.npz").toarray().reshape(-1, 64, 64)
    mp = np.load(tiny_w2_run / "factors" / "max_proj.npy").astype(np.float32)
    path = tmp_path_factory.mktemp("mn") / "minian_units.npz"
    np.savez(
        path,
        A=np.roll(A, (2, -1), axis=(1, 2)).reshape(len(A), -1),
        C=np.load(tiny_w2_run / "cnmf" / "YrA.npy"),
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


def test_shared_figures(tiny_w2_run, tmp_path):
    figures.make_shared_figures(tiny_w2_run, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} >= set(figures.SHARED_FIGURES)


def test_unit_figure(tiny_rerun):
    run = figures.load_run(tiny_rerun)
    png = figures.unit_png(run, 0, None)
    assert png[:4] == b"\x89PNG"


def test_pass1_preview(tiny_w2_run, tmp_path):
    prm = window2.run_params(tiny_w2_run)
    out = preview.make_preview(window2.raw_dir(tiny_w2_run), prm, tmp_path / "pv.png", n_frames=60)
    assert out.read_bytes()[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tune_server(tiny_w2_run):
    app = server.TuneApp(tiny_w2_run, minian_npz=None)
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
    form.update(sparse_snr="0.5", name="sp05")
    assert _post(base + "/api/run", form)["name"] == "sp05"
    st = _wait(base)
    assert st["state"] == "done", st
    assert Params.from_json(app.run_dir / "reruns" / "sp05" / "params.json").sparse_snr == 0.5
    page = _get(base + "/?run=sp05&cmp=base")[2].decode()
    assert "sp05" in page and "sparse_snr" in page
    status, ctype, png = _get(base + "/fig/sp05/cnmf.png")
    assert status == 200 and ctype == "image/png" and png[:4] == b"\x89PNG"
    saved = _post(base + "/api/save", {"run": "sp05", "file": "tuned.json"})
    assert Params.from_json(saved["path"]) == Params.from_json(
        app.run_dir / "reruns" / "sp05" / "params.json"
    )


def test_page_with_minian_overlay(tiny_w2_run, self_minian):
    app = server.TuneApp(tiny_w2_run, minian_npz=self_minian, start=False)
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


def test_params_form_roundtrip(tiny_w2_run):
    prm = window2.run_params(tiny_w2_run)
    assert server.form_to_params(server.params_to_form(prm), prm) == prm
    p1 = Params(**{**prm.asdict(), "seeds_diff_thres": 8.0, "pnr_thres": 2.0})
    assert server.params_to_form(p1)["seeds_diff_thres"] == "8"
    assert server.form_to_params(server.params_to_form(p1), prm) == p1
    p2 = server.form_to_params({"pnr_thres": "auto", "size_thres": "25, 300"}, prm)
    assert p2.pnr_thres == "auto" and p2.size_thres == (25, 300)
    with pytest.raises(ValueError, match="dl_wnd"):
        server.form_to_params({"dl_wnd": "abc"}, prm)


# ---------------------------------------------------------------------------
# Auto-tune on the page
# ---------------------------------------------------------------------------

SMALL_GRIDS = sweep.Grids(
    seeds_diff_thres=(2.0, 3.0),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_snr=(0.5, 1.0),
    loop_merge_corr=(0.9,),
    n_confirm=2,
)


def test_best_per_value_and_curves(tmp_path):
    from minime.tune import autofig

    df = pd.DataFrame(
        {
            "step": ["seeds"] * 4 + ["sparse"] * 2,
            "seeds_diff_thres": [2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            "pnr_thres": ["auto", "1.0", "auto", "1.0", "auto", "auto"],
            "sparse_snr": [1.0] * 4 + [0.5, 1.0],
            "score": [3, 5, 4, None, 6, 2],
            "good": [5, 6, 5, None, 7, 4],
            "flagged": [2, 1, 1, None, 1, 2],
        }
    )
    b = autofig.best_per_value(df, "seeds_diff_thres")
    assert b["seeds_diff_thres"].tolist() == [2.0, 3.0] and b["score"].tolist() == [5, 4]
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    doc = {"proposed": {"seeds_diff_thres": 2.0, "pnr_thres": "1.0", "sparse_snr": 0.5}}
    names = autofig.sweep_curves(csv, doc, tmp_path)
    assert names == ["sweep_seeds_diff_thres.png", "sweep_pnr_thres.png", "sweep_sparse_snr.png"]
    assert (tmp_path / names[0]).read_bytes()[:4] == b"\x89PNG"


def test_server_autotune_roundtrip(tune_server, monkeypatch):
    app, base = tune_server
    _wait(base)
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/autotune", {"diameter": "abc"})
    assert e.value.code == 400
    monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL_GRIDS)
    assert _post(base + "/api/autotune", {"diameter": "7"})["name"] == "auto"
    st = _wait(base, timeout=300)
    assert st["state"] == "done", st
    assert app.checks() is not None and app.checks().diameter == 7
    page_html = _get(base + "/?run=auto")[2].decode()
    assert "Proposal" in page_html and "sweep_sparse_snr.png" in page_html
    assert "good / flagged" in page_html
    status, ctype, png = _get(base + "/fig/auto/sweep_sparse_snr.png")
    assert status == 200 and png[:4] == b"\x89PNG"
    summ = json.loads((app.reruns_dir / "base" / "summary.json").read_text())
    assert "good" in summ and "flagged" in summ  # every run rescored with the new checks
    table = re.search(r"<table><thead>.*?</table>", page_html, re.S)[0]  # the runs table
    n_head = len(re.findall(r"<th[ >]", table))
    for r in re.findall(r"<tr class=.*?</tr>", table, re.S):
        assert r.count("<td") == n_head, r
    heads = re.findall(r"<th[^>]*>(.*?)</th>", table)
    assert heads.index("good / flagged") == heads.index("units") + 1


def test_thumb_and_flag_column(tiny_rerun):
    from minime.autotune.score import Checks
    from minime.tune import autofig

    run = figures.load_run(tiny_rerun)
    png = autofig.thumb_png(run, 0)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] == img.shape[1] == 4 * (2 * 12 + 1)
    table = figures.unit_table(run, None, Checks(7))
    assert "flag" in table and len(table) == run.A.shape[0]


def test_page_shows_flagged_gallery(tune_server, monkeypatch):
    app, base = tune_server
    _wait(base)
    if app.checks() is None:  # test order: run auto-tune if the roundtrip test has not
        monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL_GRIDS)
        _post(base + "/api/autotune", {"diameter": "7"})
        _wait(base, timeout=300)
    page_html = _get(base + "/?run=auto")[2].decode()
    assert "Flagged units" in page_html and "<th>flag</th>" in page_html
    status, ctype, png = _get(base + "/thumb/auto/0.png")
    assert status == 200 and ctype == "image/png"
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/thumb/auto/99999.png")
    assert e.value.code == 404


def test_maxproj_png_native_size(tune_server):
    app, base = tune_server
    status, ctype, png = _get(base + "/maxproj.png")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert status == 200 and ctype == "image/png"
    assert img.shape == (app.session.fac.H, app.session.fac.W_) and img.dtype == np.uint8


def test_page_has_diameter_measure(tune_server):
    _, base = tune_server
    page_html = _get(base + "/")[2].decode()
    assert 'id="mpimg" src="/maxproj.png"' in page_html and 'id="mpline"' in page_html


def test_server_rejects_bad_diameter_and_reserved_name(tune_server):
    _, base = tune_server
    _wait(base)
    for d in ("0", "1000"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/autotune", {"diameter": d})
        assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/run", {"name": "auto"})
    assert e.value.code == 400


def test_proposal_marks_only_numerically_different_values(tmp_path):
    from types import SimpleNamespace

    from minime.autotune import sweep as sw
    from minime.tune import page

    app = SimpleNamespace(reruns_dir=tmp_path, figure_path=lambda run, fig: None)
    run_values = {k: 1.0 for k in sw.SEARCHED}
    proposed = {k: 1.0 for k in sw.SEARCHED}
    run_values["merge_corr"], proposed["merge_corr"] = 3, 3.0
    run_values["inita_corr"], proposed["inita_corr"] = 3, 4.0
    doc = {
        "score": {"good": 1, "flagged": 0, **{r: 0 for r in score.REASONS}},
        "run_values": run_values,
        "proposed": proposed,
        "at_edge": [],
        "derived": {},
        "diameter": 7.0,
        "n_trials": 1,
        "wall_s": 1.0,
    }
    d = tmp_path / at.AUTO
    d.mkdir()
    (d / at.DOC).write_text(json.dumps(doc))
    html = page._proposal(app)
    row = lambda k: re.search(rf"<tr><td class='l'><b>{k}</b>.*?</tr>", html, re.S)[0]  # noqa: E731
    assert 'class="changed"' not in row("merge_corr")
    assert 'class="changed"' in row("inita_corr")
