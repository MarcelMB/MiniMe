import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from conftest import TINY
from typer.testing import CliRunner

from cafast import cli, window2
from cafast.params import Params

CNMF_FILES = ("A.npz", "C.npy", "YrA.npy", "b.npy", "f.npy")


def _load(path, name):
    return sps.load_npz(path).toarray() if name.endswith(".npz") else np.load(path)


def test_rerun_with_run_params_reproduces_full_run(tiny_run):
    out = window2.rerun(tiny_run, name="same")
    assert out == tiny_run / "reruns" / "same"
    for name in CNMF_FILES:
        a, b = _load(tiny_run / "cnmf" / name, name), _load(out / "cnmf" / name, name)
        assert np.array_equal(a, b), name
    full = json.loads((tiny_run / "report.json").read_text())["counts"]
    rep = json.loads((out / "report.json").read_text())
    assert rep["counts"] == full
    assert rep["source_run"] == str(tiny_run.resolve())
    units = pd.read_csv(out / "cnmf" / "units.csv")
    assert len(units) == full["units_final"]


def test_rerun_writes_every_stage(tiny_run):
    out = window2.rerun(tiny_run, name="stages")
    st = out / "stages"
    for name in ("seeds_init", "seeds_pnr", "seeds_merged"):
        assert len(pd.read_csv(st / f"{name}.csv")) > 0, name
    assert {"pnr", "mask_pnr"} <= set(pd.read_csv(st / "seeds_pnr.csv").columns)
    stages = window2.footprint_stages(out)
    assert list(stages) == [
        "initA",
        "init_merge",
        "spatial_0",
        "merge_0",
        "spatial_1",
        "merge_1",
    ]
    final = sps.load_npz(out / "cnmf" / "A.npz")
    assert np.array_equal(stages["merge_1"].toarray(), final.toarray())
    counts = json.loads((out / "report.json").read_text())["counts"]
    assert stages["initA"].shape[0] == counts["units_initA"]
    # the rerun looks like a run directory to the tools that read one
    assert (out / "factors" / "meta.json").exists()
    assert (out / "motion" / "arrays.npz").exists()


def test_rerun_applies_window2_overrides(tiny_run, tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"sparse_penal": 0.1, "n_iter": 1}))
    prm = window2.run_params(tiny_run, tmp_path / "p.json")
    assert prm.sparse_penal == 0.1 and prm.n_iter == 1
    assert prm.tile == TINY.tile and prm.dl_wnd == TINY.dl_wnd  # rest from the run
    out = window2.rerun(tiny_run, prm, name="sp")
    saved = Params.from_json(out / "params.json")
    assert saved == prm
    counts = json.loads((out / "report.json").read_text())["counts"]
    assert "units_after_spatial_1" not in counts


@pytest.mark.parametrize(
    "change", [{"tile": 16}, {"median_ksize": 5}, {"motion_bandpass": (2.0, 8.0)}]
)
def test_rerun_refuses_changed_factor_params(tiny_run, change):
    prm = window2.run_params(tiny_run)
    bad = Params(**{**prm.asdict(), **change})
    with pytest.raises(ValueError, match=next(iter(change))):
        window2.rerun(tiny_run, bad, name="bad")
    assert not (tiny_run / "reruns" / "bad").exists()


def test_rerun_accepts_seed_windows_only_if_stored_blocks_tile_them(tiny_run):
    prm = window2.run_params(tiny_run)
    ok = Params(**{**prm.asdict(), "seeds_wnd": 100, "seeds_stp": 50})
    window2.check_factor_params(window2.load_factors(tiny_run), ok)
    bad = Params(**{**prm.asdict(), "seeds_wnd": 77})
    with pytest.raises(ValueError, match="seeds_wnd"):
        window2.check_factor_params(window2.load_factors(tiny_run), bad)


def test_execution_params_may_differ(tiny_run):
    prm = window2.run_params(tiny_run)
    window2.check_factor_params(
        window2.load_factors(tiny_run), Params(**{**prm.asdict(), "workers": 7, "sub_batch": 3})
    )


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b"])
def test_rerun_rejects_unsafe_names(tiny_run, name):
    with pytest.raises(ValueError, match="run name"):
        window2.rerun(tiny_run, name=name)


def test_cli_cnmf_reruns_window2(tiny_run, tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"loop_merge_corr": 0.95}))
    res = CliRunner().invoke(
        cli.app, ["cnmf", str(tiny_run), "--params", str(tmp_path / "p.json"), "--name", "cli"]
    )
    assert res.exit_code == 0, res.output
    out = tiny_run / "reruns" / "cli"
    assert str(out) in res.output and "units_final" in res.output
    assert Params.from_json(out / "params.json").loop_merge_corr == 0.95


def test_cli_cnmf_reports_refused_params(tiny_run, tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"tophat_wnd": 3}))
    res = CliRunner().invoke(cli.app, ["cnmf", str(tiny_run), "--params", str(tmp_path / "p.json")])
    assert res.exit_code != 0
    assert "tophat_wnd" in res.output
