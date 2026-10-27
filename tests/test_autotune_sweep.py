import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from typer.testing import CliRunner

from minime import cli, window2
from minime.autotune import run as at
from minime.autotune import score, sweep
from minime.params import Params
from minime.report import StageTimer

D = 7.0  # synthetic cells: Gaussian sigma 3 px
SMALL = sweep.Grids(
    seeds_diff_thres=(2.0, 3.0),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_snr=(0.5, 1.0),
    loop_merge_corr=(0.9,),
    n_confirm=2,
    max_extend=0,
)


def _trial(score_: int | None, **values) -> sweep.Trial:
    sc = None if score_ is None else {"score": score_, "good": score_, "flagged": 0}
    return sweep.Trial("seeds", values, {}, sc, None, 0.0)


def test_derived_params_round_and_scale():
    p = sweep.derived_params(Params(), 7.5)
    assert (p.seeds_max_wnd, p.inita_wnd, p.dl_wnd, p.merge_dist) == (8, 8, 8, 8)
    assert p.size_thres == (11, None)  # 0.25 x pi * 3.75^2 = 44.2; no upper bound
    tiny = sweep.derived_params(Params(), 0.6)
    assert tiny.dl_wnd == 1 and tiny.size_thres[0] >= 1
    with pytest.raises(ValueError, match="diameter"):
        sweep.derived_params(Params(), 0)


def test_best_prefers_score_then_fewer_changes_then_order():
    base = Params()
    a = _trial(5, merge_corr=0.9, pnr_thres=base.pnr_thres)
    b = _trial(5, merge_corr=base.merge_corr, pnr_thres=base.pnr_thres)
    c = _trial(7, merge_corr=0.7, pnr_thres=2.0)
    assert sweep.best([a, b], base, "seeds") is b
    assert sweep.best([a, b, c], base, "seeds") is c
    assert sweep.best([_trial(None), a], base, "seeds") is a
    with pytest.raises(RuntimeError, match="seeds"):
        sweep.best([_trial(None)], base, "seeds")


def test_at_edge():
    assert sweep.at_edge(0.002, (0.002, 0.005, 0.01))
    assert not sweep.at_edge(0.005, (0.002, 0.005, 0.01))
    assert sweep.at_edge(1.0, ("auto", 1.0, 2.0))
    assert not sweep.at_edge("auto", ("auto", 1.0, 2.0))
    assert not sweep.at_edge(0.8, (0.8,))  # one-value grid: no edge


def test_sweep_runs_every_step(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    seen: list[sweep.Trial] = []
    sw = sweep.Sweep(sess, score.Checks(D), SMALL, on_trial=seen.append)
    prop = sw.run()
    steps = [t.step for t in sw.trials]
    assert set(steps) == {"run", "seeds", "merge", "inita", "confirm", "sparse", "loop_merge"}
    assert seen == sw.trials
    assert len(sw.trials) <= SMALL.n_trials()
    assert prop.dl_wnd == 5 and prop.size_thres == (9, None)  # the tiny run's tuned sizes
    assert prop.sparse_snr in SMALL.sparse_snr
    assert sw.final is not None and sw.final.score is not None
    for k in window2.FACTOR_PARAMS:
        assert getattr(prop, k) == getattr(sess.base, k)


def test_reused_init_gives_the_same_cnmf(tiny_w2_run):
    """The CNMF grid reuses one init A, C: running it twice must not change the result."""
    sess = window2.Window2Session(tiny_w2_run)
    sw = sweep.Sweep(sess, score.Checks(D), SMALL)
    sw.fc = sess.factor_corr(sw.base.noise_freq, sw.timer)
    A, C = sw._init(sw.base, StageTimer())
    A0, C0 = A.copy(), C.copy()
    r1 = sw._cnmf(sw.base, StageTimer(), A, C)
    r2 = sw._cnmf(sw.base, StageTimer(), A, C)
    assert np.array_equal(A.toarray(), A0.toarray()) and np.array_equal(C, C0)
    assert np.array_equal(r1[0].toarray(), r2[0].toarray()) and np.array_equal(r1[1], r2[1])


def test_failed_settings_are_recorded_and_skipped(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    grids = dataclasses.replace(SMALL, pnr_thres=("auto", 1e9))
    sw = sweep.Sweep(sess, score.Checks(D), grids)
    sw.run()
    failed = [t for t in sw.trials if t.error]
    assert failed and all("no seeds survived" in t.error for t in failed)
    assert all(t.values["pnr_thres"] == 1e9 for t in failed)


def test_all_failed_step_raises(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    grids = dataclasses.replace(SMALL, pnr_thres=(1e9,))
    with pytest.raises(RuntimeError, match="seeds.*no seeds survived"):
        sweep.Sweep(sess, score.Checks(D), grids).run()


def test_autotune_writes_a_rerun_whose_score_matches(tiny_w2_run):
    out = at.autotune(tiny_w2_run, D, grids=SMALL, name="auto_t")
    assert out == tiny_w2_run / "reruns" / "auto_t"
    doc = json.loads((out / "autotune.json").read_text())
    assert doc["diameter"] == D and set(doc["proposed"]) == set(sweep.SEARCHED)
    assert doc["derived"]["dl_wnd"] == 5 and "dl_wnd" in doc["kept"]
    assert set(doc["at_edge"]) <= set(sweep.SEARCHED)
    # the cached sweep and the fresh rerun agree
    A = sps.load_npz(out / "cnmf" / "A.npz")
    YrA = np.load(out / "cnmf" / "YrA.npy")
    fresh = score.summarize(score.unit_flags(A, YrA, 64, 64, at.checks_of(doc)))
    assert fresh == doc["score"]
    table = pd.read_csv(out / "sweep.csv")
    assert {"step", *sweep.SEARCHED, "good", "flagged", "score", "units_final"} <= set(
        table.columns
    )
    assert len(table) == doc["n_trials"]
    same = table["step"].isin(["run", "sparse", "loop_merge"])
    for k, v in doc["proposed"].items():
        col = table[k]
        same &= (col == v) if isinstance(v, str) else (pd.to_numeric(col, errors="coerce") == v)
    assert same.any(), "no sweep row matches the proposal"
    row = table[same].iloc[0]
    assert {k: int(row[k]) for k in ("good", "flagged", "score")} == {
        k: doc["score"][k] for k in ("good", "flagged", "score")
    }
    flags = pd.read_csv(out / "unit_flags.csv")
    assert len(flags) == A.shape[0] and "good" in flags


def test_cli_autotune(tiny_w2_run, monkeypatch):
    monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL)
    res = CliRunner().invoke(
        cli.app, ["autotune", str(tiny_w2_run), "--diameter", "7", "--name", "auto_cli"]
    )
    assert res.exit_code == 0, res.output
    assert "sparse_snr" in res.output and "good" in res.output
    assert (tiny_w2_run / "reruns" / "auto_cli" / "autotune.json").exists()


@pytest.mark.parametrize("d", ["0", "1000"])
def test_cli_autotune_rejects_bad_diameter(tiny_w2_run, d):
    res = CliRunner().invoke(cli.app, ["autotune", str(tiny_w2_run), "--diameter", d])
    assert res.exit_code == 1 and "diameter" in res.output


def test_check_diameter_bounds():
    at.check_diameter(7.0, 64, 64)
    for d in (float("inf"), float("nan"), 65.0):
        with pytest.raises(ValueError, match="diameter"):
            at.check_diameter(d, 64, 64)


def test_cli_cnmf_rejects_the_reserved_auto_name(tiny_w2_run):
    res = CliRunner().invoke(cli.app, ["cnmf", str(tiny_w2_run), "--name", "auto"])
    assert res.exit_code == 1 and "reserved" in res.output
    assert not (tiny_w2_run / "reruns" / "auto").exists()
