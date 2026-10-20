import dataclasses

import numpy as np
import pytest

from cafast import window2
from cafast.autotune import score, sweep
from cafast.params import Params
from cafast.report import StageTimer

D = 7.0  # synthetic cells: Gaussian sigma 3 px
SMALL = sweep.Grids(
    seeds_diff_thres=(2.0, 3.0),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_penal=(0.003, 0.01),
    loop_merge_corr=(0.9,),
    n_confirm=2,
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


def test_sweep_runs_every_step(tiny_run):
    sess = window2.Window2Session(tiny_run)
    seen: list[sweep.Trial] = []
    sw = sweep.Sweep(sess, score.Checks(D), SMALL, on_trial=seen.append)
    prop = sw.run()
    steps = [t.step for t in sw.trials]
    assert steps.count("seeds") == 4
    assert steps.count("merge") == 1 and steps.count("inita") == 1
    assert 1 <= steps.count("confirm") <= 2
    assert steps.count("sparse") == 2 and steps.count("loop_merge") == 1
    assert seen == sw.trials
    assert len(sw.trials) <= SMALL.n_trials()
    assert prop.dl_wnd == 7 and prop.size_thres == sweep.derived_params(sess.base, D).size_thres
    assert prop.sparse_penal in SMALL.sparse_penal
    assert sw.final is not None and sw.final.score is not None
    for k in window2.FACTOR_PARAMS:
        assert getattr(prop, k) == getattr(sess.base, k)


def test_reused_init_gives_the_same_cnmf(tiny_run):
    """The CNMF grid reuses one init A, C: running it twice must not change the result."""
    sess = window2.Window2Session(tiny_run)
    sw = sweep.Sweep(sess, score.Checks(D), SMALL)
    sw.fc = sess.factor_corr(sw.base.noise_freq, sw.timer)
    A, C = sw._init(sw.base, StageTimer())
    A0, C0 = A.copy(), C.copy()
    r1 = sw._cnmf(sw.base, StageTimer(), A, C)
    r2 = sw._cnmf(sw.base, StageTimer(), A, C)
    assert np.array_equal(A.toarray(), A0.toarray()) and np.array_equal(C, C0)
    assert np.array_equal(r1[0].toarray(), r2[0].toarray()) and np.array_equal(r1[1], r2[1])


def test_failed_settings_are_recorded_and_skipped(tiny_run):
    sess = window2.Window2Session(tiny_run)
    grids = dataclasses.replace(SMALL, pnr_thres=("auto", 1e9))
    sw = sweep.Sweep(sess, score.Checks(D), grids)
    sw.run()
    failed = [t for t in sw.trials if t.error]
    assert failed and all("no seeds survived" in t.error for t in failed)
    assert all(t.values["pnr_thres"] == 1e9 for t in failed)


def test_all_failed_step_raises(tiny_run):
    sess = window2.Window2Session(tiny_run)
    grids = dataclasses.replace(SMALL, pnr_thres=(1e9,))
    with pytest.raises(RuntimeError, match="seeds"):
        sweep.Sweep(sess, score.Checks(D), grids).run()
