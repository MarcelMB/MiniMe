"""The search of the sweep: grids built from the run's values, the run's parameters as a
candidate, derived sizes only where the run uses defaults, and grid extension at an edge."""

import dataclasses

import numpy as np
import pytest

from minime import window2
from minime.autotune import run as at
from minime.autotune import score, sweep
from minime.params import Params

D = 7.0


def _trial(score_: int | None, step: str = "seeds", **values) -> sweep.Trial:
    sc = None if score_ is None else {"score": score_, "good": score_, "flagged": 0}
    return sweep.Trial(step, values, {}, sc, None, 0.0)


# --- grids from the run's values ------------------------------------------------------


def test_relative_grids_contain_the_runs_value():
    base = Params(
        seeds_diff_thres=8,
        pnr_thres=2.0,
        merge_corr=0.75,
        init_merge_corr=0.7,
        inita_corr=0.8,
        sparse_snr=0.6,
        loop_merge_corr=0.7,
    )
    g = sweep.relative_grids(base)
    for k in sweep.SEARCHED:
        assert getattr(base, k) in getattr(g, k), k
    assert g.seeds_diff_thres == (4, 6, 8, 11, 16)
    assert g.pnr_thres == ("auto", 1.0, 1.4, 2.0, 2.8, 4.0)
    assert g.merge_corr == (0.65, 0.7, 0.75, 0.8, 0.85)
    assert g.sparse_snr == (0.3, 0.42, 0.6, 0.84, 1.2)


def test_relative_grids_clip_correlations_and_keep_integer_thresholds():
    g = sweep.relative_grids(Params(loop_merge_corr=0.95, seeds_diff_thres=3))
    assert g.loop_merge_corr == (0.85, 0.9, 0.95, 0.98)
    assert g.seeds_diff_thres == (2, 3, 4, 6)  # 1.5 and 2.1 both round to 2: once
    assert sweep.relative_grids(Params(seeds_diff_thres=1)).seeds_diff_thres[0] == 1


def test_relative_grids_for_auto_pnr_use_absolute_values():
    g = sweep.relative_grids(Params(pnr_thres="auto"))
    assert g.pnr_thres[0] == "auto" and all(isinstance(v, float) for v in g.pnr_thres[1:])


def test_relative_grids_keep_fractional_thresholds_for_float_block_max():
    g = sweep.relative_grids(Params(seeds_diff_thres=2.5), integer_diff=False)
    assert 2.5 in g.seeds_diff_thres and g.seeds_diff_thres[0] == 1.25


def test_extend_grid_steps_outward():
    assert sweep.extend(sweep.SCALE, (4, 6, 8, 11, 16), 16, integer=True) == 23
    assert sweep.extend(sweep.SCALE, (4, 6, 8, 11, 16), 4, integer=True) == 3
    assert sweep.extend(sweep.SCALE, (1, 2, 4), 1, integer=True) is None  # cannot go lower
    assert sweep.extend(sweep.CORR_OFFSETS, (0.65, 0.7, 0.75), 0.75, corr=True) == 0.8
    assert sweep.extend(sweep.CORR_OFFSETS, (0.9, 0.95, 0.98), 0.98, corr=True) is None
    assert sweep.extend(sweep.SCALE, (0.3, 0.6, 1.2), 0.3) == pytest.approx(0.214, abs=1e-3)


# --- derived sizes only where the run uses the defaults -------------------------------


def test_derived_params_keep_tuned_values():
    base = Params(merge_dist=3, size_thres=(10, None))  # tuned; the other sizes default
    p = sweep.derived_params(base, 8.0)
    assert (p.merge_dist, p.size_thres) == (3, (10, None))
    assert (p.seeds_max_wnd, p.inita_wnd, p.dl_wnd) == (8, 8, 8)
    assert sweep.derived_keys(base, 8.0) == ["seeds_max_wnd", "inita_wnd", "dl_wnd"]


# --- confirm candidates across merge values -------------------------------------------


def test_confirm_candidates_cover_distinct_merge_values():
    base = Params()
    ts = [
        _trial(10, "merge", merge_corr=0.8, inita_corr=0.8),
        _trial(9, "inita", merge_corr=0.8, inita_corr=0.7),
        _trial(8, "inita", merge_corr=0.8, inita_corr=0.6),
        _trial(5, "merge", merge_corr=0.9, inita_corr=0.8),
        _trial(2, "merge", merge_corr=0.7, inita_corr=0.8),
    ]
    c = sweep.confirm_candidates(ts, base, 3)
    assert [t.values["merge_corr"] for t in c] == [0.8, 0.9, 0.7]
    c = sweep.confirm_candidates(ts, base, 5)
    assert [t.score["score"] for t in c] == [10, 5, 2, 9, 8]
    assert sweep.confirm_candidates([_trial(None, "merge", merge_corr=0.8)], base, 3) == []


# --- the sweep on the tiny run --------------------------------------------------------


SMALL = sweep.Grids(
    seeds_diff_thres=(2, 3),
    pnr_thres=("auto", 1.0),
    merge_corr=(0.8,),
    init_merge_corr=(0.8,),
    inita_corr=(0.8,),
    sparse_snr=(0.5, 1.0),
    loop_merge_corr=(0.9,),
    n_confirm=2,
    max_extend=0,
)


def test_sweep_tries_the_runs_parameters_first_and_never_proposes_worse(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    sw = sweep.Sweep(sess, score.Checks(D), SMALL)
    prop = sw.run()
    run_trial = sw.trials[0]
    assert run_trial.step == "run" and run_trial.score is not None
    assert run_trial.values == {k: getattr(sess.base, k) for k in sweep.SEARCHED}
    assert sw.final.score["score"] >= run_trial.score["score"]
    if sw.final is run_trial:
        assert prop == sess.base and not sw.better_than_run
    else:
        assert sw.better_than_run and prop.merge_dist == 7
    # the search base keeps the tiny run's tuned sizes and derives the default ones
    assert sw.base.dl_wnd == sess.base.dl_wnd == 5 and sw.base.merge_dist == 7


def test_sweep_uses_relative_grids_by_default(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    sw = sweep.Sweep(sess, score.Checks(D))
    assert sw.grids == sweep.relative_grids(
        sess.base, integer_diff=np.issubdtype(sess.fac.block_max.dtype, np.integer)
    )
    assert sw.grids.n_trials() == at.n_trials(sess)


def test_sweep_extends_a_grid_whose_best_is_at_the_edge(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    grids = dataclasses.replace(SMALL, sparse_snr=(0.5, 1.0), max_extend=1)
    sw = sweep.Sweep(sess, score.Checks(D), grids)
    sw.run()
    sp = [t for t in sw.trials if t.step == "sparse"]
    vals = sorted({t.values["sparse_snr"] for t in sp})
    best = sweep.best(sp, sw.base, "sparse").values["sparse_snr"]
    if best in (0.5, 1.0) and len(vals) == 2:
        pytest.fail("best at the edge but the grid was not extended")
    assert len(vals) <= 3 and len(sw.trials) <= grids.n_trials()


def test_autotune_doc_records_the_run_trial(tiny_w2_run):
    out = at.autotune(tiny_w2_run, D, grids=SMALL, name="auto_s")
    doc = at.load_doc(out)
    assert set(doc["run_score"]) == set(doc["score"])
    assert doc["better_than_run"] == (doc["score"]["score"] > doc["run_score"]["score"])
    assert doc["kept"] == ["seeds_max_wnd", "inita_wnd", "dl_wnd", "size_thres"]
    assert set(doc["grids"]["seeds_diff_thres"]) >= {2, 3}  # the grids used, extended or not
