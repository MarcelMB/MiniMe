"""The local search: a data-driven start, one step down and up per key, few settings."""

import json

import numpy as np
import pandas as pd
import pytest

from minime import seeds, window2
from minime.autotune import run as at
from minime.autotune import score, sweep
from minime.params import Params

D = 7.0


def test_step_value_scales_thresholds_and_shifts_correlations():
    assert sweep.step_value("sparse_snr", 1.0, up=True) == 1.4
    assert sweep.step_value("sparse_snr", 1.0, up=False) == 0.714
    assert sweep.step_value("merge_corr", 0.8, up=True) == 0.85
    assert sweep.step_value("merge_corr", 0.98, up=True) is None  # clipped to CORR_RANGE
    assert sweep.step_value("seeds_diff_thres", 6, up=True, integer=True) == 8
    assert sweep.step_value("seeds_diff_thres", 2, up=False, integer=True) == 1
    assert sweep.step_value("seeds_diff_thres", 1, up=False, integer=True) is None
    assert sweep.step_value("pnr_thres", "auto", up=True) is None


def test_contrast_knee_finds_where_noise_stops_dominating():
    # noise: count falls 4x per grey level up to 5; from 5 on it falls < 1.5x (cells)
    hist = np.array([4**8, 4**7, 4**6, 4**5, 4**4, 4**3, 50, 40, 32, 26, 21])
    assert seeds.contrast_knee(hist) == 5
    assert seeds.contrast_knee(np.array([1000, 100, 10, 1])) is None


def test_contrast_hist_matches_the_seed_rule(tiny_w2_run):
    """A pixel is a seed candidate at diff iff its contrast > diff: the histogram's tail above
    diff counts the local maxima ``local_max_union`` keeps in the same windows."""
    fac = window2.Window2Session(tiny_w2_run).fac
    p = Params.from_json(tiny_w2_run / "params.json")
    ranges = seeds.window_ranges(fac.block_edges, fac.T, p.seeds_wnd, p.seeds_stp)
    kernels = seeds.disk_kernels(2, p.seeds_max_wnd)
    hist = seeds.contrast_hist(fac.block_max, ranges, kernels)
    for diff in (1, 3):
        n = sum(
            int(
                seeds.local_max_union(
                    seeds.window_projection(fac.block_max, i0, i1), kernels, diff
                ).sum()
            )
            for i0, i1 in ranges
        )
        assert hist[diff + 1 :].sum() == n


def test_local_search_is_bounded_and_never_proposes_worse(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    seen: list[sweep.Trial] = []
    sw = sweep.LocalSearch(sess, score.Checks(D), on_trial=seen.append)
    p = sw.run()
    assert seen[0].step == "run" and seen[1].step == "start"
    assert len(sw.trials) <= sw.n_trials() <= at.n_trials(sess)
    assert set(sw.start) == set(sweep.SEARCHED) and sw.start["pnr_thres"] == sweep.START_PNR
    if sw.better_than_run:
        assert sw.final.score["score"] > sw.run_trial.score["score"]
    else:
        assert p == sess.base
    # each key's tried values hold the start and at most one side beyond the first step back
    for k in sweep.LOCAL_ORDER:
        assert sw.final.values[k] in getattr(sw.grids, k) or not sw.better_than_run


def test_autotune_local_writes_the_proposal_it_scored(tiny_w2_run):
    out = at.autotune(tiny_w2_run, D, name="auto_l")
    doc = json.loads((out / "autotune.json").read_text(encoding="utf-8"))
    assert doc["strategy"] == "local" and set(doc["start"]) == set(sweep.SEARCHED)
    table = pd.read_csv(out / "sweep.csv")
    assert len(table) == doc["n_trials"] <= at.n_trials(window2.Window2Session(tiny_w2_run))
    same = pd.Series(True, index=table.index)
    for k, v in doc["proposed"].items():
        col = table[k]
        same &= (col == v) if isinstance(v, str) else (pd.to_numeric(col, errors="coerce") == v)
    assert same.any(), "no sweep row matches the proposal"
    assert int(table[same].iloc[0]["score"]) == doc["score"]["score"]


def test_a_second_pass_runs_only_after_a_move(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    one = sweep.LocalSearch(sess, score.Checks(D))
    two = sweep.LocalSearch(sess, score.Checks(D), passes=2)
    one.run(), two.run()
    assert one.passes == sweep.LOCAL_PASSES == 1
    assert len(one.trials) <= len(two.trials) <= two.n_trials() <= at.n_trials(sess, passes=2)
    if one.final.values == one.run_trial.values or not one.better_than_run:
        return
    assert two.final.score["score"] >= one.final.score["score"]


def test_autotune_rejects_zero_passes(tiny_w2_run):
    with pytest.raises(ValueError, match="passes"):
        at.autotune(tiny_w2_run, D, name="auto_p", passes=0)
