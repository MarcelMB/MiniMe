"""The time-subset search: ``Factors.subset``, the middle segment, and the decision on the
whole session."""

import dataclasses
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from minime import cli, corr, seeds, window2
from minime.autotune import run as at
from minime.autotune import score, sweep

D = 7.0
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


def test_subset_slices_time_and_keeps_the_product(tiny_w2_run):
    fac = window2.load_factors(tiny_w2_run)
    e = fac.block_edges
    t0, t1 = int(e[1]), int(e[-2])
    sub = fac.subset(t0, t1)
    assert (sub.T, sub.t0, sub.session_T) == (t1 - t0, t0, fac.T)
    i0, i1 = 1, len(e) - 2
    assert np.array_equal(sub.block_max, fac.block_max[i0:i1])
    assert np.array_equal(sub.block_edges, e[i0 : i1 + 1] - t0)
    assert np.array_equal(sub.max_proj, fac.block_max[i0:i1].max(axis=0))
    pix = np.arange(0, fac.P, 97)
    assert np.allclose(sub.traces(pix), fac.traces(pix)[:, t0:t1], atol=1e-3)
    for a, b in zip(sub.tiles, fac.tiles, strict=True):
        assert np.allclose(a.V.T @ a.V, np.eye(a.rank), atol=1e-5)  # orthonormal again
        assert a.mean is b.mean and a.sigma is b.sigma
    # the whole session as a subset: same energy, same correlations
    whole = fac.subset(0, fac.T)
    assert np.allclose(whole.energy_flat(), fac.energy_flat(), rtol=1e-4)
    fc, fw = corr.FactorCorr(fac, 0.06), corr.FactorCorr(whole, 0.06)
    assert np.allclose(fw.norm2, fc.norm2, rtol=1e-4)
    with pytest.raises(ValueError, match="block edges"):
        fac.subset(int(np.setdiff1d(np.arange(fac.T), e)[0]), t1)


def test_subset_seeds_use_the_sessions_windows_inside_it(tiny_w2_run):
    fac = window2.load_factors(tiny_w2_run)
    p = window2.run_params(tiny_w2_run)
    e = fac.block_edges
    sub = fac.subset(int(e[2]), int(e[-3]))
    inside = [
        w
        for w in seeds.rolling_windows(fac.T, p.seeds_wnd, p.seeds_stp)
        if sub.t0 <= w.start and w.stop <= sub.t0 + sub.T
    ]
    args = (p.seeds_wnd, p.seeds_stp, p.seeds_max_wnd, 2)
    s = seeds.seeds_init(sub.block_max, sub.block_edges, sub.T, *args, sub.t0, sub.session_T)
    assert inside and s["seeds"].max() <= len(inside)
    full = seeds.seeds_init(fac.block_max, fac.block_edges, fac.T, *args)
    same = seeds.seeds_init(fac.block_max, fac.block_edges, fac.T, *args, 0, fac.T)
    assert full.equals(same)
    with pytest.raises(ValueError, match="seed window"):
        seeds.seeds_init(sub.block_max[:1], sub.block_edges[:2], 10, *args, 0, fac.T)


def test_middle_segment():
    edges = np.arange(0, 1001, 50)
    assert sweep.middle_segment(edges, 0) is None
    assert sweep.middle_segment(edges, 1000) is None  # not longer than the segment: whole
    assert sweep.middle_segment(edges, 400) == (300, 700)
    with pytest.raises(ValueError, match="tune_frames"):
        sweep.middle_segment(edges, -1)


def test_sweep_explores_on_the_segment_and_decides_on_the_session(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    sw = sweep.Sweep(sess, score.Checks(D), SMALL, tune_frames=200)
    prop = sw.run()
    t0, t1 = sw.segment
    assert t0 > 0 and t1 < sess.fac.T and 150 <= t1 - t0 <= 250
    assert sw.fac is sess.fac  # back on the session after the decision
    steps = [t.step for t in sw.trials]
    run_values = {k: getattr(sess.base, k) for k in sweep.SEARCHED}
    # every step on the segment, then the run's values and the best settings on the session
    i = steps.index("decide")
    assert set(steps[:i]) == {"run", "seeds", "merge", "inita", "sparse", "loop_merge", "confirm"}
    assert set(steps[i:]) == {"decide"} and len(steps[i:]) <= 1 + SMALL.n_decide
    assert sw.run_trial is sw.trials[i] and sw.run_trial.values == run_values
    assert len(sw.trials) <= SMALL.n_trials()
    assert sw.final.step == "decide"
    assert sw.final.score["score"] >= sw.run_trial.score["score"]
    assert (prop == sess.base) == (not sw.better_than_run)
    # the decision is the whole session's: rerunning the proposal gives the same score
    sw2 = sweep.Sweep(sess, score.Checks(D), SMALL, tune_frames=0)
    sw2.fc = sess.factor_corr(sw2.base.noise_freq, sw2.timer)
    t = sw2._try("check", sw.final.values, sw2._full, base=prop)
    assert t.score == sw.final.score


def test_whole_session_search_has_no_decide_step(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    sw = sweep.Sweep(sess, score.Checks(D), SMALL)  # 300 frames < TUNE_FRAMES
    sw.run()
    assert sw.segment is None and "decide" not in {t.step for t in sw.trials}


def test_decide_keeps_the_run_when_the_search_failed(tiny_w2_run):
    sess = window2.Window2Session(tiny_w2_run)
    grids = dataclasses.replace(SMALL, pnr_thres=(1e9,))
    sw = sweep.Sweep(sess, score.Checks(D), grids, tune_frames=200)
    assert sw.run() == sess.base
    assert sw.final is sw.run_trial and sw.run_trial.step == "decide"
    assert sw.search_error and not sw.better_than_run


def test_cli_autotune_tune_frames(tiny_w2_run, monkeypatch):
    monkeypatch.setattr(at, "DEFAULT_GRIDS", SMALL)
    args = ["autotune", str(tiny_w2_run), "--diameter", "7", "--name", "auto_seg"]
    res = CliRunner().invoke(cli.app, [*args, "--tune-frames", "200"])
    assert res.exit_code == 0, res.output
    doc = json.loads((tiny_w2_run / "reruns" / "auto_seg" / at.DOC).read_text(encoding="utf-8"))
    assert doc["tune_frames"] == 200 and doc["frames"] == 300
    t0, t1 = doc["segment"]
    assert f"explored on frames {t0}-{t1} of 300" in res.output
    # the rerun's score is the decision's
    assert doc["score"]["score"] >= doc["run_score"]["score"]
    res = CliRunner().invoke(cli.app, [*args, "--tune-frames", "-5"])
    assert res.exit_code == 1 and "tune_frames" in res.output
