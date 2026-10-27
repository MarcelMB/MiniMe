import json
from pathlib import Path

import numpy as np
from scipy import sparse

from minime import score
from minime.testing import synth


def test_match_units_greedy_one_to_one() -> None:
    true = np.array([[10, 10], [20, 20], [40, 40]], float)
    units = np.array([[10.5, 10], [11, 11], [21, 20], [80, 80]], float)
    ui, ti = score.match_units(units, true, 5.0)
    assert set(zip(ui.tolist(), ti.tolist(), strict=True)) == {(0, 0), (2, 1)}
    s = score.summarise(ui, ti, n_units=4, n_true=3)
    assert s["recall"] == 2 / 3 and s["precision"] == 0.5 and s["n_matched"] == 2


def test_match_units_empty() -> None:
    ui, ti = score.match_units(np.zeros((0, 2)), np.array([[1.0, 1.0]]))
    assert len(ui) == 0 and len(ti) == 0
    assert score.summarise(ui, ti, 0, 1)["f1"] == 0.0


def test_footprint_centroids() -> None:
    A = np.zeros((2, 6 * 8), np.float32)
    A[0, 2 * 8 + 3] = 1.0
    A[0, 4 * 8 + 3] = 1.0
    A[1, 5 * 8 + 7] = 2.0
    c = score.footprint_centroids(sparse.csr_matrix(A), 6, 8)
    assert np.allclose(c, [[3.0, 3.0], [5.0, 7.0]])


def _pixel_units(setup: synth.SynthSetup, offset: np.ndarray, h: int, w: int):
    cents = np.rint(synth.true_centroids(setup) + offset).astype(int)
    keep = np.flatnonzero((cents >= 0).all(1) & (cents[:, 0] < h) & (cents[:, 1] < w))
    A = sparse.lil_matrix((len(keep), h * w), dtype=np.float32)
    for r, j in enumerate(keep):
        A[r, cents[j, 0] * w + cents[j, 1]] = 1.0
    return A.tocsr(), keep


def _meta(tmp_path: Path, h: int, w: int, T: int) -> Path:
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"frames": T, "height": h, "width": w, "seed": 0}))
    return p


def test_score_run_recovers_offset_and_sign(tmp_path: Path) -> None:
    h = w = 64
    T = 300
    setup = synth.synth_setup(h, w, T, 0)
    meta = _meta(tmp_path, h, w, T)
    c = np.array([3, -2])
    for s in (1, -1):
        A, keep = _pixel_units(setup, s * c, h, w)
        traces = (setup.C[keep] - 20.0).astype(np.float32)
        out = score.score_run(A, traces, h, w, meta, setup.shifts + c)
        assert out["n_matched"] == len(keep) and out["n_units"] == len(keep)
        assert out["offset_sign_used"] == s
        assert out["alignment_ok"] is True
        assert out["median_match_dist_px"] < 0.5
        assert out["trace_corr_median"] > 0.999
        assert out["recall"] == len(keep) / out["n_true"]
        assert out["precision"] == 1.0


def test_alignment_not_ok_when_both_signs_bad(tmp_path: Path, capsys) -> None:
    h = w = 64
    setup = synth.synth_setup(h, w, 300, 0)
    meta = _meta(tmp_path, h, w, 300)
    # units 3 px off every true cell and shifts that give c = 0: neither sign can fix it
    A, keep = _pixel_units(setup, np.array([3, 0]), h, w)
    traces = (setup.C[keep] - 20.0).astype(np.float32)
    out = score.score_run(A, traces, h, w, meta, setup.shifts)
    assert out["alignment_ok"] is False
    assert out["median_match_dist_px"] >= score.SELF_CHECK_PX
    assert np.isfinite(out["f1"]) and np.isfinite(out["trace_corr_median"])
    assert "alignment self-check failed" in capsys.readouterr().err


def test_alignment_not_ok_with_zero_matches(tmp_path: Path) -> None:
    h = w = 2000  # bigger than the session so the unit sits far from every true cell
    setup = synth.synth_setup(64, 64, 300, 0)
    meta = _meta(tmp_path, 64, 64, 300)
    A = sparse.csr_matrix(([1.0], ([0], [1500 * w + 1500])), shape=(1, h * w), dtype=np.float32)
    out = score.score_run(A, np.zeros((1, 300), np.float32), h, w, meta, setup.shifts)
    assert out["alignment_ok"] is False and out["n_matched"] == 0
    assert out["f1"] == 0.0 and out["median_match_dist_px"] is None
    assert out["trace_corr_median"] is None
