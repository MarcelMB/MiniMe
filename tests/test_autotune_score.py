import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from scipy.signal import lfilter

from caanfast.autotune import score

H = W = 50


def _footprints(blobs: list[list[tuple[int, int, int]]]) -> sps.csr_matrix:
    """One row per unit: the union of square blobs ``(row, col, half-side)``, value 1."""
    rows = []
    for unit in blobs:
        img = np.zeros((H, W), np.float32)
        for r, c, s in unit:
            img[r - s : r + s + 1, c - s : c + s + 1] = 1.0
        rows.append(img.ravel())
    return sps.csr_matrix(np.array(rows))


def _transient(rng: np.random.Generator, T: int = 2000, amp: float = 20.0) -> np.ndarray:
    s = np.zeros(T)
    s[rng.choice(T, 10, replace=False)] = amp
    return lfilter([1.0], [1.0, -0.9], s)


def _good_traces(K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([_transient(rng) + rng.normal(0, 1, 2000) for _ in range(K)])


CH = score.Checks(diameter=6)  # cell area 28.3 px -> size 8.5-84.8 px


def test_pnr_flags_noise_and_flat_traces():
    rng = np.random.default_rng(1)
    A = _footprints([[(10, 10, 2)], [(10, 30, 2)], [(30, 10, 2)]])
    traces = np.vstack([_good_traces(1)[0], rng.normal(0, 1, 2000), np.zeros(2000)])
    f = score.unit_flags(A, traces, H, W, CH)
    assert f["low_pnr"].tolist() == [False, True, True]
    assert score.trace_pnr(np.zeros((1, 100)))[0] == 0.0  # flat: 0, not inf


def test_size_and_pieces():
    A = _footprints(
        [
            [(10, 10, 2)],  # 25 px: fine
            [(10, 30, 0)],  # 1 px: too small
            [(30, 25, 6)],  # 169 px: too large
            [(40, 5, 2), (40, 30, 2)],  # two equal pieces
            [(20, 40, 2), (20, 46, 0)],  # one piece + a 1-px speck (25 / 26 > 0.9)
        ]
    )
    f = score.unit_flags(A, _good_traces(5), H, W, CH)
    assert f["bad_size"].tolist() == [False, True, True, False, False]
    assert f["in_pieces"].tolist() == [False, False, False, True, False]
    assert f.loc[3, "main_piece"] == pytest.approx(0.5)


def test_duplicate_flags_the_weaker_correlated_neighbour():
    rng = np.random.default_rng(2)
    A = _footprints([[(10, 10, 2)], [(10, 10, 2)], [(10, 14, 2)], [(40, 40, 2)]])
    t0 = _transient(rng)
    traces = np.vstack(
        [
            t0 + rng.normal(0, 0.5, 2000),
            0.5 * t0 + rng.normal(0, 0.5, 2000),  # same cell, weaker
            _transient(rng) + rng.normal(0, 0.5, 2000),  # neighbour, other cell
            t0 + rng.normal(0, 0.5, 2000),  # same trace but far away
        ]
    )
    f = score.unit_flags(A, traces, H, W, CH)
    assert f["duplicate"].tolist() == [False, True, False, False]


def test_empty_and_degenerate_units():
    empty = score.unit_flags(sps.csr_matrix((0, H * W)), np.zeros((0, 100)), H, W, CH)
    assert score.summarize(empty) == {
        "good": 0,
        "flagged": 0,
        "score": 0,
        "low_pnr": 0,
        "in_pieces": 0,
        "bad_size": 0,
        "duplicate": 0,
    }
    A = sps.vstack([_footprints([[(10, 10, 2)]]), sps.csr_matrix((1, H * W))]).tocsr()
    f = score.unit_flags(A, _good_traces(2), H, W, CH)  # second row has no pixels
    assert f["good"].tolist() == [True, False]


def test_summarize_counts_reasons():
    f = pd.DataFrame(
        {
            "low_pnr": [False, True, True],
            "in_pieces": [False, False, True],
            "bad_size": [False, False, False],
            "duplicate": [False, False, False],
        }
    )
    f["good"] = ~f[list(score.REASONS)].any(axis=1)
    assert score.summarize(f) == {
        "good": 1,
        "flagged": 2,
        "score": -1,
        "low_pnr": 2,
        "in_pieces": 1,
        "bad_size": 0,
        "duplicate": 0,
    }


@pytest.mark.parametrize("d", [0, -3])
def test_checks_reject_bad_diameter(d):
    with pytest.raises(ValueError, match="diameter"):
        score.Checks(diameter=d)
