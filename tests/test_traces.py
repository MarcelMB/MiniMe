import sys

import numpy as np

from minime import traces


def test_empty_stays_in_ram_unless_spilling(tmp_path, monkeypatch):
    assert type(traces.empty((3, 5))) is np.ndarray
    monkeypatch.setattr(traces, "RAM_FRACTION", 0.0)
    assert type(traces.empty((3, 5))) is np.ndarray  # no spill_to block: never spills
    with traces.spill_to(tmp_path / "spill"):
        X = traces.empty((3, 5))
    assert isinstance(X, np.memmap) and X.dtype == np.float32 and X.shape == (3, 5)
    X[:] = np.arange(15).reshape(3, 5)
    assert X.sum() == 105
    if sys.platform != "win32":  # POSIX: the temporary file has no name from the start
        assert not any((tmp_path / "spill").iterdir())


def test_spill_stays_in_ram_below_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(traces, "total_ram", lambda: 16 * 1000 * 4)
    with traces.spill_to(tmp_path):
        assert type(traces.empty((10, 100))) is np.ndarray  # 4000 B = 1/16 of 64000
        assert isinstance(traces.empty((11, 100)), np.memmap)


def test_scaled_rows_and_all_finite_in_blocks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 50)).astype(np.float32)
    rows, s = np.array([4, 1, 3]), np.array([2.0, 0.5, 3.0], dtype=np.float32)
    assert np.array_equal(traces.scaled_rows(X, rows, s, chunk=7), X[rows] * s[:, None])
    assert traces.all_finite(X) and traces.all_finite(np.ones(4))
    X[2, 45] = np.nan
    assert not traces.all_finite(X) and traces.all_finite(X[:, :45])
