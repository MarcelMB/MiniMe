import numpy as np
import pytest

from caanfast.serve.traces import lowpass, lowpass_gain, overview, window, zooms


def test_gain_is_one_below_half_cutoff_half_at_cutoff_zero_above():
    g = lowpass_gain(1000, fs=100.0, f_lp=5.0)  # 0.1 Hz bins: 5 Hz is bin 50
    f = np.fft.rfftfreq(1000, 0.01)
    assert np.all(g[f < 2.5] == 1.0)
    assert g[50] == pytest.approx(0.5)
    assert np.all(g[f >= 7.5] == 0.0)
    assert np.all(np.diff(g) <= 1e-12)


def test_lowpass_removes_a_fast_sine_and_keeps_a_slow_one():
    t = np.arange(1000) / 100.0
    slow, fast = np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 12.0 * t)
    y = lowpass(slow + fast, fs=100.0, f_lp=5.0)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, slow, atol=1e-4)


def test_short_traces_are_returned_unchanged():
    x = np.arange(5, dtype=np.float32)
    np.testing.assert_array_equal(lowpass(x, 30.0, 5.0), x)


def test_window_with_margin_matches_the_full_trace_filter():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=6000)).astype(np.float32)
    full = lowpass(x, 30.0, 3.0)
    w = window(x, 2000, 2600, fs=30.0, lowpass_hz=3.0)
    assert w.shape == (600,)
    assert np.max(np.abs(w - full[2000:2600])) < 0.01 * np.std(x)


def test_window_checks_its_arguments():
    x = np.zeros(100, np.float32)
    np.testing.assert_array_equal(window(x, 10, 20), x[10:20])
    with pytest.raises(IndexError):
        window(x, 90, 101)
    with pytest.raises(IndexError):
        window(x, 20, 20)
    with pytest.raises(ValueError):
        window(x, 0, 50, fs=None, lowpass_hz=5.0)


def test_overview_min_max_per_bin():
    C = np.arange(20, dtype=np.float32).reshape(2, 10)
    ov = overview(C, bins=5)
    assert ov.shape == (2, 2, 5)
    np.testing.assert_array_equal(ov[0, 0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(ov[1, 1], [11, 13, 15, 17, 19])
    assert overview(C, bins=50).shape == (2, 2, 10)


def test_zooms_pick_brightest_units_and_spread_windows():
    z = zooms(np.array([1.0, 5.0, 3.0]), T=3 * 3600 * 20, fps=20.0, n_units=2)
    assert z["units"] == [1, 2]
    assert len(z["windows"]) == 9
    assert z["windows"][1] == {"start": 24000, "stop": 30000}
    assert zooms(np.array([1.0]), T=100, fps=None)["windows"] == []
