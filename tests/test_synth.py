import hashlib

import numpy as np

from minime.testing import synth
from minime.testing import synth_effects as fx


def test_synth_movie_is_deterministic_uint8():
    a = synth.synth_movie(32, 32, 20, seed=1)
    b = synth.synth_movie(32, 32, 20, seed=1)
    assert a.dtype == np.uint8 and a.shape == (20, 32, 32)
    assert np.array_equal(a, b)


def test_truth_matches_generator_without_noise():
    h, w, T = 48, 48, 60
    setup = synth.synth_setup(h, w, T, seed=0)
    movie = synth.synth_movie(h, w, T, seed=0)
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, h, w, chunk=25)])
    # noise sd is 4 grey levels; clean must be within a few sd everywhere
    diff = movie.astype(float) - clean.astype(float)
    assert abs(diff.mean()) < 0.5 and diff.std() < 5.0
    assert setup.shifts.shape == (T, 2)


def test_true_centroids_inside_frame():
    setup = synth.synth_setup(64, 64, 10, seed=0)
    c = synth.true_centroids(setup)
    assert c.shape[1] == 2
    assert (c > -10).all() and (c < 74).all()


def test_write_session_layout(tmp_path):
    meta = synth.write_session(tmp_path / "raw", minutes=0.05, fps=30, segment=30, seed=0, side=32)
    assert sorted(p.name for p in (tmp_path / "raw").glob("*.avi")) == ["0.avi", "1.avi", "2.avi"]
    assert meta["frames"] == 90 and (tmp_path / "meta.json").exists()


def _digest(h, w, T, seed, **kw):
    return hashlib.sha256(synth.synth_movie(h, w, T, seed, **kw).tobytes()).hexdigest()[:16]


def test_default_session_is_byte_identical():
    # pinned on 2026-10-27 from the pre-options generator
    assert _digest(64, 64, 300, 0) == "ce39843f43462047"
    assert _digest(32, 32, 20, 1) == "de776615f5097caf"


# --- synth_effects building blocks ------------------------------------------------------


def test_lognormal_factors_mean_one():
    rng = np.random.default_rng(0)
    f = fx.lognormal_factors(rng, 0.3, 20000)
    assert abs(f.mean() - 1.0) < 0.02 and abs(f.std() - 0.3) < 0.02
    assert np.array_equal(fx.lognormal_factors(rng, 0.0, 5), np.ones(5))


def test_smooth_field_range_and_scale():
    f = fx.smooth_field(np.random.default_rng(0), (40, 50), 8.0)
    assert f.shape == (40, 50) and f.dtype == np.float32
    assert f.min() == 0.0 and f.max() == 1.0
    assert np.abs(np.diff(f, axis=1)).max() < 0.2  # smooth


def test_ou_series_is_positive_with_unit_mean():
    s = fx.ou_series(np.random.default_rng(0), 20000, 10.0)
    assert s.shape == (20000,) and (s > 0).all() and abs(s.mean() - 1.0) < 0.05


def test_lowpass_is_causal():
    x = np.zeros((1, 50), np.float32)
    x[0, 10] = 1.0
    y = fx.lowpass(x, 5.0)
    assert (y[0, :10] == 0).all() and y[0, 10] > 0 and y[0, 11] < y[0, 10] * 1.01 and y[0, 20] > 0


def test_markov_spikes_burst():
    s = fx.markov_spikes(np.random.default_rng(0), 50, 3000)
    assert s.shape == (50, 3000) and set(np.unique(s)) <= {0.0, 1.0}
    rate = s.mean()
    assert 0.002 < rate < 0.05
    # bursts: a spike is followed by another within 5 frames far more often than chance
    later = s[:, 1:-4] + s[:, 2:-3] + s[:, 3:-2] + s[:, 4:-1] + s[:, 5:]
    follow = (s[:, :-5].astype(bool) & (later > 0)).sum()
    assert follow / max(s.sum(), 1) > 5 * rate * 5


def test_calcium_matches_legacy_recursion_without_rise():
    sp = np.zeros((2, 40), np.float32)
    sp[0, 3] = 1.0
    sp[1, 5] = 0.5
    c = fx.calcium(sp)
    decay = np.float32(np.exp(-1.0 / 15.0))
    assert c[0, 3] == 1.0 and c[0, 4] == decay and c[1, 5] == 0.5
    r = fx.calcium(sp, rise_tau=2.0)
    assert r[0, 3] < 0.5 and r[0].max() == 1.0 and r[0].argmax() > 3


def test_bleach_curve_endpoints():
    b = fx.bleach_curve(1000, 100.0)
    assert b[0] == 1.0 and 0.5 < b[-1] < 0.51


def test_vessel_mask_dark_curves():
    m = fx.vessel_mask(np.random.default_rng(0), (80, 80), 2)
    assert m.shape == (80, 80) and m.dtype == np.float32
    assert m.max() <= 1.0 and 0.25 < m.min() < 0.75
    assert 0.02 < (m < 0.9).mean() < 0.4
    ones = np.ones((8, 8), np.float32)
    assert np.array_equal(fx.vessel_mask(np.random.default_rng(0), (8, 8), 0), ones)


def test_warp_field_unit_peak_and_walk_amplitude():
    f = fx.warp_field(np.random.default_rng(0), (64, 64), 20.0)
    assert f.shape == (2, 64, 64) and abs(np.hypot(f[0], f[1]).max() - 1.0) < 1e-5
    a = fx.walk_amplitude(np.random.default_rng(0), 5000, 2.0)
    assert abs(np.quantile(np.abs(a), 0.99) - 2.0) < 1e-4


def test_warp_replicates_border():
    frame = np.full((32, 32), 7.0, np.float32)
    field = np.ones((2, 32, 32), np.float32)
    out = fx.warp(frame, field, 50.0)  # far beyond the frame
    assert out.shape == (32, 32) and np.isfinite(out).all() and (out == 7.0).all()
    assert np.array_equal(fx.warp(frame, field, 0.0), frame)


def test_frame_map_semantics():
    fm = fx.frame_map(np.random.default_rng(0), 100, dropped=5, duplicated=5)
    assert fm.shape == (100,) and fm[0] == 0 and (fm == -1).sum() == 5
    dup = [t for t in range(1, 100) if fm[t] != t and fm[t] != -1]
    assert len(dup) == 5 and all(fm[t] == fm[t - 1] for t in dup)
    assert np.array_equal(fx.frame_map(np.random.default_rng(0), 10, 0, 0), np.arange(10))
