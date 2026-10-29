import hashlib
import json
import platform
import sys

import numpy as np
import pytest
from scipy import sparse

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


def _legacy_movie(h: int, w: int, T: int, seed: int) -> np.ndarray:
    """The generator as it was before ``SynthOptions`` (commit daf67a0), verbatim."""
    rng = np.random.default_rng(seed)
    m = 8
    ch, cw = h + 2 * m, w + 2 * m
    ax = np.arange(15) - 7
    patch = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * 3.0**2))
    patch = (patch / patch.max()).astype(np.float32)
    n_cells = max(1, int(ch * cw / 900))
    ys = rng.integers(0, ch - 15, n_cells)
    xs = rng.integers(0, cw - 15, n_cells)
    spikes = (rng.random((n_cells, T)) < 0.01).astype(np.float32)
    spikes = spikes * rng.uniform(0.5, 1.5, (n_cells, 1)).astype(np.float32)
    decay = np.float32(np.exp(-1.0 / 15.0))
    traces = np.empty_like(spikes)
    acc = np.zeros(n_cells, dtype=np.float32)
    for t in range(T):
        acc = acc * decay + spikes[:, t]
        traces[:, t] = acc
    C = 20.0 + traces * 60.0
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    vign = 40.0 + 30.0 * np.exp(
        -((yy - ch / 2) ** 2 + (xx - cw / 2) ** 2) / (2 * (0.4 * max(ch, cw)) ** 2)
    )
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * np.arange(T) / (30 * 60))
    shifts = np.clip(np.cumsum(rng.integers(-1, 2, (T, 2)), axis=0), -m, m)
    out = np.empty((T, h, w), dtype=np.uint8)
    for t0 in range(0, T, 50):
        t1 = min(T, t0 + 50)
        n = t1 - t0
        canvas = vign[None] * drift[t0:t1, None, None].astype(np.float32)
        canvas = canvas + rng.standard_normal((n, ch, cw), dtype=np.float32) * 4.0
        for k in range(n_cells):
            canvas[:, ys[k] : ys[k] + 15, xs[k] : xs[k] + 15] += (
                C[k, t0:t1, None, None] * patch[None]
            )
        for i in range(n):
            dy, dx = shifts[t0 + i]
            out[t0 + i] = np.clip(canvas[i, m + dy : m + dy + h, m + dx : m + dx + w], 0, 255)
    return out


def _digest(movie: np.ndarray) -> str:
    return hashlib.sha256(movie.tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("h, w, T, seed", [(64, 64, 300, 0), (32, 32, 20, 1), (48, 40, 120, 3)])
def test_default_session_is_byte_identical(h, w, T, seed):
    """The options-free path reproduces the legacy generator exactly, on any platform."""
    assert np.array_equal(synth.synth_movie(h, w, T, seed), _legacy_movie(h, w, T, seed))


@pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason="digests pinned on Apple silicon; x86 float32 exp/sin differ by an ULP",
)
def test_default_session_digests_on_apple_silicon():
    # pinned on 2026-10-27 from the pre-options generator
    assert _digest(synth.synth_movie(64, 64, 300, 0)) == "ce39843f43462047"
    assert _digest(synth.synth_movie(32, 32, 20, 1)) == "de776615f5097caf"


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


def test_vessel_mask_wide_dark_trunks_and_faint_branches():
    m = fx.vessel_mask(np.random.default_rng(0), (300, 300), 2)
    assert m.shape == (300, 300) and m.dtype == np.float32
    assert m.max() <= 1.0 and 0.3 < m.min() < 0.8
    assert 0.02 < (m < 0.9).mean() < 0.3
    # the darkest point sits on a trunk several px wide, not on a 2 px line
    y, x = np.unravel_index(int(m.argmin()), m.shape)
    assert (m[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] < 0.95).mean() > 0.5
    faint = fx.vessel_mask(np.random.default_rng(0), (300, 300), 2, density=0.02)
    assert faint.min() > m.min()
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


# --- options, extended truth, shared renderer ------------------------------------------

ALL_ON = synth.SynthOptions(
    background_drift=15.0,
    neuropil=10.0,
    overlap=2,
    late_active=1,
    bleach_tau=100.0,
    vessels=2,
    motion_step=2,
    motion_margin=12,
    nonrigid=1.5,
    dropped=3,
    duplicated=3,
    size_cv=0.3,
    amp_cv=0.3,
    noise_sd=6.0,
    bursts=True,
    rise_tau=2.0,
    hemo=20.0,
    edge_band=3,
    out_of_focus=2,
)


def test_options_round_trip_and_scenarios():
    d = ALL_ON.to_dict()
    assert synth.SynthOptions.from_dict(d) == ALL_ON
    assert synth.SynthOptions.from_dict(None) == synth.SynthOptions()
    assert synth.SCENARIOS["clean"] == synth.SynthOptions()
    assert set(synth.SCENARIOS) >= {"realistic", "overlap", "late_active", "vessels", "hard"}
    assert synth.SCENARIOS["hard"].out_of_focus > 0
    assert synth.SCENARIOS["grin_v4"].aperture > 0


def test_truth_matches_generator_with_every_effect():
    h, w, T = 48, 48, 90
    setup = synth.synth_setup(h, w, T, seed=3, options=ALL_ON)
    movie = synth.synth_movie(h, w, T, seed=3, options=ALL_ON)
    clean = np.concatenate([c for _, c, _ in synth.truth_chunks(setup, h, w, chunk=7)])
    diff = movie.astype(float) - clean.astype(float)
    assert abs(diff.mean()) < 0.5 and diff.std() < 7.0
    assert setup.C.shape[0] == setup.ys.size == setup.spikes.shape[0]
    assert setup.F0.shape == setup.C.shape and setup.patches.shape[0] == setup.ys.size


@pytest.mark.parametrize(
    "opts",
    [
        dict(background_drift=15.0),
        dict(neuropil=10.0),
        dict(overlap=2),
        dict(late_active=1),
        dict(bleach_tau=100.0),
        dict(vessels=2),
        dict(nonrigid=1.5),
        dict(dropped=3, duplicated=3),
        dict(amp_cv=0.3),
        dict(bursts=True, rise_tau=2.0),
        dict(hemo=20.0),
        dict(edge_band=3),
        dict(out_of_focus=3),
        dict(spike_rate=0.003, rate_cv=2.0),
        dict(amp_scale=0.5, cell_f0=10.0, decay_tau=24.0),
        dict(aperture=1.0, shot_noise=0.4, background=1.5),
    ],
)
def test_single_effect_keeps_cells_and_motion(opts):
    base = synth.synth_setup(64, 64, 60, seed=0)
    on = synth.synth_setup(64, 64, 60, seed=0, options=synth.SynthOptions(**opts))
    n = base.ys.size
    assert np.array_equal(synth.true_centroids(on)[:n], synth.true_centroids(base))
    assert np.array_equal(on.shifts, base.shifts)


def test_size_cv_keeps_positions_but_grows_patch():
    base = synth.synth_setup(64, 64, 30, seed=0)
    on = synth.synth_setup(64, 64, 30, seed=0, options=synth.SynthOptions(size_cv=0.3))
    assert on.sigma.size == base.ys.size and on.sigma.std() > 0
    assert on.patch.shape[0] >= 15 and on.patch.shape[0] % 2 == 1


def test_overlap_more_than_cells():
    s = synth.synth_setup(16, 16, 20, seed=0, options=synth.SynthOptions(overlap=5))
    assert s.ys.size == max(1, int(32 * 32 / 900)) + 5
    assert (s.group[-5:] >= 0).all()


def test_overlap_pairs_are_close_and_correlated():
    s = synth.synth_setup(64, 64, 600, seed=0, options=synth.SynthOptions(overlap=2))
    n = s.ys.size - 2
    for k in range(2):
        j = int(np.flatnonzero(s.group[:n] == k)[0])
        d = np.hypot(s.ys[n + k] - s.ys[j], s.xs[n + k] - s.xs[j])
        assert 4.0 <= d <= 10.0
        assert np.corrcoef(s.C[j], s.C[n + k])[0, 1] > 0.5


def test_late_active_clamped_and_first_active():
    s = synth.synth_setup(64, 64, 300, seed=0, options=synth.SynthOptions(late_active=99))
    c = synth.true_centroids(s)
    inside = (c >= 5).all(axis=1) & (c < 59).all(axis=1)
    assert inside.any() and (s.spikes[inside, :210] == 0).all()
    assert (s.first_active[inside] >= 210).all() and (s.first_active[~inside] < 210).any()
    base = synth.synth_setup(64, 64, 300, seed=0)
    has = base.spikes.any(axis=1)
    assert (base.first_active[has] < 300).all() and (base.first_active[~has] == 300).all()


def test_bleach_scales_baseline_and_cells():
    s = synth.synth_setup(64, 64, 200, seed=0, options=synth.SynthOptions(bleach_tau=50.0))
    assert np.allclose(s.F0[:, 0], 20.0) and np.allclose(s.F0[:, -1], 20.0 * s.bleach[-1])
    assert s.bleach[-1] < 0.52


def test_frame_map_matches_movie():
    h, w, T = 32, 32, 120
    o = synth.SynthOptions(dropped=5, duplicated=5)
    movie = np.concatenate([c for _, c in synth.synth_chunks(h, w, T, seed=1, chunk=7, options=o)])
    s = synth.synth_setup(h, w, T, seed=1, options=o)
    for t in range(T):
        src = s.frame_map[t]
        if src == -1:
            assert not movie[t].any()
        elif src != t:
            assert np.array_equal(movie[t], movie[t - 1])
    assert (s.frame_map == -1).sum() == 5
    cells = np.concatenate([c for _, _, c in synth.truth_chunks(s, h, w, chunk=100)])
    assert not cells[s.frame_map == -1].any()


def test_large_motion_clipped():
    o = synth.SynthOptions(motion_step=5, motion_margin=6)
    s = synth.synth_setup(32, 32, 500, seed=0, options=o)
    assert np.abs(s.shifts).max() <= 6 and s.margin == 6
    assert synth.synth_movie(32, 32, 20, seed=0, options=o).shape == (20, 32, 32)


def test_vessels_darken_movie():
    base = synth.synth_movie(64, 64, 20, seed=0)
    on = synth.synth_movie(64, 64, 20, seed=0, options=synth.SynthOptions(vessels=2))
    assert on.mean() < base.mean() - 0.5


def test_edge_band_darkens_border():
    m = synth.synth_movie(64, 64, 5, seed=0, options=synth.SynthOptions(edge_band=4))
    assert m[:, :4].mean() < 0.4 * m[:, 10:-10, 10:-10].mean()


def test_write_session_stores_options(tmp_path):
    o = synth.SynthOptions(vessels=1, dropped=2)
    meta = synth.write_session(tmp_path / "raw", 0.05, segment=30, seed=0, side=32, options=o)
    assert meta["options"] == o.to_dict()
    saved = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert synth.SynthOptions.from_dict(saved["options"]) == o
    plain = synth.write_session(tmp_path / "raw2", 0.05, segment=30, side=32)
    assert synth.SynthOptions.from_dict(plain["options"]) == synth.SynthOptions()


def test_detectable_cells_inside_and_active():
    s = synth.synth_setup(128, 128, 300, seed=0)
    det = synth.detectable(s, 128, 128)
    c = synth.true_centroids(s)
    nspk = (s.spikes > 0).sum(axis=1)
    inside = (c >= 5).all(axis=1) & (c < 123).all(axis=1)
    assert det.dtype == bool and det.shape == (s.ys.size,)
    assert np.array_equal(det, inside & (nspk >= 2))
    assert 0 < det.sum() < s.ys.size


def test_late_active_cells_are_detectable_and_fire_late():
    o = synth.SynthOptions(late_active=2)
    s = synth.synth_setup(128, 128, 300, seed=0, options=o)
    late = np.flatnonzero((s.first_active >= 210) & (s.first_active < 300))
    assert late.size >= 2
    assert synth.detectable(s, 128, 128)[late].all()
    assert ((s.spikes[late, 210:] > 0).sum(axis=1) >= 2).all()


def test_generator_peak_memory_stays_near_three_chunk_arrays():
    import tracemalloc

    h = w = 300
    n = 100
    chunk_bytes = n * (h + 16) * (w + 16) * 4  # one float32 canvas chunk
    gen = synth.synth_chunks(h, w, n, seed=0, chunk=n)
    tracemalloc.start()
    next(gen)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 3.3 * chunk_bytes, peak / chunk_bytes


def test_out_of_focus_cells_are_wide_dim_and_not_in_the_cell_truth():
    h = w = 64
    o = synth.SynthOptions(out_of_focus=3)
    base, on = synth.synth_setup(h, w, 200, seed=0), synth.synth_setup(h, w, 200, seed=0, options=o)
    assert on.oof_ys.size == 3 and on.oof_signal.shape == (3, 200)
    assert (on.oof_sigma > 2 * synth.CELL_SIGMA).all()
    assert on.oof_signal.max() < synth.CELL_AMP * synth.OOF_AMP[1] * 1.5
    _, clean_b, cells_b = next(synth.truth_chunks(base, h, w, chunk=200))
    _, clean_o, cells_o = next(synth.truth_chunks(on, h, w, chunk=200))
    assert np.array_equal(cells_b, cells_o)  # CNMF's target is unchanged
    assert clean_o.astype(int).sum() > clean_b.astype(int).sum()  # but the movie is brighter


def test_truth_npz_holds_the_scoring_arrays_and_footprints(tmp_path):
    o = synth.SynthOptions(overlap=2, late_active=1, vessels=1, out_of_focus=2)
    synth.write_session(tmp_path / "raw", 0.05, segment=30, seed=0, side=48, options=o)
    s = synth.synth_setup(48, 48, 90, 0, o)
    with np.load(tmp_path / "truth.npz") as z:
        assert np.array_equal(z["centroids"], synth.true_centroids(s))
        assert np.array_equal(z["cat_late"], s.late) and z["cat_late"].sum() == 1
        assert z["cat_overlap"].sum() >= 3 and z["oof_centroids"].shape == (2, 2)
        A = sparse.csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]), shape=z["A_shape"])
    assert A.shape == (s.ys.size, 48 * 48) and np.isclose(A.max(), 1.0)
    inside = synth.detectable(s, 48, 48)
    peak = np.asarray(A[inside].argmax(axis=1)).ravel()
    c = np.rint(synth.true_centroids(s)[inside]).astype(int)
    assert np.array_equal(peak, c[:, 0] * 48 + c[:, 1])


def test_grin_aperture_matches_the_v4_radial_profile():
    t, rim = fx.grin_aperture((600, 600), 1.0)
    yy, xx = np.mgrid[0:600, 0:600]
    r = np.hypot(yy - 299.5, xx - 299.5) / 300
    centre = t[r < 0.2].mean()
    rel = [t[(r >= a) & (r < b)].mean() / centre for a, b in [(0.4, 0.6), (0.9, 1.0), (1.2, 1.42)]]
    assert np.allclose(rel, [0.79, 0.28, 0.03], atol=0.06)  # real V4: 0.79, 0.28, 0.03
    assert rim[300, 300] < 0.01 and rim[0, 0] == 1.0


def test_spike_rate_and_spread_make_most_cells_quiet():
    base = synth.synth_setup(128, 128, 2000, seed=0)
    o = synth.SynthOptions(spike_rate=0.002, rate_cv=3.0)
    on = synth.synth_setup(128, 128, 2000, seed=0, options=o)
    per_cell = (on.spikes > 0).sum(axis=1)
    assert per_cell.sum() < (base.spikes > 0).sum() / 2
    assert np.median(per_cell) < per_cell.mean() / 2  # a few busy cells, most quiet


def test_shot_noise_grows_with_brightness():
    o = synth.SynthOptions(aperture=1.0, shot_noise=0.5, noise_sd=1.0)
    movie = synth.synth_movie(96, 96, 60, seed=0, options=o).astype(float)
    sd = np.diff(movie, axis=0).std(axis=0) / np.sqrt(2)
    mean = movie.mean(axis=0)
    assert sd[mean > np.quantile(mean, 0.9)].mean() > 1.5 * sd[mean < np.quantile(mean, 0.1)].mean()
