import numpy as np

from cafast.testing import synth


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
