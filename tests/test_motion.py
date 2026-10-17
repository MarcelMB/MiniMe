import numpy as np

from cafast import motion, preprocess
from cafast.testing import synth


def _aligned_rmse(est, true):
    best = 1e9
    for sign in (-1, 1):
        d = est - sign * true
        best = min(best, float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())))
    return best


def test_tophat_square_matches_cv2():
    import cv2

    f = synth.synth_movie(64, 64, 3, 0)
    out = preprocess.tophat(f, wnd=15)
    k = np.ones((31, 31), np.uint8)
    assert out.dtype == np.uint8
    assert np.array_equal(out[1], cv2.morphologyEx(f[1], cv2.MORPH_TOPHAT, k))


def test_motion_estimate_accuracy():
    h, w, T = 256, 256, 300  # at 128x128 the faithful port gives ~0.29 px (small FOV)
    movie = synth.synth_movie(h, w, T, 0)
    setup = synth.synth_setup(h, w, T, 0)
    pre = preprocess.preprocess(movie, 7, 15)
    tmpl = motion.make_template(pre[:200], down=2)
    assert tmpl.shape == (h, w)
    sh = motion.estimate(pre, tmpl)
    assert sh.shape == (T, 2)
    assert _aligned_rmse(sh, setup.shifts.astype(float)) < 0.15
    q = motion.frame_quality(motion.apply_shifts(pre, sh), tmpl)
    assert q.shape == (T,) and q.min() > 0.5
    assert motion.template_image(tmpl).shape == (h // 2, w // 2)


def test_apply_shifts_float32():
    f = np.random.default_rng(0).random((2, 32, 32)).astype(np.float32)
    out = motion.apply_shifts(f, np.array([[0.0, 0.0], [1.0, 2.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out[0], f[0])
    # content moves by -(dy, dx): out[y, x] = f[y + dy, x + dx]
    assert np.allclose(out[1, :-1, :-2], f[1, 1:, 2:], atol=1e-6)


def test_sidecar_roundtrip_and_validation(tmp_path):
    sc = motion.MotionSidecar(
        shifts=np.zeros((10, 2), np.float32),
        template=np.ones((8, 8), np.float32),
        quality=np.ones(10, np.float32),
        params={"max_shift": 20, "down": 2},
        raw_hash="abc",
        meta={},
    )
    motion.save_sidecar(tmp_path / "motion", sc)
    back = motion.load_sidecar(tmp_path / "motion")
    assert np.array_equal(back.shifts, sc.shifts) and back.params == sc.params
    assert back.meta["n_frames"] == 10 and back.meta["template_shape"] == [8, 8]
    assert (
        motion.load_if_valid(tmp_path / "motion", "abc", {"down": 2, "max_shift": 20}, 10)
        is not None
    )
    assert motion.load_if_valid(tmp_path / "motion", "zzz", sc.params, 10) is None
    assert (
        motion.load_if_valid(tmp_path / "motion", "abc", {"max_shift": 10, "down": 2}, 10) is None
    )
    assert motion.load_if_valid(tmp_path / "motion", "abc", sc.params, 11) is None
    assert motion.load_if_valid(tmp_path / "nope", "abc", sc.params, 10) is None
