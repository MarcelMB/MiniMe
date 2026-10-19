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


def test_preprocess_steps_can_be_skipped():
    """``ksize <= 1`` / ``wnd <= 0`` skip a step: input that was preprocessed upstream."""
    f = synth.synth_movie(64, 64, 3, 0)
    assert np.array_equal(preprocess.denoise(f, 1), f)
    assert np.array_equal(preprocess.tophat(f, 0), f)
    assert np.array_equal(preprocess.preprocess(f, 1, 0), f)
    assert np.array_equal(preprocess.preprocess(f, 1, 15), preprocess.tophat(f, 15))


def _vessels_and_flicker(T: int = 400, side: int = 128, seed: int = 0):
    """Static vessel texture + vignetting + sparse flickering cells, rigidly shifted."""
    import cv2

    rng = np.random.default_rng(seed)
    pad = side + 40
    tex = cv2.GaussianBlur(rng.normal(0, 1, (pad, pad)).astype(np.float32), (0, 0), 2.5)
    tex = 25 * tex / tex.std()
    yy, xx = np.mgrid[:pad, :pad] - pad / 2
    vignette = 80 * np.exp(-(yy**2 + xx**2) / (2 * (pad / 3) ** 2))
    cells = rng.integers(10, pad - 10, (60, 2))
    blob = np.exp(-(np.arange(-7, 8)[:, None] ** 2 + np.arange(-7, 8)[None] ** 2) / 8.0)
    shifts = np.cumsum(rng.normal(0, 0.7, (T, 2)), axis=0).clip(-8, 8)
    movie = np.empty((T, side, side), np.uint8)
    for t in range(T):
        f = 40 + vignette + tex
        for cy, cx in cells[rng.random(len(cells)) < 0.1]:
            f[cy - 7 : cy + 8, cx - 7 : cx + 8] += 90 * blob
        m = np.array([[1, 0, 20 + shifts[t, 1]], [0, 1, 20 + shifts[t, 0]]], np.float32)
        f = cv2.warpAffine(f.astype(np.float32), m, (side, side), flags=cv2.WARP_INVERSE_MAP)
        movie[t] = np.clip(f + rng.normal(0, 4, f.shape), 0, 255)
    return movie, shifts


def test_bandpassed_registration_tracks_static_structure():
    movie, true = _vessels_and_flicker()
    den = preprocess.denoise(movie, 3)
    reg = preprocess.bandpass(den, 1.0, 6.0)
    assert reg.dtype == np.float32 and reg.shape == movie.shape
    tmpl = motion.make_template(reg[:200], down=2)
    assert _aligned_rmse(motion.estimate(reg, tmpl), true) < 0.5


def test_bandpass_close_to_exact_dog():
    import cv2

    den = preprocess.denoise(synth.synth_movie(128, 128, 5, 0), 7)
    exact = np.stack(
        [
            cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 6.0)
            for x in den.astype(np.float32)
        ]
    )
    fast = preprocess.bandpass(den, 1.0, 6.0)
    assert np.corrcoef(exact.ravel(), fast.ravel())[0, 1] > 0.98  # 0.989 here, 0.993 at 600 px


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


def test_template_image_shape_for_odd_sizes():
    from types import SimpleNamespace

    from cafast import pipeline
    from cafast.params import Params

    h, w = 603, 601  # cv2.resize gives 302 x 300 here, not h // 2 x w // 2 = 301 x 300
    pre = preprocess.preprocess(synth.synth_movie(h, w, 4, 0), 7, 15)
    tmpl = motion._mean_template(pre, 2)
    img = motion.template_image(tmpl)
    assert img.shape == tmpl.prep_shape == motion._prep(pre, 2).shape[1:] == (302, 300)
    assert np.allclose(img, motion._prep(pre, 2).mean(axis=0), atol=1e-3)
    # sidecar rerun: the rebuilt template gives the same quality as the original
    sc = SimpleNamespace(template=img)
    info = SimpleNamespace(height=h, width=w)
    back = pipeline._template_from_sidecar(sc, info, Params(motion_down=2))
    assert back.prep_shape == tmpl.prep_shape and back.shape == (h, w)
    np.testing.assert_allclose(
        motion.frame_quality(pre, back), motion.frame_quality(pre, tmpl), atol=1e-5
    )
