import numpy as np
import pytest

from minime import motion, preprocess
from minime.params import Params
from minime.testing import synth


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
    assert back.fill is None  # zero fill: no fill image stored
    sc.fill = np.full((8, 8), 7, np.uint8)
    motion.save_sidecar(tmp_path / "motion", sc)
    back = motion.load_sidecar(tmp_path / "motion")
    assert back.fill.dtype == np.uint8 and np.array_equal(back.fill, sc.fill)


def test_template_image_shape_for_odd_sizes():
    from types import SimpleNamespace

    from minime import pipeline
    from minime.params import Params

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


@pytest.mark.parametrize("down", [0, 3, 4])
def test_params_refuse_motion_down_other_than_1_or_2(down):
    with pytest.raises(ValueError, match="motion_down"):
        Params(motion_down=down)


def _shifted_movie(shifts: np.ndarray, side: int = 96, seed: int = 0) -> np.ndarray:
    """Blob texture moved rigidly by integer (dy, dx) per frame, float32 (``estimate`` of a
    frame at ``d`` vs a template at ``0`` is ``-d``)."""
    import cv2

    rng = np.random.default_rng(seed)
    pad = side + 40
    tex = cv2.GaussianBlur(rng.normal(0, 1, (pad, pad)).astype(np.float32), (0, 0), 2.0)
    out = np.empty((len(shifts), side, side), np.float32)
    for t, (dy, dx) in enumerate(shifts.astype(int)):
        out[t] = tex[20 + dy : 20 + dy + side, 20 + dx : 20 + dx + side]
        out[t] += rng.normal(0, 0.05, (side, side)).astype(np.float32)
    return out


def test_make_template_defaults_unchanged():
    """iters=1, chunks=1, centre="mean" is the old align-once-then-average template."""
    rng = np.random.default_rng(1)
    f = _shifted_movie(rng.integers(-3, 4, (40, 2)))
    t0 = motion._mean_template(f, 2)
    old = motion._mean_template(motion.apply_shifts(f, motion.estimate(f, t0, 20)), 2)
    assert np.array_equal(motion.make_template(f, 2, 20).fft, old.fft)


def test_spread_template_sits_at_median_position():
    """Chunks at two positions (6 of 10 at (3, 2)): the median-centred template takes the
    majority position, not the blur between them."""
    pos = np.repeat([[3, 2]] * 6 + [[-5, -4]] * 4, 10, axis=0)
    f = _shifted_movie(pos)
    t = motion.make_template(f, 2, 20, iters=2, chunks=10, centre="median")
    s = motion.estimate(f, t, 20)
    assert np.allclose(s[:60], 0, atol=0.3) and np.allclose(s[60:], [8, 6], atol=0.3)


def test_best_chunk_skips_corrupt_and_minority_runs():
    """The spread template starts from the run that agrees best with the others: not a run
    of corrupted frames, not a run at a minority position."""
    pos = np.repeat([[0, 0]] + [[2, -3]] * 3 + [[-6, 5]], 8, axis=0)
    f = _shifted_movie(pos)
    f[:8] = np.random.default_rng(5).normal(0, 1, f[:8].shape)  # run 0: noise frames
    best = motion._best_chunk(f, 5, 2)
    assert any(np.array_equal(best, f[k * 8 : k * 8 + 8]) for k in (1, 2, 3))


def test_spread_template_ignores_corrupt_run():
    """A run of noise frames does not move the spread template off the majority position."""
    pos = np.repeat([[3, 2]] * 7 + [[-4, -4]] * 3, 10, axis=0)
    f = _shifted_movie(pos)
    f[:10] = np.random.default_rng(6).normal(0, 1, f[:10].shape)
    t = motion.make_template(f, 2, 20, iters=2, chunks=10, centre="median")
    s = motion.estimate(f, t, 20)
    assert np.allclose(s[10:70], 0, atol=0.3) and np.allclose(s[70:], [7, 6], atol=0.3)


def test_fix_outliers_only_touches_weak_far_frames():
    s = np.zeros((20, 2))
    s[5] = [15, -12]  # weak peak: corrupted frame -> replaced
    s[12] = [6, 0]  # strong peak: fast real motion -> kept
    peak = np.full(20, 0.5)
    peak[5] = 0.1
    out = motion.fix_outliers(s, peak, 3.0)
    assert np.allclose(out[5], 0) and np.allclose(out[12], [6, 0])
    assert np.array_equal(motion.fix_outliers(s, peak, 0.0), s)


def test_estimate_peak_matches_estimate_and_flags_black_frames():
    rng = np.random.default_rng(2)
    f = _shifted_movie(rng.integers(-3, 4, (30, 2)))
    t = motion.make_template(f, 2, 20)
    f[7] = 0.0
    sh, pk = motion.estimate_peak(f, t, 20)
    assert np.array_equal(sh, motion.estimate(f, t, 20))
    assert pk[7] == 0.0 and np.median(pk) > 0.5


def test_apply_shifts_replicate_has_no_zero_band():
    f = np.full((2, 20, 30), 50, np.uint8)
    sh = np.array([[3.0, -4.0], [-2.5, 1.5]])
    zero = motion.apply_shifts(f, sh)
    rep = motion.apply_shifts(f, sh, replicate=True)
    assert (zero == 0).any() and (rep == 50).all()
    with pytest.raises(ValueError, match="motion_fill"):
        Params(motion_fill="replicate")


def test_fill_outside_sets_exactly_the_pixels_read_from_outside():
    """``fill_outside`` overwrites the pixels whose bilinear source leaves the frame, which a
    zero-filled warp of an all-255 frame shows as < 255, and nothing else."""
    rng = np.random.default_rng(1)
    h, w, n = 37, 53, 200
    sh = rng.uniform(-9, 9, (n, 2))
    # OpenCV < 5 (the Intel-Mac pin) rounds the source position to 1/32 px, so a shift within
    # 1/64 px of a whole one reads the edge pixel as inside; keep the fractions clear of that
    frac = sh - np.round(sh)
    sh += np.where(np.abs(frac) < 1 / 16, np.sign(frac) * 1 / 16, 0)
    sh[:10] = np.round(sh[:10])  # whole-pixel shifts: the far tap has weight 0
    outside = motion.apply_shifts(np.full((n, h, w), 255, np.uint8), sh) < 255
    f = rng.integers(40, 60, (n, h, w)).astype(np.uint8)
    warped = motion.apply_shifts(f, sh, replicate=True)
    out = motion.fill_outside(warped.copy(), sh, np.full((h, w), 7, np.uint8))
    assert np.array_equal(out == 7, outside)
    assert np.array_equal(out[~outside], warped[~outside])


def test_fill_outside_example_and_fill_image():
    rng = np.random.default_rng(4)
    f = rng.integers(40, 60, (3, 20, 30)).astype(np.uint8)
    sh = np.array([[0.0, 0.0], [3.0, -4.0], [-2.0, 2.0]])
    fill = np.full((20, 30), 77, np.uint8)
    w = motion.apply_shifts(f, sh, replicate=True)
    out = motion.fill_outside(w.copy(), sh, fill)
    # frame 1 reads rows y + 3 (rows 17-19 from outside) and columns x - 4 (columns 0-3)
    assert (out[1, 17:] == 77).all() and (out[1, :, :4] == 77).all()
    assert (out[1, :17, 4:] == w[1, :17, 4:]).all() and np.array_equal(out[0], f[0])
    # floor of the exact mean: (10 + 30 + 31) // 3 = 23, (255 * 3) // 3 = 255
    img = motion.fill_image(np.array([[[10, 255]], [[30, 255]], [[31, 255]]], np.uint8), 0 * sh)
    assert img.dtype == np.uint8 and img.tolist() == [[23, 255]]
    # only the frames holding data count: frame 1 (dy = 3) reads rows 17-19 from outside,
    # frame 2 (dy = -2) rows 0-1, so row 19 averages frames 0 and 2, row 0 frames 0 and 1
    f = np.stack([np.full((20, 30), v, np.uint8) for v in (10, 200, 31)])
    img = motion.fill_image(f, sh)
    assert img[10, 10] == (10 + 200 + 31) // 3
    assert img[19, 10] == (10 + 31) // 2 and img[0, 10] == (10 + 200) // 2
    assert img[19, 29] == 10  # frame 1 reads row 19, frame 2 column 29 from outside
    assert motion.fill_image(f[1:2], sh[1:2])[19, 10] == 200  # no frame holds data


def test_at_max_shift_flags_the_search_window_edge():
    sh = np.array([[0.0, 0.0], [18.9, 0.0], [0.0, -19.0], [21.0, 3.0], [-5.0, 19.4]])
    assert motion.at_max_shift(sh, 20, 2).tolist() == [False, False, True, True, True]
    assert motion.at_max_shift(sh, 20, 1).tolist() == [False, False, False, True, False]
