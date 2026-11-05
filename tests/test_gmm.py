import numpy as np
import pytest

from minime.gmm import fit_gmm_1d

DISTRIBUTIONS = {
    "bimodal": lambda rng, n: np.r_[rng.normal(0, 1, n), rng.normal(4, 0.7, n // 4)],
    "lognormal": lambda rng, n: rng.lognormal(0, 1, n),
    "uniform": lambda rng, n: rng.uniform(0, 1, n),
    "heavy": lambda rng, n: rng.standard_t(2, n),
    "overlap": lambda rng, n: np.r_[rng.normal(0, 1, n), rng.normal(1.5, 1, n)],
}


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@pytest.mark.parametrize("k", [2, 3])
def test_matches_sklearn_gaussian_mixture(name, k):
    """Same means and labels as ``GaussianMixture(k, random_state=0)``, the fit Minian used."""
    sklearn_mixture = pytest.importorskip("sklearn.mixture")
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = DISTRIBUTIONS[name](rng, int(rng.integers(30, 3000)))
        sk = sklearn_mixture.GaussianMixture(n_components=k, random_state=0).fit(x[:, None])
        g = fit_gmm_1d(x, k)
        np.testing.assert_allclose(g.means, sk.means_.ravel(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(g.variances, sk.covariances_.ravel(), rtol=1e-9)
        assert (g.predict(x) == sk.predict(x[:, None])).all()


def test_separates_two_clusters():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 0.5, 400), rng.normal(5, 0.5, 100)]
    g = fit_gmm_1d(x, 2)
    hi = g.predict(x) == g.means.argmax()
    assert hi[:400].sum() == 0 and hi[400:].all()
    np.testing.assert_allclose(np.sort(g.weights), [0.2, 0.8], atol=1e-6)


def test_upper_cut_between_two_clusters_keeps_the_whole_top():
    """Two clusters: the cut lies between the means and keeps the top cluster plus anything
    above it, even values far above a narrow top component that ``predict`` hands back."""
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 1.5, 400), rng.normal(5, 0.3, 100)]
    g = fit_gmm_1d(x, 2)
    cut = g.upper_cut()
    assert 0 < cut < 5
    assert (x[400:] >= cut).all() and (x[:400] >= cut).sum() <= 2
    far = np.array([12.0])
    assert g.predict(far)[0] != g.means.argmax() and far[0] >= cut


def test_upper_cut_of_one_hump_is_its_upper_tail():
    """One hump fitted as a narrow and a wide component with (near) equal means: the cut is
    where the wide one takes over above the means, the same whichever mean is higher."""
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 0.8, 900), rng.normal(0, 2.5, 400)]
    g = fit_gmm_1d(x, 2)
    assert abs(g.means[0] - g.means[1]) < 0.5
    cut = g.upper_cut()
    assert cut > max(g.means) + 0.5
    m = g.means.mean()
    a, b = (
        type(g)(g.weights, m + np.array(d), g.variances).upper_cut()
        for d in ([1e-3, -1e-3], [-1e-3, 1e-3])
    )
    assert abs(a - b) < 0.01  # which mean is (barely) higher does not matter
    low = x[x < min(g.means) - 3]
    assert len(low) and (low < cut).all()  # never the low tail


def test_upper_cut_keeps_all_when_the_wide_component_wins_everywhere():
    from minime.gmm import Gmm1D

    g = Gmm1D(np.array([0.99, 0.01]), np.array([0.0, 0.1]), np.array([4.0, 0.25]))
    assert g.upper_cut() == -np.inf


def test_too_few_values_raises():
    with pytest.raises(ValueError, match="2 values for a 3-component mixture"):
        fit_gmm_1d(np.array([1.0, 2.0]), 3)
