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


def test_too_few_values_raises():
    with pytest.raises(ValueError, match="2 values for a 3-component mixture"):
        fit_gmm_1d(np.array([1.0, 2.0]), 3)
