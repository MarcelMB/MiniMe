"""1-D Gaussian mixture fitted by EM, in place of sklearn's ``GaussianMixture(random_state=0)``.

The seed filters fit a mixture to one value per seed (log PNR or ptp). This repeats sklearn's
default fit step for step: k-means++ centres drawn from ``RandomState(0)``, Lloyd's k-means
with sklearn's centre-shift tolerance, then EM (``reg_covar`` added to each variance,
``10 * eps`` added to each component count) stopped once the mean log-likelihood changes by
less than ``tol``. The early stop matters: real log-PNR is not cleanly bimodal, and EM run to
convergence moves the threshold by hundreds of seeds, so the start and the stopping rule are
part of what the ``auto`` threshold means (checked against sklearn in ``tests/test_gmm.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

TOL = 1e-3  # sklearn's change in mean log-likelihood that ends EM
REG_COVAR = 1e-6  # sklearn's floor added to each variance
MAX_ITER = 100  # sklearn's EM iteration cap
KMEANS_TOL = 1e-4  # sklearn's KMeans tolerance, times the data variance
KMEANS_MAX_ITER = 300  # sklearn's KMeans iteration cap
SEED = 0  # the random_state the seed filters always passed to sklearn


@dataclass(frozen=True)
class Gmm1D:
    """Fitted mixture: component ``weights``, ``means`` and ``variances``, each (k,)."""

    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most likely component of each value in ``x``."""
        return _weighted_log_prob(np.ravel(x), self).argmax(1)


def _weighted_log_prob(x: np.ndarray, g: Gmm1D) -> np.ndarray:
    """(n, k) ``log w_k + log N(x | mu_k, var_k)``."""
    d2 = (x[:, None] - g.means) ** 2 / g.variances
    return -0.5 * (np.log(2 * np.pi) + np.log(g.variances) + d2) + np.log(g.weights)


def _kmeans_plusplus(x: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """sklearn's greedy k-means++ centres (``2 + log k`` candidates per centre)."""
    n = len(x)
    centres = np.empty(k)
    centres[0] = x[rng.choice(n, p=np.full(n, 1 / n))]
    closest = (x - centres[0]) ** 2
    pot = closest.sum()
    for c in range(1, k):
        ids = np.searchsorted(np.cumsum(closest), rng.uniform(size=2 + int(np.log(k))) * pot)
        ids = np.minimum(ids, n - 1)
        dist = np.minimum(closest, (x[None, :] - x[ids, None]) ** 2)
        best = dist.sum(1).argmin()
        pot, closest, centres[c] = dist[best].sum(), dist[best], x[ids[best]]
    return centres


def _kmeans_1d(x: np.ndarray, k: int) -> np.ndarray:
    """Labels of sklearn's ``KMeans(k, n_init=1, random_state=0)`` on 1-D ``x``."""
    x = x - x.mean()  # sklearn centres the data before seeding
    centres = _kmeans_plusplus(x, k, np.random.RandomState(SEED))
    tol = x.var() * KMEANS_TOL
    old = np.full(len(x), -1)
    for _ in range(KMEANS_MAX_ITER):
        labels = np.abs(x[:, None] - centres).argmin(1)
        new = np.array(
            [x[labels == j].mean() if (labels == j).any() else centres[j] for j in range(k)]
        )
        shift = ((new - centres) ** 2).sum()
        centres = new
        if np.array_equal(labels, old):
            return labels
        if shift <= tol:
            break
        old = labels
    return np.abs(x[:, None] - centres).argmin(1)  # labels that match the final centres


def _m_step(x: np.ndarray, resp: np.ndarray) -> Gmm1D:
    """Weights, means and variances from (n, k) responsibilities, as sklearn's full covariance."""
    nk = resp.sum(0) + 10 * np.finfo(resp.dtype).eps
    means = resp.T @ x / nk
    variances = (resp * (x[:, None] - means) ** 2).sum(0) / nk + REG_COVAR
    return Gmm1D(nk / nk.sum(), means, variances)


def fit_gmm_1d(x: np.ndarray, n_components: int) -> Gmm1D:
    """Fit a ``n_components`` Gaussian mixture to the values ``x``, as sklearn's defaults."""
    x = np.ravel(x).astype(np.float64)
    if len(x) < n_components:
        raise ValueError(f"{len(x)} values for a {n_components}-component mixture")
    g = _m_step(x, np.eye(n_components)[_kmeans_1d(x, n_components)])
    lower = -np.inf
    for _ in range(MAX_ITER):
        wlp = _weighted_log_prob(x, g)
        norm = logsumexp(wlp, axis=1)
        g = _m_step(x, np.exp(wlp - norm[:, None]))
        prev, lower = lower, norm.mean()
        if abs(lower - prev) < TOL:
            break
    return g
