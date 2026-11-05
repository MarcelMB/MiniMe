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

    def upper_cut(self) -> float:
        """One-sided threshold of a two-component fit: values from it on belong to the top.

        Two clusters (each component wins at its own mean): the crossing of the weighted
        densities between the means. One hump (one component wins at both means, the two
        differ in width, as log PNR without edge artefacts): the crossing above the means
        where the wider component takes over, so the upper tail; ``-inf`` when it wins
        everywhere. ``predict`` alone is two-sided: a wide top component also claims the
        lowest values, a narrow one drops the highest, and with near-equal means which one
        is "top" flips with rounding.
        """
        if len(self.means) != 2:
            raise ValueError(f"upper_cut needs 2 components, got {len(self.means)}")
        lo, hi = np.argsort(self.means)
        if self._wins(hi, lo, self.means[hi]) and self._wins(lo, hi, self.means[lo]):
            r = self._crossings(hi, lo)
            return float(r[(r > self.means[lo]) & (r < self.means[hi])][0])
        wide, narrow = np.argsort(self.variances)[::-1]
        r = self._crossings(wide, narrow)
        return float(r.max()) if r.size else -np.inf

    def _log_density(self, j: int) -> np.ndarray:
        """``(a, b, c)``: ``log w_j + log N(x | mu_j, var_j) = a x^2 + b x + c``."""
        m, v = self.means[j], self.variances[j]
        return np.array(
            [
                -0.5 / v,
                m / v,
                np.log(self.weights[j]) - 0.5 * np.log(2 * np.pi * v) - 0.5 * m * m / v,
            ]
        )

    def _wins(self, j: int, other: int, x: float) -> bool:
        a, b, c = self._log_density(j) - self._log_density(other)
        return bool(a * x * x + b * x + c > 0)

    def _crossings(self, j: int, other: int) -> np.ndarray:
        """Sorted real ``x`` where components ``j`` and ``other`` are equally likely."""
        a, b, c = self._log_density(j) - self._log_density(other)
        if a == 0:
            return np.array([-c / b]) if b != 0 else np.empty(0)
        r = np.roots([a, b, c])
        return np.sort(r[np.isreal(r)].real)


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
