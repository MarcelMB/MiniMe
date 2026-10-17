import numpy as np
from scipy.optimize import nnls

from cafast.spatial_kernel import solve_nnqp


def test_nnqp_matches_nnls_per_pixel():
    rng = np.random.default_rng(0)
    K, T, P = 6, 300, 50
    C = np.abs(rng.standard_normal((K, T)))
    Cc = C - C.mean(1, keepdims=True)
    G = Cc @ Cc.T
    Wtrue = np.abs(rng.standard_normal((P, K))) * (rng.random((P, K)) < 0.5)
    Y = Wtrue @ C + 0.1 * rng.standard_normal((P, T))
    B = Y @ Cc.T
    lam = np.full(P, 0.5)
    sub = (Wtrue > 0) | (rng.random((P, K)) < 0.2)
    sub[0] = True
    indptr = np.r_[0, np.cumsum(sub.sum(1))]
    indices = np.concatenate([np.flatnonzero(s) for s in sub])
    vals = solve_nnqp(G, B, lam, indptr.astype(np.int64), indices.astype(np.int64))
    for i in range(P):
        idx = indices[indptr[i] : indptr[i + 1]]
        # NNLS of the equivalent problem: ||L w - (L^-T)(b - lam)||, L = chol(G[idx][:, idx])
        Gs = G[np.ix_(idx, idx)]
        L = np.linalg.cholesky(Gs)
        rhs = np.linalg.solve(L, B[i, idx] - lam[i])
        w_ref, _ = nnls(L.T, rhs)
        w = vals[indptr[i] : indptr[i + 1]]

        def f(x, Gs=Gs, b=B[i, idx], lam_i=lam[i]):
            return 0.5 * x @ Gs @ x - b @ x + lam_i * x.sum()

        assert f(w) <= f(w_ref) + 1e-6 * max(1.0, abs(f(w_ref)))
        assert (w >= 0).all()


def test_nnqp_empty_pixel_and_inactive_unit():
    # pixel 0 has no allowed unit; pixel 1 allows a constant-trace unit (G_kk = 0)
    G = np.array([[0.0, 0.0], [0.0, 4.0]])
    B = np.array([[0.0, 0.0], [3.0, 8.0]])
    lam = np.array([1.0, 1.0])
    indptr = np.array([0, 0, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    vals = solve_nnqp(G, B, lam, indptr, indices)
    assert vals.shape == (2,)
    assert vals[0] == 0.0 and np.isclose(vals[1], (8.0 - 1.0) / 4.0)
