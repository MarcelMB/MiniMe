import numpy as np
import pytest

from minime import compress, gpu

pytestmark = pytest.mark.skipif(not gpu.available(), reason="needs MLX and a Metal GPU")


def _tile(T: int = 3000, side: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = np.cumsum(rng.normal(0, 1, (T, 3)), axis=0)
    fp = rng.random((3, side * side))
    movie = 60 + sig @ fp + rng.normal(0, 3, (T, side * side))
    return np.clip(movie, 0, 255).astype(np.uint8).reshape(T, side, side)


def test_mlx_products_match_cpu():
    raw = _tile()
    cpu = compress.TileMatrix(raw, chunk=700)  # ragged last chunk on both sides
    g = gpu.MlxTileMatrix(raw, chunk=1100)
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((cpu.P, 12)).astype(np.float32)
    Z = rng.standard_normal((cpu.T, 12)).astype(np.float32)
    for got, ref in ((g.t_matmul(Q), cpu.t_matmul(Q)), (g.matmul(Z), cpu.matmul(Z))):
        assert got.shape == ref.shape and got.dtype == np.float32
        assert np.abs(got - ref).max() <= 1e-4 * np.abs(ref).max()


def test_compress_tile_on_gpu_keeps_the_cpu_subspace():
    raw = _tile()
    a = compress.compress_tile(raw, k0=8, seed=0)
    b = compress.compress_tile(raw, k0=8, seed=0, device="gpu")
    assert a.W.shape == b.W.shape
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.sigma, b.sigma)
    # same column space: principal angles ~0 (the factorization is unique only up to signs)
    Ua, Ub = np.linalg.qr(a.W)[0], np.linalg.qr(b.W)[0]
    assert np.linalg.svd(Ua.T @ Ub, compute_uv=False).min() > 1 - 1e-4
    rec_a = a.W @ a.V.T
    rec_b = b.W @ b.V.T
    assert np.abs(rec_a - rec_b).max() <= 1e-3 * np.abs(rec_a).max()


def test_check_device_names_the_problem():
    with pytest.raises(ValueError, match="device must be one of"):
        gpu.check_device("tpu")
    gpu.check_device("cpu")
