import numpy as np
import pytest

from minime import compress, gpu

pytestmark = pytest.mark.skipif(not gpu.available(), reason="needs wgpu and a GPU")


def _tile(T: int = 3000, h: int = 16, w: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = np.cumsum(rng.normal(0, 1, (T, 3)), axis=0)
    fp = rng.random((3, h * w))
    movie = 60 + sig @ fp + rng.normal(0, 3, (T, h * w))
    return np.clip(movie, 0, 255).astype(np.uint8).reshape(T, h, w)


def _check_products(raw: np.ndarray, **kw) -> None:
    cpu = compress.TileMatrix(raw, chunk=700)  # ragged last chunk on the CPU side
    g = gpu.WgpuTileMatrix(raw, **kw)
    rng = np.random.default_rng(1)
    Q = np.asfortranarray(rng.standard_normal((cpu.P, 12)).astype(np.float32))
    Z = np.asfortranarray(rng.standard_normal((cpu.T, 12)).astype(np.float32))
    for got, ref in ((g.t_matmul(Q), cpu.t_matmul(Q)), (g.matmul(Z), cpu.matmul(Z))):
        assert got.shape == ref.shape and got.dtype == np.float32
        assert np.abs(got - ref).max() <= 1e-4 * np.abs(ref).max()


def test_products_match_cpu():
    _check_products(_tile())


def test_products_on_an_odd_edge_tile_and_ragged_frames():
    # P = 35 x 37 is not a multiple of 4 or 64, T = 1234 not a multiple of 64
    _check_products(_tile(T=1234, h=35, w=37))


def test_products_over_several_gpu_buffers():
    raw = _tile(T=3000)
    g_frames = gpu.chunk_frames(64, max_bytes=64 * 640)  # 640 frames per buffer
    assert g_frames == 640
    _check_products(raw, max_chunk_bytes=64 * 640)


def _same_subspace(a, b) -> None:
    assert a.W.shape == b.W.shape
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.sigma, b.sigma)
    assert np.allclose(a.energy, b.energy)
    # same column space: principal angles ~0 (the factorization is unique only up to signs)
    Ua, Ub = np.linalg.qr(a.W)[0], np.linalg.qr(b.W)[0]
    assert np.linalg.svd(Ua.T @ Ub, compute_uv=False).min() > 1 - 1e-4
    rec_a = a.W @ a.V.T
    rec_b = b.W @ b.V.T
    assert np.abs(rec_a - rec_b).max() <= 1e-3 * np.abs(rec_a).max()


def test_compress_tile_on_gpu_keeps_the_cpu_subspace():
    raw = _tile()
    _same_subspace(
        compress.compress_tile(raw, k0=8, seed=0),
        compress.compress_tile(raw, k0=8, seed=0, device="gpu"),
    )


def test_compress_tile_windows_on_gpu_keeps_the_cpu_subspace():
    raw = _tile(T=3000)

    def read(t0, t1):
        return raw[t0:t1]

    _same_subspace(
        compress.compress_tile_windows(read, len(raw), 1000, k0=8, seed=0),
        compress.compress_tile_windows(read, len(raw), 1000, k0=8, seed=0, device="gpu"),
    )


def test_summary_and_check_device():
    assert isinstance(gpu.summary(), str)
    with pytest.raises(ValueError, match="device must be one of"):
        gpu.check_device("tpu")
    gpu.check_device("cpu")
    gpu.check_device("gpu")
