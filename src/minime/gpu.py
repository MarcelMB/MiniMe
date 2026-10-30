"""Optional Apple-GPU (MLX) products for the pass-2 randomized SVD.

``MlxTileMatrix`` is ``compress.TileMatrix`` with ``matmul`` / ``t_matmul`` on the GPU: the
uint8 tile is copied to the GPU once and every chunk is converted, centred and multiplied
there in float32, as on the CPU. The mean, energy and noise (float64 sums, FFT), the QR
factorizations and the small SVD stay on the CPU: Apple GPUs have no float64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from minime.compress import SVD_CHUNK, TileMatrix

if TYPE_CHECKING:
    import mlx.core as mx

DEVICES = ("cpu", "gpu")
GPU_CHUNK = 16384  # frames per GPU product: 16384 x 4096 px float32 = 256 MB
INSTALL_HINT = "uv pip install 'minime-cnmf[gpu]' (MLX, Apple-silicon Macs only)"


def available() -> bool:
    """True if MLX imports and a Metal GPU is present."""
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(mx.metal.is_available())


def check_device(device: str) -> None:
    """Raise ``ValueError`` for an unknown device or ``RuntimeError`` for an unusable GPU."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "gpu" and not available():
        raise RuntimeError(f"device 'gpu' needs MLX and a Metal GPU: {INSTALL_HINT}")


class MlxTileMatrix(TileMatrix):
    """``Y`` (P x T, mean removed) of one tile with the products run by MLX on the GPU."""

    def __init__(self, raw: np.ndarray, chunk: int = GPU_CHUNK) -> None:
        import mlx.core as mx

        super().__init__(raw, SVD_CHUNK)  # mean and energy on the CPU, as before
        self.mx = mx
        self.gchunk = chunk
        self.raw_g = mx.array(np.ascontiguousarray(self.raw))  # (T, P) uint8
        self.mean_g = mx.array(self.mean)

    def _block_g(self, c0: int) -> mx.array:
        """Frames c0.. as (n, P) float32 on the GPU, mean removed."""
        return self.raw_g[c0 : c0 + self.gchunk].astype(self.mx.float32) - self.mean_g

    def t_matmul(self, Q: np.ndarray) -> np.ndarray:
        """``Y^T Q``: (T, ell)."""
        mx = self.mx
        Qg = mx.array(np.ascontiguousarray(Q, dtype=np.float32))
        parts = []
        for c0 in range(0, self.T, self.gchunk):
            part = self._block_g(c0) @ Qg
            mx.eval(part)  # one chunk at a time: the float copy never exceeds gchunk x P
            parts.append(part)
        return np.asfortranarray(np.array(mx.concatenate(parts, axis=0)))

    def matmul(self, Z: np.ndarray) -> np.ndarray:
        """``Y Z``: (P, ell)."""
        mx = self.mx
        Zg = mx.array(np.ascontiguousarray(Z, dtype=np.float32))
        acc = mx.zeros((self.P, Z.shape[1]), dtype=mx.float32)
        for c0 in range(0, self.T, self.gchunk):
            acc = acc + self._block_g(c0).T @ Zg[c0 : c0 + self.gchunk]
            mx.eval(acc)
        return np.asfortranarray(np.array(acc))
