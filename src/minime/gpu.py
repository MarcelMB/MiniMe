"""Optional GPU products for the pass-2 randomized SVD, on any GPU through wgpu (WebGPU).

``WgpuTileMatrix`` is ``compress.TileMatrix`` with ``matmul`` / ``t_matmul`` run by two WGSL
compute shaders: the uint8 tile is packed four pixels per 32-bit word, uploaded once, and
every product unpacks, centres and multiplies it in float32 on the GPU. The mean, energy and
noise (float64 sums, FFT), the QR factorizations and the small SVD stay on the CPU, as on the
CPU path.

wgpu talks to Metal (macOS), Vulkan (Linux) and DX12 / Vulkan (Windows) through the graphics
driver the OS already has, so integrated Intel, AMD and Apple GPUs work without a toolkit. No
GPU here has float64; the products are float32 like the CPU's, summed in a different order,
so results match the CPU path to float32 rounding, not bit for bit.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from minime.compress import SVD_CHUNK, TileMatrix

DEVICES = ("cpu", "gpu")
INSTALL_HINT = (
    "uv pip install 'minime-cnmf[gpu]' (wgpu: any GPU with a Metal, Vulkan or DX12 driver)"
)
TILE = 64  # output tile of one workgroup (16 x 16 threads, 4 x 4 outputs each)
TK = 32  # reduction tile
GPU_CHUNK_BYTES = 1 << 30  # at most this much of the packed tile per GPU buffer
MIN_WORKGROUPS = 2048  # split the frame reduction of ``matmul`` until this many run at once
_UNIFORM_WORDS = 8

# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

_HEADER = """
struct Params { T: u32, P: u32, ell: u32, wpr: u32, L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> y: array<u32>;         // T x P/4 words, row-major
@group(0) @binding(1) var<storage, read> b: array<f32>;         // Q: P x ell, or Z: T x ell
@group(0) @binding(2) var<storage, read> mean: array<f32>;      // P
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> prm: Params;
const TK: u32 = 32u;
var<workgroup> As: array<array<f32, 64>, TK>;        // [k][row]
var<workgroup> Bs: array<array<vec4<f32>, 16>, TK>;  // [k][col / 4]
"""

# out[t, j] = sum_p (Y[t, p] - mean[p]) Q[p, j]; one 64 x 64 (t, j) tile per workgroup.
_SHADER_YT_Q = (
    _HEADER
    + """
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x; let ty = lid.y;
    let lin = ty * 16u + tx;
    let t0 = wid.y * 64u; let j0 = wid.x * 64u;
    var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
    for (var k0: u32 = 0u; k0 < prm.P; k0 += TK) {
        for (var r = 0u; r < 2u; r++) {              // 64 rows x 8 words
            let idx = lin + r * 256u;
            let tl = idx / 8u; let wl = idx % 8u;
            let word = y[(t0 + tl) * prm.wpr + (k0 / 4u) + wl];
            let kl = wl * 4u;
            As[kl][tl]      = f32(word & 0xFFu)          - mean[k0 + kl];
            As[kl + 1u][tl] = f32((word >> 8u) & 0xFFu)  - mean[k0 + kl + 1u];
            As[kl + 2u][tl] = f32((word >> 16u) & 0xFFu) - mean[k0 + kl + 2u];
            As[kl + 3u][tl] = f32(word >> 24u)           - mean[k0 + kl + 3u];
        }
        for (var r = 0u; r < 2u; r++) {              // 32 px x 16 vec4
            let idx = lin + r * 256u;
            let kl = idx / 16u; let jl = idx % 16u;
            let base = (k0 + kl) * prm.ell + j0 + jl * 4u;
            Bs[kl][jl] = vec4<f32>(b[base], b[base + 1u], b[base + 2u], b[base + 3u]);
        }
        workgroupBarrier();
        for (var k = 0u; k < TK; k++) {
            let bv = Bs[k][tx];
            acc0 += As[k][ty * 4u] * bv;      acc1 += As[k][ty * 4u + 1u] * bv;
            acc2 += As[k][ty * 4u + 2u] * bv; acc3 += As[k][ty * 4u + 3u] * bv;
        }
        workgroupBarrier();
    }
    let ob = (t0 + ty * 4u) * prm.ell + j0 + tx * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let v = accs[i];
        let o = ob + i * prm.ell;
        out[o] = v.x; out[o + 1u] = v.y; out[o + 2u] = v.z; out[o + 3u] = v.w;
    }
}
"""
)

# out[s, p, j] = sum_{t in slice s} (Y[t, p] - mean[p]) Z[t, j]; one 64 x 64 (p, j) tile and
# one slice of ``L`` frames per workgroup; the slices are summed on the CPU.
_SHADER_Y_Z = (
    _HEADER
    + """
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x; let ty = lid.y;
    let lin = ty * 16u + tx;
    let p0 = wid.y * 64u; let j0 = wid.x * 64u;
    let ts0 = wid.z * prm.L;
    let ts1 = min(ts0 + prm.L, prm.T);
    var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
    for (var t0: u32 = ts0; t0 < ts1; t0 += TK) {
        for (var r = 0u; r < 2u; r++) {              // 32 frames x 16 words
            let idx = lin + r * 256u;
            let tl = idx / 16u; let wl = idx % 16u;
            let word = y[(t0 + tl) * prm.wpr + (p0 / 4u) + wl];
            let pl = wl * 4u;
            As[tl][pl]      = f32(word & 0xFFu)          - mean[p0 + pl];
            As[tl][pl + 1u] = f32((word >> 8u) & 0xFFu)  - mean[p0 + pl + 1u];
            As[tl][pl + 2u] = f32((word >> 16u) & 0xFFu) - mean[p0 + pl + 2u];
            As[tl][pl + 3u] = f32(word >> 24u)           - mean[p0 + pl + 3u];
        }
        for (var r = 0u; r < 2u; r++) {              // 32 frames x 16 vec4
            let idx = lin + r * 256u;
            let kl = idx / 16u; let jl = idx % 16u;
            let base = (t0 + kl) * prm.ell + j0 + jl * 4u;
            Bs[kl][jl] = vec4<f32>(b[base], b[base + 1u], b[base + 2u], b[base + 3u]);
        }
        workgroupBarrier();
        for (var k = 0u; k < TK; k++) {
            let bv = Bs[k][tx];
            acc0 += As[k][ty * 4u] * bv;      acc1 += As[k][ty * 4u + 1u] * bv;
            acc2 += As[k][ty * 4u + 2u] * bv; acc3 += As[k][ty * 4u + 3u] * bv;
        }
        workgroupBarrier();
    }
    let ob = wid.z * prm.P * prm.ell + (p0 + ty * 4u) * prm.ell + j0 + tx * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let v = accs[i];
        let o = ob + i * prm.ell;
        out[o] = v.x; out[o + 1u] = v.y; out[o + 2u] = v.z; out[o + 3u] = v.w;
    }
}
"""
)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


@functools.cache
def _adapter() -> Any | None:
    """The wgpu adapter of this process, or ``None`` when wgpu or a usable GPU is missing."""
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:  # noqa: BLE001  (ImportError, or wgpu's "no suitable adapter")
        return None


@functools.cache
def _device() -> Any:
    """The wgpu device of this process, with the buffer limits raised to the adapter's."""
    adapter = _adapter()
    if adapter is None:
        raise RuntimeError(f"no usable GPU for wgpu: {INSTALL_HINT}")
    lim = adapter.limits
    return adapter.request_device_sync(
        required_limits={
            "max-buffer-size": lim["max-buffer-size"],
            "max-storage-buffer-binding-size": lim["max-storage-buffer-binding-size"],
        }
    )


@functools.cache
def _pipeline(name: str) -> Any:
    import wgpu  # noqa: F401  (the device exists already)

    dev = _device()
    src = {"yt_q": _SHADER_YT_Q, "y_z": _SHADER_Y_Z}[name]
    module = dev.create_shader_module(code=src)
    return dev.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"}
    )


def available() -> bool:
    """True if wgpu imports and finds a GPU."""
    return _adapter() is not None


def summary() -> str | None:
    """The GPU wgpu would use (``"Apple M1 Max (IntegratedGPU) via Metal"``), or ``None``."""
    a = _adapter()
    return None if a is None else str(a.summary)


def check_device(device: str) -> None:
    """Raise ``ValueError`` for an unknown device or ``RuntimeError`` for an unusable GPU."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "gpu" and not available():
        raise RuntimeError(f"device 'gpu' needs wgpu and a GPU: {INSTALL_HINT}")


def chunk_frames(P_padded: int, max_bytes: int | None = None) -> int:
    """Frames of a tile with ``P_padded`` pixels per GPU buffer: a multiple of ``TILE`` that
    fits the adapter's storage-binding limit and ``GPU_CHUNK_BYTES``."""
    lim = _adapter().limits
    cap = min(int(lim["max-buffer-size"]), int(lim["max-storage-buffer-binding-size"]))
    cap = min(cap, GPU_CHUNK_BYTES if max_bytes is None else max_bytes)
    n = cap // P_padded // TILE * TILE
    if n < TILE:
        raise RuntimeError(f"GPU buffers of {cap} bytes cannot hold {TILE} frames of {P_padded} px")
    return n


# ---------------------------------------------------------------------------
# Tile matrix
# ---------------------------------------------------------------------------


class WgpuTileMatrix(TileMatrix):
    """``Y`` (P x T, mean removed) of one tile with the products run by wgpu on the GPU.

    ``max_chunk_bytes`` caps the packed frames per GPU buffer (tests use it to force several
    chunks); the default is the adapter limit or ``GPU_CHUNK_BYTES``.
    """

    def __init__(
        self, raw: np.ndarray, chunk: int = SVD_CHUNK, max_chunk_bytes: int | None = None
    ) -> None:
        import wgpu

        super().__init__(raw, chunk)  # mean and energy on the CPU, as before
        self.dev = _device()
        self.Pp = _round_up(self.P, TILE)
        self.frames = chunk_frames(self.Pp, max_chunk_bytes)
        mean_p = np.zeros(self.Pp, dtype=np.float32)
        mean_p[: self.P] = self.mean
        self.mean_buf = self.dev.create_buffer_with_data(
            data=mean_p, usage=wgpu.BufferUsage.STORAGE
        )
        # the tile as (Tp, Pp / 4) little-endian words, in buffers of at most ``frames`` rows;
        # new buffers are zero-filled, so a full-width tile is written straight from ``raw``
        # and only an edge tile (P < Pp) goes through a padded copy
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        self.chunks: list[tuple[int, int, Any]] = []  # (t0, padded rows, buffer)
        for t0 in range(0, self.T, self.frames):
            n = min(self.frames, self.T - t0)
            n_pad = _round_up(n, TILE)
            buf = self.dev.create_buffer(size=n_pad * self.Pp, usage=usage)
            if self.Pp == self.P:
                self.dev.queue.write_buffer(buf, 0, np.ascontiguousarray(self.raw[t0 : t0 + n]))
            else:
                block = np.zeros((n, self.Pp), dtype=np.uint8)
                block[:, : self.P] = self.raw[t0 : t0 + n]
                self.dev.queue.write_buffer(buf, 0, block)
                del block
            self.chunks.append((t0, n_pad, buf))
        self.dev.queue.submit([])

    def _release_cpu_tile(self) -> None:
        """Drop the CPU copy of the tile: the products only need the GPU buffers. Called by
        the first product, after ``noise()`` (which reads ``raw``) has run in every caller."""
        self.raw = None  # type: ignore[assignment]

    def _params(self, T: int, ellp: int, L: int = 0) -> Any:
        import wgpu

        prm = np.zeros(_UNIFORM_WORDS, dtype=np.uint32)
        prm[:5] = (T, self.Pp, ellp, self.Pp // 4, L)
        return self.dev.create_buffer_with_data(data=prm, usage=wgpu.BufferUsage.UNIFORM)

    def _run(
        self, name: str, y_buf: Any, b: np.ndarray, out_size: int, prm: Any, wg: tuple
    ) -> np.ndarray:
        """Dispatch ``name`` over ``wg`` workgroups and return ``out`` as float32."""
        import wgpu

        dev = self.dev
        b_buf = dev.create_buffer_with_data(data=b, usage=wgpu.BufferUsage.STORAGE)
        out_buf = dev.create_buffer(
            size=out_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        pipe = _pipeline(name)
        bg = dev.create_bind_group(
            layout=pipe.get_bind_group_layout(0),
            entries=[
                {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                for i, buf in enumerate((y_buf, b_buf, self.mean_buf, out_buf, prm))
            ],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*wg)
        cp.end()
        dev.queue.submit([enc.finish()])
        res = np.frombuffer(dev.queue.read_buffer(out_buf), dtype=np.float32)
        b_buf.destroy()
        out_buf.destroy()
        return res

    def t_matmul(self, Q: np.ndarray) -> np.ndarray:
        """``Y^T Q``: (T, ell)."""
        self._release_cpu_tile()
        ell = Q.shape[1]
        ellp = _round_up(ell, TILE)
        Qp = np.zeros((self.Pp, ellp), dtype=np.float32)
        Qp[: self.P, :ell] = Q
        out = np.empty((self.T, ell), dtype=np.float32, order="F")
        for t0, n_pad, y_buf in self.chunks:
            prm = self._params(n_pad, ellp)
            res = self._run("yt_q", y_buf, Qp, n_pad * ellp, prm, (ellp // TILE, n_pad // TILE, 1))
            n = min(n_pad, self.T - t0)
            out[t0 : t0 + n] = res.reshape(n_pad, ellp)[:n, :ell]
            prm.destroy()
        return out

    def matmul(self, Z: np.ndarray) -> np.ndarray:
        """``Y Z``: (P, ell)."""
        self._release_cpu_tile()
        ell = Z.shape[1]
        ellp = _round_up(ell, TILE)
        acc = np.zeros((self.Pp, ellp), dtype=np.float32)
        tiles = (self.Pp // TILE) * (ellp // TILE)
        for t0, n_pad, y_buf in self.chunks:
            n = min(n_pad, self.T - t0)
            Zp = np.zeros((n_pad, ellp), dtype=np.float32)
            Zp[:n, :ell] = Z[t0 : t0 + n]
            slices = max(1, min(-(-MIN_WORKGROUPS // tiles), n_pad // TK))
            L = _round_up(-(-n_pad // slices), TK)
            slices = -(-n_pad // L)
            prm = self._params(n_pad, ellp, L)
            res = self._run(
                "y_z",
                y_buf,
                Zp,
                slices * self.Pp * ellp,
                prm,
                (ellp // TILE, self.Pp // TILE, slices),
            )
            acc += res.reshape(slices, self.Pp, ellp).sum(axis=0)
            prm.destroy()
        return np.asfortranarray(acc[: self.P, :ell])
