// The GPU half of a frame worker's per-frame stages (`frame.worker.ts`): batches of up to `maxB` raw
// frames go up once; `denoise` runs the median (and the registration image, read back downsampled for
// the CPU's shift estimate), `correct` then warps and top-hats the denoised frames by the shifts the
// CPU found (and warps the registration image for the quality). The denoised frames and the
// registration images stay on the GPU between the two.
//
// Buffers, one u32 / f32 per pixel: X0 the denoised frames, X1 the registration images, X2 / X3
// scratch (the Gaussian's sums, the warped frames, the top-hat's passes); half-size H0..H2 for the
// half-resolution Gaussian; D0 / D1 the downsampled registration images (plain and warped).

import type { Kernels } from "./kernels";
import * as K from "./kernels";

const ceil = (a: number, b: number) => Math.ceil(a / b);

export type GpuInfo = { vendor: string; architecture: string; device: string; description: string; fallback: boolean };

/** Why a frame worker's GPU stopped (a lost device, a failed submit): its stages go on on the CPU. */
export class GpuFailure extends Error {}

export class GpuFrames {
  readonly hw: number;
  readonly half: number;
  private pipes: Record<string, GPUComputePipeline> = {};
  private readonly buf: Record<string, GPUBuffer> = {};
  private failed = "";
  /** Batches done (the tests' injected device loss counts them). */
  batches = 0;

  constructor(
    readonly device: GPUDevice,
    readonly k: Kernels,
    readonly maxB: number,
  ) {
    const { H, W } = k;
    this.hw = H * W;
    this.half = (H >> 1) * (W >> 1);
    device.lost.then((i) => {
      this.failed ||= `the GPU device was lost (${i.reason || "unknown"}: ${i.message})`;
    });
    device.addEventListener("uncapturederror", (e) => {
      this.failed ||= `GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`;
    });
    const S = GPUBufferUsage.STORAGE;
    const D = GPUBufferUsage.COPY_DST;
    const C = GPUBufferUsage.COPY_SRC;
    const R = GPUBufferUsage.MAP_READ;
    const words = ceil(maxB * this.hw, 4) * 4;
    const mk = (name: string, size: number, usage: number) => {
      this.buf[name] = device.createBuffer({ size: Math.max(16, size), usage, label: name });
    };
    mk("params", 16, GPUBufferUsage.UNIFORM | D);
    mk("in", words, S | D);
    mk("shifts", maxB * 8, S | D);
    mk("taps", 4 * Math.max(1, k.f32taps.length), S | D);
    for (const x of ["X0", "X1", "X2", "X3"]) mk(x, 4 * words, S | C);
    for (const x of ["H0", "H1", "H2"]) mk(x, 4 * maxB * this.half, S);
    mk("RS", 8 * maxB * this.half, S);
    mk("D0", 4 * maxB * this.half, S | C);
    mk("D1", 4 * maxB * this.half, S | C);
    mk("out", words, S | C);
    mk("rD0", 4 * maxB * this.half, R | D);
    mk("rD1", 4 * maxB * this.half, R | D);
    mk("rOut", words, R | D);
    if (k.f32taps.length) device.queue.writeBuffer(this.buf.taps, 0, Float32Array.from(k.f32taps));
  }

  /** Build the pipelines (once; the browser caches the compiled shaders). */
  async compile(): Promise<void> {
    const { k, device } = this;
    const defs: Record<string, string> = {
      unpack: K.unpack,
      pack: K.pack,
      median: K.median(k.median),
      shiftU8: K.shiftU8,
      shiftF32: K.shiftF32,
      areaRowsF32: K.areaRowsF32,
      areaColsF32: K.areaColsF32,
    };
    if (k.tophat > 0) {
      defs.erodeRows = K.morph(k.tophat, true, false);
      defs.erodeCols = K.morph(k.tophat, true, true);
      defs.dilateRows = K.morph(k.tophat, false, false);
      defs.dilateCols = K.morph(k.tophat, false, true);
      defs.tophatFinal = K.tophatFinal;
    }
    if (k.u8taps.length) {
      defs.gaussU8Rows = K.gaussU8Rows(k.u8taps);
      defs.gaussU8Cols = K.gaussU8Cols(k.u8taps);
      defs.areaDownU8 = K.areaDownU8;
      defs.gaussF32Rows = K.gaussF32Rows(k.f32taps.length);
      defs.gaussF32Cols = K.gaussF32Cols(k.f32taps.length);
      defs.upSub = K.upSub;
    }
    const names = Object.keys(defs);
    const built = await Promise.all(
      names.map((n) =>
        device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: device.createShaderModule({ code: defs[n], label: n }), entryPoint: "main" },
          label: n,
        }),
      ),
    );
    names.forEach((n, i) => {
      this.pipes[n] = built[i];
    });
    this.check();
  }

  /** Throw `GpuFailure` once the device is lost or a GPU error was seen. */
  check(): void {
    if (this.failed) throw new GpuFailure(this.failed);
  }

  private pass(enc: GPUCommandEncoder, name: string, bufs: string[], x: number, y: number, z: number): void {
    const pipe = this.pipes[name];
    const p = enc.beginComputePass();
    p.setPipeline(pipe);
    p.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: ["params", ...bufs].map((b, binding) => ({ binding, resource: { buffer: this.buf[b] } })),
      }),
    );
    p.dispatchWorkgroups(x, y, z);
    p.end();
  }

  /** `area_down2_f32` of `n` float32 frames `src` into `dst` (two passes through RS, see `kernels.ts`). */
  private downsample(enc: GPUCommandEncoder, src: string, dst: string, n: number): void {
    const { H, W } = this.k;
    this.pass(enc, "areaRowsF32", [src, "RS"], ceil(W >> 1, 16), ceil(H, 16), n);
    this.pass(enc, "areaColsF32", ["RS", dst], ceil(W >> 1, 16), ceil(H >> 1, 16), n);
  }

  private grid1d(items: number): [number, number] {
    const groups = ceil(items, 256);
    return [Math.min(groups, 65535), ceil(groups, 65535)];
  }

  private setParams(n: number): void {
    const { H, W } = this.k;
    this.device.queue.writeBuffer(this.buf.params, 0, new Uint32Array([H, W, n, n * this.hw]));
  }

  private async read(name: string, bytes: number): Promise<ArrayBuffer> {
    const b = this.buf[name];
    const size = ceil(bytes, 4) * 4;
    try {
      await b.mapAsync(GPUMapMode.READ, 0, size);
    } catch (e) {
      this.check();
      throw new GpuFailure(`a GPU readback failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    const copy = b.getMappedRange(0, size).slice(0, bytes);
    b.unmap();
    this.check();
    return copy;
  }

  /** Upload `n` raw frames (`n * H * W` bytes, no row padding), median them (X0) and, with `reg`, make
   * their registration images (X1) and the downsampled ones; resolves to those (`n` x half pixels,
   * float32) with `reg`, else to null once submitted. */
  async denoise(frames: Uint8Array, n: number, reg: boolean, readDen = false): Promise<Float32Array | null> {
    this.check();
    if (n > this.maxB || frames.length !== n * this.hw) throw new Error(`${n} frames do not fit the GPU batch`);
    const { H, W } = this.k;
    const q = this.device.queue;
    this.setParams(n);
    const bytes = n * this.hw;
    const whole = bytes & ~3;
    q.writeBuffer(this.buf.in, 0, frames.buffer as ArrayBuffer, frames.byteOffset, whole);
    if (bytes > whole) {
      const tail = new Uint8Array(4);
      tail.set(frames.subarray(whole, bytes));
      q.writeBuffer(this.buf.in, whole, tail);
    }
    const enc = this.device.createCommandEncoder();
    const [gx, gy] = this.grid1d(ceil(bytes, 4));
    this.pass(enc, "unpack", ["in", "X2"], gx, gy, 1);
    this.pass(enc, "median", ["X2", "X0"], ceil(W, 16), ceil(H, 16), n);
    if (readDen) {
      this.pass(enc, "pack", ["X0", "out"], gx, gy, 1);
      enc.copyBufferToBuffer(this.buf.out, 0, this.buf.rOut, 0, ceil(bytes, 4) * 4);
    }
    if (reg) {
      const hw2 = [ceil(W >> 1, 16), ceil(H >> 1, 16), n] as const;
      this.pass(enc, "gaussU8Rows", ["X0", "X2"], ceil(W, 16), ceil(H, 16), n);
      this.pass(enc, "gaussU8Cols", ["X2", "X3"], ceil(W, 16), ceil(H, 16), n);
      this.pass(enc, "areaDownU8", ["X0", "H0"], ...hw2);
      this.pass(enc, "gaussF32Rows", ["H0", "H1", "taps"], ...hw2);
      this.pass(enc, "gaussF32Cols", ["H1", "H2", "taps"], ...hw2);
      this.pass(enc, "upSub", ["H2", "X3", "X1"], ceil(W, 16), ceil(H, 16), n);
      this.downsample(enc, "X1", "D0", n);
      enc.copyBufferToBuffer(this.buf.D0, 0, this.buf.rD0, 0, 4 * n * this.half);
      if (readDen) {
        // the self-test's readback of the registration images, made on first use
        this.buf.rReg ??= this.device.createBuffer({
          size: this.buf.X1.size,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          label: "rReg",
        });
        enc.copyBufferToBuffer(this.buf.X1, 0, this.buf.rReg, 0, 4 * bytes);
      }
    }
    q.submit([enc.finish()]);
    this.batches++;
    this.denDebug = readDen ? await this.read("rOut", bytes).then((b) => new Uint8Array(b)) : null;
    this.regDebug = readDen && reg ? await this.read("rReg", 4 * bytes).then((b) => new Uint8Array(b)) : null;
    return reg ? new Float32Array(await this.read("rD0", 4 * n * this.half)) : null;
  }

  /** The denoised frames and registration images (float32 bytes) of the last `denoise(.., readDen =
   * true)` (the self-test). */
  denDebug: Uint8Array | null = null;
  regDebug: Uint8Array | null = null;

  /** Warp the `n` denoised frames by `shifts` (`(n, 2)` float32 dy, dx), top-hat them (`blk`, packed
   * uint8) and, with `warped`, warp the registration images and downsample them (for the quality). */
  async correct(
    shifts: Float32Array,
    n: number,
    want: { blk: boolean; warped: boolean },
  ): Promise<{ blk: Uint8Array | null; warped: Float32Array | null }> {
    this.check();
    const { H, W } = this.k;
    this.setParams(n);
    this.device.queue.writeBuffer(this.buf.shifts, 0, shifts.buffer as ArrayBuffer, shifts.byteOffset, 8 * n);
    const enc = this.device.createCommandEncoder();
    const bytes = n * this.hw;
    const [gx, gy] = this.grid1d(ceil(bytes, 4));
    const g = [ceil(W, 16), ceil(H, 16), n] as const;
    if (want.blk) {
      this.pass(enc, "shiftU8", ["X0", "X2", "shifts"], ...g);
      if (this.k.tophat > 0) {
        const m = [ceil(W, 32), ceil(H, 8), n] as const;
        // X0 (the denoised frames) is free once warped: the top-hat's passes go through X3 and X0
        this.pass(enc, "erodeRows", ["X2", "X3"], ...m);
        this.pass(enc, "erodeCols", ["X3", "X0"], ...m);
        this.pass(enc, "dilateRows", ["X0", "X3"], ...m);
        this.pass(enc, "dilateCols", ["X3", "X0"], ...m);
        this.pass(enc, "tophatFinal", ["X2", "X0", "out"], gx, gy, 1);
      } else this.pass(enc, "pack", ["X2", "out"], gx, gy, 1);
      enc.copyBufferToBuffer(this.buf.out, 0, this.buf.rOut, 0, ceil(bytes, 4) * 4);
    }
    if (want.warped) {
      this.pass(enc, "shiftF32", ["X1", "X3", "shifts"], ...g);
      this.downsample(enc, "X3", "D1", n);
      enc.copyBufferToBuffer(this.buf.D1, 0, this.buf.rD1, 0, 4 * n * this.half);
    }
    this.device.queue.submit([enc.finish()]);
    const [blk, warped] = await Promise.all([
      want.blk ? this.read("rOut", bytes).then((b) => new Uint8Array(b)) : null,
      want.warped ? this.read("rD1", 4 * n * this.half).then((b) => new Float32Array(b)) : null,
    ]);
    return { blk, warped };
  }

  destroy(): void {
    for (const b of Object.values(this.buf)) b.destroy();
  }
}

/** The adapter of this thread's WebGPU, or why there is none to use: no WebGPU, no adapter, or a
 * software one (SwiftShader, a fallback adapter), which is far slower than the CPU path. */
export async function openAdapter(): Promise<{ adapter: GPUAdapter; info: GpuInfo } | { reason: string }> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return { reason: "this browser has no WebGPU" };
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (e) {
    return { reason: `no GPU adapter (${e instanceof Error ? e.message : String(e)})` };
  }
  if (!adapter) return { reason: "WebGPU has no adapter here (no usable GPU, or it is switched off)" };
  const i = adapter.info;
  const info: GpuInfo = {
    vendor: i?.vendor ?? "",
    architecture: i?.architecture ?? "",
    device: i?.device ?? "",
    description: i?.description ?? "",
    fallback: Boolean(i?.isFallbackAdapter ?? adapter.isFallbackAdapter),
  };
  if (info.fallback || /swiftshader/i.test(`${info.architecture} ${info.description}`))
    return { reason: `only a software GPU (${gpuName(info)}), slower than the CPU` };
  return { adapter, info };
}

/** The adapter's name as the panel and `report.json` show it. */
export function gpuName(i: GpuInfo): string {
  const parts = [i.description, i.vendor, i.architecture, i.device].filter(Boolean);
  return [...new Set(parts)].join(" · ") || "GPU";
}

/** A device for `maxB` frames of `k`'s size, with the buffer limits that needs (or the adapter's). */
export async function openDevice(adapter: GPUAdapter, k: Kernels, maxB: number): Promise<GPUDevice> {
  const need = 4 * Math.ceil((maxB * k.H * k.W) / 4) * 4;
  const lim = adapter.limits;
  if (need > lim.maxStorageBufferBindingSize || need > lim.maxBufferSize)
    throw new Error(`a batch of ${maxB} frames needs ${need} bytes per buffer, the GPU allows ${lim.maxBufferSize}`);
  const requiredLimits = {
    maxBufferSize: Math.max(need, Math.min(lim.maxBufferSize, 2 ** 28)),
    maxStorageBufferBindingSize: Math.max(need, Math.min(lim.maxStorageBufferBindingSize, 2 ** 27)),
  };
  return adapter.requestDevice({ requiredLimits, label: "minime" });
}
