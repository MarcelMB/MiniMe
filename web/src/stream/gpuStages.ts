// The GPU path of a frame worker's tasks (`frame.worker.ts`): pass 1 (with the shift estimate), the
// streaming pass's preprocessing and the template, a batch of frames at a time. The GPU does the
// kernels (`gpu/frames.ts`), the stage's wasm engine everything after them with the CPU path's own code
// (`rust/minime-core/src/gpu.rs`), so the outputs are the CPU path's bit for bit. A GPU that fails (a
// lost device, an error) is switched off for the rest of the run: the frames from the failed batch on
// go through the CPU path, which nothing of the failed batch has touched.

import { GpuFailure, GpuFrames, type GpuInfo, gpuName, openAdapter, openDevice } from "../gpu/frames";
import type { Kernels } from "../gpu/kernels";
import { forFrames, type Segment } from "./frames";
import {
  gpu_kernels,
  gpu_reference,
  type Pass1Engine,
  type StreamSolver,
  type TemplateWorker,
} from "./pkg/minime_core.js";
import { FLUSH } from "./slabs";

/** Frames per GPU batch: a worker's GPU buffers are ~8 MB per frame (one u32 / f32 per pixel of the
 * denoised frame, the registration image and two scratch images, at 600 x 600). */
export const BATCH = 8;

/** `n` raw frames `[from, to)` of the session without row padding, one after the other. */
export function readFrames(segs: Segment[], from: number, to: number, H: number, W: number): Uint8Array {
  const px = H * W;
  const out = new Uint8Array((to - from) * px);
  let [i, stride] = [0, W];
  forFrames(
    segs,
    from,
    to,
    (s) => {
      stride = s;
    },
    (c) => {
      const o = out.subarray(i * px, (i + 1) * px);
      if (stride === W) o.set(c.subarray(0, px));
      else for (let y = 0; y < H; y++) o.set(c.subarray(y * stride, y * stride + W), y * W);
      i++;
    },
  );
  return out;
}

/** This frame worker's GPU: opened when the run turns it on, its buffers built for a stage's kernels on
 * first use (and kept for a later stage whose kernels they cover). `off` says why it went off. */
export class WorkerGpu {
  off = "";
  info: GpuInfo | null = null;
  private adapter: GPUAdapter | null = null;
  private frames: GpuFrames | null = null;
  /** Test hooks (`ToWorker.gpuTest`): "lost" loses the device after a few batches. */
  constructor(private readonly test = "") {}

  /** Open the adapter; resolves to why it cannot be used ("" when it can). */
  async open(): Promise<string> {
    if (this.test === "nogpu") this.off = "this browser has no WebGPU (test)";
    if (this.off) return this.off;
    const a = await openAdapter();
    if ("reason" in a) {
      this.off = a.reason;
      return this.off;
    }
    this.adapter = a.adapter;
    this.info = a.info;
    return "";
  }

  get name(): string {
    return this.info ? gpuName(this.info) : "";
  }

  /** The GPU buffers for kernels `k`, or null when the GPU is off. */
  async for(k: Kernels): Promise<GpuFrames | null> {
    if (this.off || !this.adapter) return null;
    const f = this.frames;
    const covers =
      f &&
      f.k.H === k.H &&
      f.k.W === k.W &&
      f.k.median === k.median &&
      f.k.tophat === k.tophat &&
      (k.u8taps.length === 0 || JSON.stringify([f.k.u8taps, f.k.f32taps]) === JSON.stringify([k.u8taps, k.f32taps]));
    if (covers) return f;
    try {
      f?.destroy();
      this.frames = null;
      const device = await openDevice(this.adapter, k, BATCH);
      const g = new GpuFrames(device, k, BATCH);
      await g.compile();
      this.frames = g;
      return g;
    } catch (e) {
      this.fail(e);
      return null;
    }
  }

  /** Free the GPU's buffers and device (the worker's GPU is turned off). */
  close(): void {
    const f = this.frames;
    this.frames = null;
    try {
      f?.destroy();
      f?.device.destroy();
    } catch {
      // gone already
    }
  }

  /** Switch the GPU off for the rest of the run (`off` the reason). */
  fail(e: unknown): void {
    this.off ||= e instanceof Error ? e.message : String(e);
    try {
      this.frames?.destroy();
    } catch {
      // a lost device's buffers are gone already
    }
    this.frames = null;
  }

  /** After each batch: the "lost" test hook loses the device after the third batch ("lost:N": the N-th). */
  afterBatch(g: GpuFrames): void {
    const m = /^lost(?::(\d+))?$/.exec(this.test);
    if (m && g.batches === Number(m[1] ?? 3)) g.device.destroy();
  }
}

/** `f()`, a call into the GPU only: whatever it throws switches the GPU off (null). The engines' calls
 * stay outside, so their errors are the run's. */
async function onGpu<T>(gpu: WorkerGpu, f: () => Promise<T>): Promise<T | null> {
  try {
    return await f();
  } catch (e) {
    gpu.fail(e instanceof GpuFailure ? e : new GpuFailure(`GPU error: ${e instanceof Error ? e.message : String(e)}`));
    return null;
  }
}

/** A stage's kernels (`gpu_kernels()` of its engine), or null with the GPU off when the GPU cannot give
 * the stage's bits (its parameters, `gpuplan::kernels`). */
export function stageKernels(gpu: WorkerGpu, get: () => string): Kernels | null {
  try {
    return JSON.parse(get()) as Kernels;
  } catch (e) {
    gpu.fail(e);
    return null;
  }
}

/** Batch sizes from window frame `a` to `b` of a chunk whose engine holds `buffered` frames: at most
 * `BATCH`, never across a multiple of `FLUSH` (the tile store's blocks). */
export function batches(a: number, b: number, buffered: number): number[] {
  const out: number[] = [];
  let c = buffered;
  for (let f = a; f < b; ) {
    const n = Math.min(BATCH, b - f, FLUSH - c);
    out.push(n);
    f += n;
    c = (c + n) % FLUSH;
  }
  return out;
}

/** Pass 1 on window frames `[a, b)` through the GPU: returns the frame it got to (`b`, or where the GPU
 * failed: the caller does the rest on the CPU). `flush` writes the tile buffers out at every `FLUSH`
 * frames of the chunk, as the CPU path does. The GPU work overlaps the CPU's: batch `i`'s warp and
 * top-hat run while the next batch's frames are read, and the next batch's median and registration
 * image while batch `i` is folded and written. */
export async function pass1Gpu(
  engine: Pass1Engine,
  gpu: WorkerGpu,
  segs: Segment[],
  a: number,
  b: number,
  flush: () => void,
  tick: (done: number) => void,
): Promise<number> {
  if (a >= b) return a;
  const k = stageKernels(gpu, () => engine.gpu_kernels());
  const g = k && (await gpu.for(k));
  if (!g) return a;
  const [H, W, t0] = [engine.height(), engine.width(), engine.t0()];
  const sizes = batches(a, b, engine.buffered());
  const starts = sizes.map((_, i) => a + sizes.slice(0, i).reduce((x, y) => x + y, 0));
  const read = (i: number) => readFrames(segs, t0 + starts[i], t0 + starts[i] + sizes[i], H, W);
  const quiet = <T>(p: Promise<T | null>) => {
    p.catch(() => undefined);
    return p;
  };
  const first = read(0);
  let down = quiet(onGpu(gpu, () => g.denoise(first, sizes[0], true)));
  for (let i = 0; i < sizes.length; i++) {
    const [f, n] = [starts[i], sizes[i]];
    const d = await down;
    if (!d) return f;
    const given = engine.given_shifts(f, n);
    let est = new Float64Array(0);
    let used = given;
    if (!given.length) {
      est = engine.estimate_down(n, d);
      used = Float32Array.from(est);
    }
    const corrected = quiet(onGpu(gpu, () => g.correct(used, n, { blk: true, warped: true })));
    const next = i + 1 < sizes.length ? read(i + 1) : null;
    const out = await corrected;
    if (!out) return f;
    gpu.afterBatch(g);
    // the next batch's median and registration image go on while this one is folded and written
    if (next) down = quiet(onGpu(gpu, () => g.denoise(next, sizes[i + 1], true)));
    engine.push_corrected(n, out.blk as Uint8Array, out.warped as Float32Array, est, used);
    if (engine.buffered() === FLUSH) flush();
    tick(f + n - a);
  }
  return b;
}

/** The streaming pass's frames `[a, b)`: preprocessed on the GPU, solved by `solver`; returns the frame it
 * got to (see `pass1Gpu`). */
export async function streamGpu(
  solver: StreamSolver,
  gpu: WorkerGpu,
  segs: Segment[],
  H: number,
  W: number,
  a: number,
  b: number,
  tick: (done: number) => void,
): Promise<number> {
  if (a >= b) return a;
  const k = stageKernels(gpu, () => solver.gpu_kernels());
  const g = k && (await gpu.for(k));
  if (!g) return a;
  const sizes: number[] = [];
  for (let f = a; f < b; f += BATCH) sizes.push(Math.min(BATCH, b - f));
  const prep = (f: number, n: number) => {
    const raw = readFrames(segs, f, f + n, H, W);
    const sh = solver.shifts(f, n);
    const p = onGpu(gpu, async () => {
      await g.denoise(raw, n, false);
      return g.correct(sh, n, { blk: true, warped: false });
    });
    p.catch(() => undefined);
    return p;
  };
  let f = a;
  let pending = prep(a, sizes[0]);
  for (let i = 0; i < sizes.length; i++) {
    const n = sizes[i];
    const out = await pending;
    if (!out) return f;
    gpu.afterBatch(g);
    // the next batch is read and preprocessed while this one is solved
    if (i + 1 < sizes.length) pending = prep(f + n, sizes[i + 1]);
    solver.solve_prepped(f, n, out.blk as Uint8Array);
    f += n;
    tick(f - a);
  }
  return f;
}

/** Template frames `[a, b)` of round 1 (prepared registration images) or round 2 (aligned to the first
 * template, then prepared), pushed to `preps` in order; returns the frame it got to (see `pass1Gpu`). */
export async function templateGpu(
  template: TemplateWorker,
  gpu: WorkerGpu,
  segs: Segment[],
  round: 1 | 2,
  a: number,
  b: number,
  preps: Float32Array[],
): Promise<number> {
  const kernels = stageKernels(gpu, () => template.gpu_kernels());
  const g = kernels && (await gpu.for(kernels));
  if (!g) return a;
  let f = a;
  while (f < b) {
    const n = Math.min(BATCH, b - f);
    const raw = readFrames(segs, f, f + n, kernels.H, kernels.W);
    const down = await onGpu(gpu, async () => (await g.denoise(raw, n, true)) as Float32Array);
    if (!down) return f;
    let img = down;
    if (round === 2) {
      const used = Float32Array.from(template.estimate_first_down(n, down));
      const w = await onGpu(gpu, async () => (await g.correct(used, n, { blk: false, warped: true })).warped);
      if (!w) return f;
      img = w;
    }
    gpu.afterBatch(g);
    const all = template.prep_down(n, img);
    const len = all.length / n;
    for (let i = 0; i < n; i++) preps.push(all.subarray(i * len, (i + 1) * len));
    f += n;
  }
  return f;
}

/** The self-test: session frames `at` through every GPU kernel against the CPU's results
 * (`gpu_reference`), with `shifts[i]` for frame `i` (ties, sub-pixel, borders). Resolves to "" when
 * every value is the CPU's, bit for bit, else to what differed. */
export async function selfTest(
  gpu: WorkerGpu,
  segs: Segment[],
  params: Uint8Array,
  at: number[],
  shifts: number[],
  test = "",
): Promise<string> {
  const k = stageKernels(gpu, () => gpu_kernels(params));
  const g = k && (await gpu.for(k));
  if (!k || !g) return gpu.off || "the GPU did not start";
  const px = k.H * k.W;
  const n = at.length;
  const raw = new Uint8Array(n * px);
  at.forEach((t, i) => {
    raw.set(readFrames(segs, t, t + 1, k.H, k.W), i * px);
  });
  const sh = Float32Array.from(shifts);
  const r = await onGpu(gpu, async () => {
    const down = (await g.denoise(raw, n, true, true)) as Float32Array;
    const den = g.denDebug as Uint8Array;
    const reg = g.regDebug as Uint8Array;
    return { down, den, reg, out: await g.correct(sh, n, { blk: true, warped: true }) };
  });
  if (!r) return gpu.off;
  const { down, den, reg, out } = r;
  if (test === "mismatch") (out.blk as Uint8Array)[12345 % px] ^= 1;
  const half = down.length / n;
  const parts: [string, Uint8Array, (i: number) => Uint8Array][] = [
    ["median", den, (i) => den.subarray(i * px, (i + 1) * px)],
    ["registration image", reg, (i) => reg.subarray(4 * i * px, 4 * (i + 1) * px)],
    ["downsampled registration image", new Uint8Array(0), (i) => new Uint8Array(down.buffer, 4 * i * half, 4 * half)],
    ["shift + top-hat", new Uint8Array(0), (i) => (out.blk as Uint8Array).subarray(i * px, (i + 1) * px)],
    [
      "shifted registration image",
      new Uint8Array(0),
      (i) => new Uint8Array((out.warped as Float32Array).buffer, 4 * i * half, 4 * half),
    ],
  ];
  const names = ["den", "reg", "down", "blk", "warped_down"];
  for (let i = 0; i < n; i++) {
    const ref = gpu_reference(raw.subarray(i * px, (i + 1) * px), params, sh[2 * i], sh[2 * i + 1]) as Map<
      string,
      Uint8Array
    >;
    for (const [j, [what, , got]] of parts.entries()) {
      const want = ref.get(names[j]) as Uint8Array;
      const mine = got(i);
      let bad = 0;
      let first = -1;
      for (let q = 0; q < want.length; q++)
        if (mine[q] !== want[q]) {
          bad++;
          if (first < 0) first = q;
        }
      if (bad || mine.length !== want.length) {
        let values = "";
        if (first >= 0 && j !== 0 && j !== 3) {
          const q = first >> 2;
          const f = (b: Uint8Array) =>
            new Float32Array(b.buffer.slice(b.byteOffset + 4 * q, b.byteOffset + 4 * q + 4))[0];
          values = `: ${f(mine)} on the GPU, ${f(want)} on the CPU`;
          if (j === 2) {
            // the four registration-image values behind a downsampled one
            const w2 = k.W >> 1;
            const [y, x] = [Math.floor(q / w2), q % w2];
            const r = ref.get("reg") as Uint8Array;
            const v = (yy: number, xx: number) =>
              new Float32Array(
                r.buffer.slice(r.byteOffset + 4 * (yy * k.W + xx), r.byteOffset + 4 * (yy * k.W + xx) + 4),
              )[0];
            values += ` from ${[v(2 * y, 2 * x), v(2 * y, 2 * x + 1), v(2 * y + 1, 2 * x), v(2 * y + 1, 2 * x + 1)].join(", ")}`;
          }
        }
        return `self-test: the GPU's ${what} differs from the CPU's in ${bad} of ${want.length} bytes (frame ${at[i]}, first at byte ${first}${values})`;
      }
    }
  }
  return "";
}

/** The speed test's pass 1 on frames `[a, b)` of the session (a throwaway engine, `files` from
 * `bench_template`): through the GPU or the CPU path, the tiles dropped. Resolves to its milliseconds. */
export async function benchPass1(
  engine: Pass1Engine,
  gpu: WorkerGpu | null,
  segs: Segment[],
  a: number,
  b: number,
): Promise<number> {
  const began = performance.now();
  const drop = () => void engine.take_tiles();
  let f = 0;
  if (gpu) f = await pass1Gpu(engine, gpu, segs, 0, b - a, drop, () => undefined);
  forFrames(
    segs,
    a + f,
    b,
    (s) => engine.set_stride(s),
    (c) => {
      engine.push_raw(c);
      if (engine.buffered() === FLUSH) drop();
    },
  );
  return performance.now() - began;
}
