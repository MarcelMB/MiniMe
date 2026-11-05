// Where a run's per-frame stages go (the run's worker, `stream.worker.ts`): the GPU self-test on the
// session's own frames, the speed test (cached per machine, browser GPU, frame size and worker
// count), the choice (`gpu/plan.ts`) and turning each frame worker's GPU on or off.

import { idbKV, type KV } from "../browser/kv";
import { chunkWeights, type Device, decide, type SpeedTest, speedKey, splitWorkers } from "../gpu/plan";
import type { Segment } from "./frames";
import { readFrames } from "./gpuStages";
import { bench_template } from "./pkg/minime_core.js";
import type { Pool } from "./pool";
import type { GtResult } from "./poolProtocol";

/** `report.json`'s `device`: what was asked, what ran, on which GPU, why, and the tests behind it. */
export type DeviceReport = {
  asked: Device;
  used: "cpu" | "gpu" | "both";
  gpu: string;
  /** Why the run took what it took ("the GPU was fastest", the self-test's failure, ...). */
  why: string;
  self_test: string;
  workers: number;
  gpu_workers: number;
  speed_test: SpeedTest | null;
  speed_test_cached: boolean;
  /** The self-test's and speed test's seconds at the run's start. */
  seconds: number;
  /** Frames each stage did on the GPU (`p1` pass 1, `s1` streaming, `t1` / `t2` the template). */
  gpu_frames: Record<string, number>;
  /** A frame worker's GPU that went off during the run, and why (its frames then took the CPU path). */
  fallbacks: string[];
};

/** Self-test frames (start, middle, end of the session) and their shifts: sub-pixel, a tie, and one
 * far enough for the border's zeros. */
const SELF_TEST_SHIFTS = [1.37, -2.61, -0.5, 0.5, -13.25, 7.0625];
/** Frames per worker in each speed-test run (four GPU batches), and runs per mode: the fastest counts,
 * as whatever else the machine does only slows a run down. */
const SPEED_FRAMES = 32;
const SPEED_REPEATS = 2;

let kv: KV | null = null;
const cache = () => (kv ??= idbKV("speed"));
const encode = (v: unknown) => new TextEncoder().encode(JSON.stringify(v));

/** Turn the GPU of the first `gpuWorkers` frame workers on and the rest off; returns the GPU name the
 * first one reports and the workers whose GPU would not open (with why). */
export async function turnGpu(pool: Pool, gpuWorkers: number, test: string): Promise<{ name: string; off: string[] }> {
  const off: string[] = [];
  let name = "";
  await pool.run("g0", pool.size, (i) => ({ type: "g0", on: i < gpuWorkers, test }), {
    deal: true,
    onResult: (i, r) => {
      if (i < gpuWorkers && r.off) off.push(`frame worker ${i + 1}: ${r.off}`);
      name ||= r.name;
    },
  });
  return { name, off };
}

/** The GPU self-test on worker 0: session frames through every kernel, compared with the CPU's. */
export async function selfTest(pool: Pool, params: Uint8Array, T: number, test: string): Promise<GtResult> {
  if (test === "nogpu") return { failure: "this browser has no WebGPU (test)", name: "" };
  const at = [0, Math.floor(T / 2), T - 1];
  let gt: GtResult = { failure: "the self-test gave no result", name: "" };
  if (test) await turnGpu(pool, 1, test); // the hook travels with the GPU
  await pool.run("gt", 1, () => ({ type: "gt", params, at, shifts: SELF_TEST_SHIFTS }), {
    deal: true,
    onResult: (_, r) => {
      gt = r;
    },
  });
  return gt;
}

/** Pass-1 frames per second on the pool: CPU (every worker), GPU (one worker, then every worker) and
 * the split, each worker on its own `SPEED_FRAMES` frames of the session. */
export async function speedTest(
  pool: Pool,
  segs: Segment[],
  params: Uint8Array,
  T: number,
  H: number,
  W: number,
  gpu: string,
  key: string,
  test: string,
): Promise<SpeedTest> {
  const began = performance.now();
  const N = pool.size;
  const F = Math.min(SPEED_FRAMES, T);
  const pass1 = JSON.parse(new TextDecoder().decode(params)) as Record<string, unknown>;
  const template = bench_template(params, readFrames(segs, 0, 1, H, W));
  const starts = Array.from({ length: N }, (_, i) => Math.floor((i * Math.max(0, T - F)) / Math.max(1, N - 1 || 1)));
  const task = (i: number, n = F) => {
    const a = starts[i];
    const files = new Map([
      ["params.json", encode({ ...pass1, t0: a, t1: a + n })],
      ["template.npy", template],
    ]);
    return { type: "gb" as const, files, a, b: a + n };
  };
  /** Frames per second of `workers` running side by side (each its own frames): the sum of their own
   * rates, which a pass with its chunks handed to idle workers gets; and each one's rate. */
  const run = async (workers: number, n = F) => {
    const rates = new Array<number>(workers).fill(0);
    await pool.run("gb", workers, (i) => task(i, n), {
      deal: true,
      onResult: (i, r) => {
        rates[i] = r.frames / (r.ms / 1e3);
      },
    });
    return { fps: rates.reduce((a, b) => a + b, 0), rates };
  };
  await turnGpu(pool, N, test);
  await run(N, 8); // warm up: the shaders compile, the engines' plans are made
  let [gpuAll, gpuOne, cpu] = [0, 0, 0];
  for (let k = 0; k < SPEED_REPEATS; k++) {
    await turnGpu(pool, N, test);
    gpuAll = Math.max(gpuAll, (await run(N)).fps);
    await turnGpu(pool, 1, test);
    gpuOne = Math.max(gpuOne, (await run(1)).fps);
    await turnGpu(pool, 0, test);
    cpu = Math.max(cpu, (await run(N)).fps);
  }
  let both = 0;
  let bothGpu = 0;
  let bothRates: number[] = [];
  if (N > 1) {
    bothGpu = splitWorkers(N, gpuAll, gpuOne);
    await turnGpu(pool, bothGpu, test);
    for (let k = 0; k < SPEED_REPEATS; k++) {
      const r = await run(N);
      if (r.fps > both) [both, bothRates] = [r.fps, r.rates];
    }
  }
  await turnGpu(pool, 0, test);
  return {
    key,
    when: Date.now(),
    gpu,
    gpuReason: "",
    workers: N,
    cores: navigator.hardwareConcurrency || 0,
    H,
    W,
    fps: { cpu, gpu: gpuAll, both },
    bothGpu,
    bothRates,
    seconds: (performance.now() - began) / 1e3,
  };
}

/** The cached speed test for this machine and frame size, or a new one (cached). */
export async function cachedSpeedTest(
  pool: Pool,
  segs: Segment[],
  params: Uint8Array,
  T: number,
  H: number,
  W: number,
  gpu: string,
  test: string,
  force = false,
): Promise<{ test: SpeedTest; cached: boolean }> {
  const key = speedKey(gpu, navigator.hardwareConcurrency || 0, H, W, pool.size);
  const old =
    force || test
      ? undefined
      : await cache()
          .get<SpeedTest>(key)
          .catch(() => undefined);
  if (old) return { test: old, cached: true };
  const t = await speedTest(pool, segs, params, T, H, W, gpu, key, test);
  if (!test)
    await cache()
      .set(key, t)
      .catch(() => undefined);
  return { test: t, cached: false };
}

/** Decide the run's device and set the pool's GPUs. `params`: the session's pass-1 `params.json`.
 * Returns the report and, for the split, the workers' rates to size their chunks of a dealt pass 1. */
export async function setupDevice(
  pool: Pool | undefined,
  segs: Segment[],
  params: Uint8Array,
  T: number,
  H: number,
  W: number,
  asked: Device,
  test: string,
  log: (what: string) => void,
): Promise<{ report: DeviceReport; rates?: number[] }> {
  const began = performance.now();
  const report: DeviceReport = {
    asked,
    used: "cpu",
    gpu: "",
    why: "",
    self_test: "",
    workers: pool?.size ?? 1,
    gpu_workers: 0,
    speed_test: null,
    speed_test_cached: false,
    seconds: 0,
    gpu_frames: {},
    fallbacks: [],
  };
  const done = () => {
    report.seconds = (performance.now() - began) / 1e3;
    const where =
      report.used === "cpu"
        ? `CPU on ${report.workers} frame ${report.workers === 1 ? "worker" : "workers"}`
        : `GPU (${report.gpu}) on ${report.gpu_workers} of ${report.workers} frame workers`;
    log(`device: ${where} (${asked}: ${report.why})`);
    return report;
  };
  if (asked === "cpu" || !pool) {
    report.why = asked === "cpu" ? "CPU asked for" : "no frame workers";
    return { report: done() };
  }
  pool.onNote = (message) => {
    report.fallbacks.push(message);
    log(message);
  };
  const gt = await selfTest(pool, params, T, test);
  report.gpu = gt.name;
  report.self_test = gt.failure || "pass";
  let speed: SpeedTest | null = null;
  if (!gt.failure && (asked === "auto" || asked === "both")) {
    const s = await cachedSpeedTest(pool, segs, params, T, H, W, gt.name, test);
    speed = s.test;
    report.speed_test_cached = s.cached;
  }
  report.speed_test = speed;
  const c = decide(asked, pool.size, gt.failure, speed);
  report.used = c.use;
  report.why = c.why;
  const on = await turnGpu(pool, c.gpuWorkers, test);
  report.fallbacks.push(...on.off);
  report.gpu_workers = c.gpuWorkers - on.off.length;
  if (report.gpu_workers <= 0 && c.use !== "cpu") {
    report.used = "cpu";
    report.why = `the GPU would not open (${on.off.join("; ")})`;
  }
  const rates = c.use === "both" ? chunkWeights(pool.size, speed?.bothRates) : undefined;
  return { report: done(), rates };
}
