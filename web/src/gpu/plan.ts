// Which device a run uses (`auto`, `cpu`, `gpu`, `both`), from the GPU self-test and the speed test,
// and what the speed panel estimates from them. Pure functions: the frame workers measure, this
// decides.

export type Device = "auto" | "cpu" | "gpu" | "both";
export const DEVICES: readonly Device[] = ["auto", "cpu", "gpu", "both"];

/** Bump when the kernels or the stages' GPU code change: a cached speed test is then run again. */
export const SPEED_TEST_VERSION = "gpu-path-1";

/** The speed test of one machine, browser GPU and frame size (cached in IndexedDB, `cache.ts`). */
export type SpeedTest = {
  key: string;
  /** When it ran (ms since the epoch). */
  when: number;
  /** The GPU's name, or why there is none to use (no WebGPU, a software adapter, the self-test). */
  gpu: string;
  gpuReason: string;
  workers: number;
  cores: number;
  H: number;
  W: number;
  /** Pass-1 frames per second with every frame worker on the CPU, every one on the GPU, and the
   * split (`bothGpu` workers on the GPU, the rest on the CPU); 0 where not measured. */
  fps: { cpu: number; gpu: number; both: number };
  bothGpu: number;
  /** Each worker's frames per second in the split run (the weights of its chunks). */
  bothRates: number[];
  /** The test's own seconds. */
  seconds: number;
};

export type Choice = {
  /** What the run uses. */
  use: "cpu" | "gpu" | "both";
  /** Frame workers on the GPU (the first ones). */
  gpuWorkers: number;
  why: string;
};

/** The device for a run asked to use `asked` on `workers` frame workers. `gpuReason` is why the GPU
 * cannot be used here (empty when it can); `test` the speed test (auto picks its fastest mode). */
export function decide(asked: Device, workers: number, gpuReason: string, test: SpeedTest | null): Choice {
  if (asked === "cpu") return { use: "cpu", gpuWorkers: 0, why: "CPU asked for" };
  if (gpuReason) return { use: "cpu", gpuWorkers: 0, why: gpuReason };
  const split = Math.min(workers - 1, Math.max(1, test?.bothGpu || Math.ceil(workers / 2)));
  if (asked === "gpu") return { use: "gpu", gpuWorkers: workers, why: "GPU asked for" };
  if (asked === "both")
    return workers > 1
      ? { use: "both", gpuWorkers: split, why: "GPU and CPU asked for" }
      : { use: "gpu", gpuWorkers: workers, why: "GPU and CPU asked for, but one worker: GPU" };
  if (!test) return { use: "cpu", gpuWorkers: 0, why: "no speed test" };
  const { cpu, gpu, both } = test.fps;
  const best = Math.max(cpu, gpu, workers > 1 ? both : 0);
  if (best <= 0 || best === cpu) return { use: "cpu", gpuWorkers: 0, why: "the CPU was fastest" };
  if (best === gpu) return { use: "gpu", gpuWorkers: workers, why: "the GPU was fastest" };
  return { use: "both", gpuWorkers: split, why: "GPU and CPU together were fastest" };
}

/** How many of `workers` frame workers the split puts on the GPU: as many as keep it busy, judged by
 * the GPU's throughput with all of them (`gpuAll`) against one worker's alone (`gpuOne`), and at least
 * one on each side. */
export function splitWorkers(workers: number, gpuAll: number, gpuOne: number): number {
  if (workers < 2 || gpuOne <= 0) return Math.max(1, workers - 1);
  return Math.min(workers - 1, Math.max(1, Math.ceil(gpuAll / gpuOne)));
}

/** Share of a CPU run's time in the per-frame stages the device changes (template, pass 1 with the
 * shift estimate, the streaming pass's preprocessing), and pass 1's share: measured on the 5-min synth
 * with 8 workers on an M1 Max (28.6 s: pass 1 ~12 s, template ~1 s, streaming preprocessing ~2 s of
 * the 5.2 s pass). A guide for the estimate only. */
const DEVICE_SHARE = 0.52;
const PASS1_SHARE = 0.42;

/** About how long a session of `frames` frames takes in each mode, from the measured pass-1 rates:
 * a CPU run is pass 1's time over its share; the device changes only `DEVICE_SHARE` of it. Seconds,
 * 0 where the mode was not measured. */
export function estimateSeconds(frames: number, fps: SpeedTest["fps"]): SpeedTest["fps"] {
  if (fps.cpu <= 0) return { cpu: 0, gpu: 0, both: 0 };
  const cpu = frames / fps.cpu / PASS1_SHARE;
  const of = (f: number) => (f > 0 ? cpu * (1 - DEVICE_SHARE) + cpu * DEVICE_SHARE * (fps.cpu / f) : 0);
  return { cpu, gpu: of(fps.gpu), both: of(fps.both) };
}

/** Weights of a pooled pass 1's chunks when it deals one chunk per worker: equal, unless the workers
 * run at different rates (the split), then in proportion to them. */
export function chunkWeights(workers: number, rates: number[] | undefined): number[] | undefined {
  if (!rates || rates.length !== workers || rates.some((r) => !(r > 0))) return undefined;
  const lo = Math.min(...rates);
  const hi = Math.max(...rates);
  return hi / lo < 1.15 ? undefined : rates;
}

/** The cache key of a speed test. */
export function speedKey(gpu: string, cores: number, H: number, W: number, workers: number): string {
  return [SPEED_TEST_VERSION, gpu || "no-gpu", `cores${cores}`, `${H}x${W}`, `w${workers}`].join("|");
}
