// The New run page's speed test (`SpeedPanel.tsx`): a worker runs the GPU self-test and the speed test
// on the chosen videos (`ToWorker.speedTest`, answered with `speed`); the result is cached per machine
// in IndexedDB (`stream/device.ts` writes it, this reads it back).

import { gpuName, openAdapter } from "../gpu/frames";
import { type SpeedTest, speedKey } from "../gpu/plan";
import type { FromWorker, SpeedAnswer, ToWorker } from "../stream/protocol";
import { idbKV } from "./kv";
import { RUN_LOCK } from "./support";

/** This page's GPU: its name, or why there is none to use. */
export async function pageGpu(): Promise<{ name: string; reason: string }> {
  const a = await openAdapter();
  return "reason" in a ? { name: "", reason: a.reason } : { name: gpuName(a.info), reason: "" };
}

/** The cached speed test of this machine for frames of `H` x `W` on `workers` frame workers. */
export async function cachedTest(gpu: string, H: number, W: number, workers: number): Promise<SpeedTest | null> {
  const key = speedKey(gpu, navigator.hardwareConcurrency || 0, H, W, workers);
  return (
    (await idbKV("speed")
      .get<SpeedTest>(key)
      .catch(() => undefined)) ?? null
  );
}

/** Whether a run holds the run lock (a speed test then waits for it: it would slow the run down). */
export async function runGoing(): Promise<boolean> {
  const q = await navigator.locks?.query().catch(() => null);
  return Boolean(q?.held?.some((l) => l.name === RUN_LOCK));
}

/** Run the self-test and speed test on `avis` in a worker of its own (`force`: measure again). */
export function runSpeedTest(avis: File[], params: Record<string, unknown>, force: boolean): Promise<SpeedAnswer> {
  const w = new Worker(new URL("../stream/stream.worker.ts", import.meta.url), { type: "module" });
  return new Promise<SpeedAnswer>((resolve, reject) => {
    let answer: SpeedAnswer | null = null;
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "speed") {
        const { type: _, ...a } = m;
        answer = a;
      } else if (m.type === "done") answer ? resolve(answer) : reject(new Error("the speed test gave no answer"));
      else if (m.type === "error") reject(new Error(m.message));
    };
    w.onerror = (e) => reject(new Error(e.message || "the speed test's worker did not start"));
    const msg: ToWorker = { avis, inputs: [], params, speedTest: { force } };
    w.postMessage(msg);
  }).finally(() => w.terminate());
}
