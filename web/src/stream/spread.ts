// The spread motion template (`template_source = "spread"`) and the fill image (`motion_fill =
// "template"`): `MotionEngine` holds the template's state (`motion::SpreadBuilder`), workers do the
// per-frame work (`SpreadWorker`). The template's frames are runs of consecutive session frames spread
// over the session; each stage reads them once: the runs' means, the frames' shifts against the
// current template (twice), the aligned means (twice), then the fill sums against the final template.
// Prepared frames are added in frame order (float32 sums); shifts and fill sums in any order.

import { forFrames, type Segment } from "./frames";
import type { MotionEngine, SpreadWorker } from "./pkg/minime_core.js";
import type { Pool } from "./pool";
import type { SpreadOp, UResult } from "./poolProtocol";

/** A spread task: session frames `[a, b)` (one run or part of one) of `op`, `shifts` for `warp`. */
export type SpreadTask = { op: SpreadOp; a: number; b: number; shifts?: Float64Array };

/** Run one spread task on `sw` (the frame workers' `u1`, and the run's worker without a pool). */
export function spreadFrames(sw: SpreadWorker, segs: Segment[], t: SpreadTask): UResult {
  const preps: Float32Array[] = [];
  const est: number[] = [];
  let i = 0;
  forFrames(
    segs,
    t.a,
    t.b,
    (s) => sw.set_stride(s),
    (c) => {
      if (t.op === "prep") preps.push(sw.prep(c));
      else if (t.op === "est") est.push(...sw.estimate(c));
      else if (t.op === "warp") {
        const sh = t.shifts as Float64Array;
        preps.push(sw.prep_warped(c, sh[2 * i], sh[2 * i + 1]));
      } else sw.fill_add(c);
      i++;
    },
  );
  if (i !== t.b - t.a) throw new Error(`spread template: read ${i} of frames ${t.a}-${t.b}`);
  if (t.op === "est") return { est: Float64Array.from(est) };
  if (t.op === "fill") return { fill: sw.take_fill() };
  const all = new Float32Array(preps.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of preps) {
    all.set(p, at);
    at += p.length;
  }
  return { preps: all };
}

/** Who runs the spread tasks: the pool's workers, or one `SpreadWorker` in this thread. */
export type SpreadRunner = {
  /** Every worker aligns to this template from now on. */
  setTemplate: (image: Float32Array, search: number) => Promise<void>;
  /** Run `tasks`, giving each result to `onResult` (in task order with `ordered`). */
  run: (tasks: SpreadTask[], ordered: boolean, onResult: (i: number, r: UResult) => void) => Promise<void>;
};

/** Frames per task: about 0.1 s of a worker's time. */
const TASK = 5;

/** The template frames (`engine.template_frames()`, runs of consecutive session frames) cut into tasks
 * that stay inside one run: `[index of its first template frame, task]`. */
function cut(frames: Uint32Array, op: SpreadOp, warps?: Float64Array): [number, SpreadTask][] {
  const out: [number, SpreadTask][] = [];
  let i = 0;
  while (i < frames.length) {
    let j = i + 1;
    while (j < frames.length && j - i < TASK && frames[j] === frames[j - 1] + 1) j++;
    const shifts = warps ? warps.slice(2 * i, 2 * j) : undefined;
    out.push([i, { op, a: frames[i], b: frames[i] + (j - i), shifts }]);
    i = j;
  }
  return out;
}

/** The spread template (and with `motion_fill = "template"` the fill sums) of `engine`, its stages run
 * by `runner`; `tick(n)` after every `n` frames. Leaves `engine` ready for `build_template` (a
 * first-frames template's fill is `fillTemplate`'s). */
export async function spreadTemplate(
  engine: MotionEngine,
  runner: SpreadRunner,
  tick: (n: number) => void,
): Promise<void> {
  const frames = engine.template_frames();
  for (let stage = engine.spread_stage(); stage !== 4; stage = engine.spread_stage()) {
    if (stage === 0 || stage === 2) {
      const warps = stage === 2 ? Float64Array.from(engine.spread_warps()) : undefined;
      const tasks = cut(frames, stage === 0 ? "prep" : "warp", warps);
      await runner.run(
        tasks.map(([, t]) => t),
        true,
        (k, r) => {
          const [i0, t] = tasks[k];
          const preps = r.preps as Float32Array;
          const len = preps.length / (t.b - t.a);
          for (let j = 0; j < t.b - t.a; j++) engine.spread_add_prepped(i0 + j, preps.subarray(j * len, (j + 1) * len));
          tick(t.b - t.a);
        },
      );
    } else if (stage === 1) {
      await runner.setTemplate(Float32Array.from(engine.spread_image()), engine.spread_search());
      const tasks = cut(frames, "est");
      await runner.run(
        tasks.map(([, t]) => t),
        false,
        (k, r) => {
          const [i0, t] = tasks[k];
          engine.spread_set_estimates(i0, r.est as Float64Array);
          tick(t.b - t.a);
        },
      );
    } else await fillTemplate(engine, runner, tick);
  }
}

/** The fill sums of the template's frames against the built template (`engine.fill_pending()`). */
export async function fillTemplate(
  engine: MotionEngine,
  runner: SpreadRunner,
  tick: (n: number) => void,
): Promise<void> {
  if (!engine.fill_pending()) return;
  await runner.setTemplate(Float32Array.from(engine.spread_image()), engine.spread_search());
  const tasks = cut(engine.template_frames(), "fill");
  await runner.run(
    tasks.map(([, t]) => t),
    false,
    (k, r) => {
      engine.add_fill(r.fill as Uint32Array);
      tick(tasks[k][1].b - tasks[k][1].a);
    },
  );
}

/** The tasks on one `SpreadWorker` in this thread (a run without a pool). */
export function localRunner(sw: SpreadWorker, segs: Segment[]): SpreadRunner {
  return {
    setTemplate: async (image, search) => sw.set_template(image, search),
    run: async (tasks, _ordered, onResult) => {
      tasks.forEach((t, i) => {
        onResult(i, spreadFrames(sw, segs, t));
      });
    },
  };
}

/** The tasks on the pool's workers (`u0` must have been broadcast). */
export function poolRunner(pool: Pool): SpreadRunner {
  return {
    setTemplate: (image, search) => pool.broadcast({ type: "utmpl", image, search }),
    run: (tasks, ordered, onResult) =>
      pool.run("u1", tasks.length, (i) => ({ type: "u1", ...tasks[i] }), {
        ordered,
        onResult: (i, r) => onResult(i, r),
      }),
  };
}
