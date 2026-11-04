// A pool of frame workers (`frame.worker.ts`): several wasm instances, each in its own Web Worker,
// for the stages whose frames or tiles are independent. No SharedArrayBuffer: a worker gets its
// inputs in a message and returns its results in one, so the page needs no COOP/COEP headers.
//
// `run` hands numbered tasks to idle workers (dynamic: a slow worker takes fewer) and gives the
// results back, optionally in task order (`ordered`: the next result is held until every earlier one
// has been given out, and no more than `maxAhead` tasks run ahead of it, which bounds the memory
// of the results waiting).

import type { FromFrameWorker, ToFrameWorker } from "./poolProtocol";

type Task = Extract<ToFrameWorker, { id: number }>;
type TaskKind = "p1" | "s1" | "p2" | "t1" | "t2";
type Done<K extends TaskKind> = Extract<FromFrameWorker, { type: "done"; kind: K }>;
type Without<T, K extends string> = T extends unknown ? Omit<T, K> : never;

type Slot = {
  w: Worker;
  busy: boolean;
  heapMb: number;
  handler: ((m: FromFrameWorker) => void) | null;
  fail: ((e: Error) => void) | null;
};

export type RunOptions<K extends TaskKind> = {
  /** Give the results in task order. */
  ordered?: boolean;
  /** With `ordered`: tasks allowed to run ahead of the next result to give out (default 2 per worker). */
  maxAhead?: number;
  onResult: (i: number, r: Done<K>) => void;
  /** A running task's own count (frames done so far). */
  onProgress?: (i: number, done: number) => void;
  /** Deal every task to a worker up front (task `i` to worker `i % size`, which runs its tasks in
   * order) instead of handing one out whenever a worker is idle: the work then goes on while the
   * caller's thread is busy elsewhere and cannot hand anything out. */
  deal?: boolean;
};

/** Frame workers to use: `params.workers` when it is a whole number >= 1, else the machine's
 * cores less two (the run's own worker and the page), at most 8. */
export function workerCount(params: Record<string, unknown>): number {
  const asked = Number(params.workers);
  if (Number.isInteger(asked) && asked >= 1) return Math.min(asked, 32);
  return Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 2));
}

export class Pool {
  /** Milliseconds all tasks of each kind kept a worker busy, for the stage's utilisation. */
  readonly busyMs: Record<TaskKind, number> = { p1: 0, s1: 0, p2: 0, t1: 0, t2: 0 };
  private constructor(private readonly slots: Slot[]) {}

  /** `n` workers, each with the session's AVIs indexed; rejects (and stops them) when one fails to start in `timeoutMs`. */
  static async create(n: number, avis: File[], scratch: string, timeoutMs = 30_000): Promise<Pool> {
    const slots: Slot[] = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./frame.worker.ts", import.meta.url), { type: "module" });
      const s: Slot = { w, busy: false, heapMb: 0, handler: null, fail: null };
      w.onmessage = (e: MessageEvent<FromFrameWorker>) => s.handler?.(e.data);
      w.onerror = (e) => s.fail?.(new Error(`a frame worker failed: ${e.message || "could not start"}`));
      w.onmessageerror = () => s.fail?.(new Error("a frame worker sent a message that could not be read"));
      return s;
    });
    const pool = new Pool(slots);
    try {
      await pool.broadcast({ type: "init", avis, scratch }, timeoutMs);
    } catch (e) {
      pool.close();
      throw e;
    }
    return pool;
  }

  get size(): number {
    return this.slots.length;
  }

  /** Wasm heap of all workers, MB (as last reported). */
  heapMb(): number {
    return this.slots.reduce((n, s) => n + s.heapMb, 0);
  }

  /** Send a context message to every worker and wait for each one's `ack`. */
  broadcast(msg: Exclude<ToFrameWorker, { id: number }>, timeoutMs = 0): Promise<void> {
    return Promise.all(
      this.slots.map(
        (s) =>
          new Promise<void>((resolve, reject) => {
            const timer = timeoutMs
              ? setTimeout(() => reject(new Error("a frame worker did not start")), timeoutMs)
              : 0;
            const done = (f: () => void) => {
              clearTimeout(timer);
              s.handler = s.fail = null;
              f();
            };
            s.fail = (e) => done(() => reject(e));
            s.handler = (m) => {
              if (m.type === "error") done(() => reject(new Error(m.message)));
              else if (m.type === "ack") {
                s.heapMb = m.heapMb;
                done(resolve);
              }
            };
            s.w.postMessage(msg);
          }),
      ),
    ).then(() => undefined);
  }

  /** Run tasks `0..count` of `kind`; `make(i)` is task `i`'s message (without its `id`). Resolves
   * when every result has been given out; rejects on the first error (the caller closes the pool). */
  run<K extends TaskKind>(
    kind: K,
    count: number,
    make: (i: number) => Without<Extract<Task, { type: K }>, "id">,
    opts: RunOptions<K>,
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let next = 0;
      let given = 0;
      let over = false;
      const waiting = new Map<number, Done<K>>();
      const maxAhead = opts.maxAhead ?? 2 * this.slots.length;
      const fail = (e: Error) => {
        if (over) return;
        over = true;
        for (const s of this.slots) s.handler = s.fail = null;
        reject(e);
      };
      const pump = () => {
        if (opts.deal) return;
        for (const s of this.slots) {
          if (over || next >= count || (opts.ordered && next - given >= maxAhead)) return;
          if (s.busy) continue;
          s.busy = true;
          s.w.postMessage({ ...make(next), id: next });
          next++;
        }
      };
      const give = (i: number, r: Done<K>) => {
        this.busyMs[r.kind] += r.ms;
        opts.onResult(i, r);
        given++;
      };
      for (const s of this.slots) {
        s.fail = fail;
        s.handler = (m) => {
          try {
            if (m.type === "error") return fail(new Error(m.message));
            if (m.type === "progress") return opts.onProgress?.(m.id, m.done);
            if (m.type !== "done") return;
            if (m.kind !== kind) throw new Error(`a ${m.kind} result came back for a ${kind} task`);
            s.busy = false;
            s.heapMb = m.heapMb;
            const r = m as Done<K>;
            if (opts.ordered) {
              waiting.set(r.id, r);
              for (let r2 = waiting.get(given); r2; r2 = waiting.get(given)) {
                waiting.delete(given);
                give(r2.id, r2);
              }
            } else give(r.id, r);
            if (given === count && !over) {
              over = true;
              for (const t of this.slots) t.handler = t.fail = null;
              resolve();
            } else pump();
          } catch (e) {
            fail(e instanceof Error ? e : new Error(String(e)));
          }
        };
      }
      if (count === 0) return resolve();
      if (opts.deal) {
        for (let i = 0; i < count; i++) this.slots[i % this.slots.length].w.postMessage({ ...make(i), id: i });
        return;
      }
      pump();
    });
  }

  /** Stop every worker (their OPFS handles go with them). */
  close(): void {
    for (const s of this.slots) {
      s.handler = s.fail = null;
      s.w.terminate();
    }
  }
}
