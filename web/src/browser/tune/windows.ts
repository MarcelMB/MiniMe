import type { FromTuneWorker, Summary, ToTuneWorker, TrialOut, WindowLoad } from "../../stream/tuneProtocol";
import type { Checks } from "../../types";

export const TUNE_BUDGET_BYTES = 1.2 * 2 ** 30;
export const WORKING_SET_FACTOR = 3;
export type SessionTrial = {
  counts: Record<string, number>;
  score: Summary | null;
  error: string | null;
  size10Median: number | null;
  seconds: number;
  perWindow: { i: number; out: TrialOut }[];
};
/** Per window, from the reference trial. */
export type Caps = { seeds_pnr: number; seeds_merged: number } | null;
export type WindowUnits = { i: number; t0: number; t1: number; bytes: Uint8Array };
type Asked =
  Extract<ToTuneWorker, { id: number }> extends infer M ? (M extends { id: number } ? Omit<M, "id"> : never) : never;
type Slot = {
  w: Worker;
  windows: number[];
  pending: Map<number, { ok: (m: FromTuneWorker) => void; no: (e: Error) => void }>;
  ack: (() => void) | null;
  fail: ((e: Error) => void) | null;
};

const median = (v: number[]): number | null => {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
};

export class TuneWorkers {
  private slots: Slot[] = [];
  private nextId = 1;
  private caps = new Map<number, { seeds_pnr: number; seeds_merged: number }>();
  heapPeakMb = 0;

  constructor(
    private makeWorker: () => Worker,
    private runSession: Uint8Array,
    private maxWorkers: number,
  ) {}

  /** `min(maxWorkers, windows, floor(budget / (WORKING_SET_FACTOR * perWindowBytes)))`, at least 1. */
  static workerCount(windows: number, perWindowBytes: number, maxWorkers: number, budget = TUNE_BUDGET_BYTES): number {
    return Math.max(
      1,
      Math.min(maxWorkers, windows, Math.floor(budget / (WORKING_SET_FACTOR * Math.max(1, perWindowBytes)))),
    );
  }

  private slot(): Slot {
    const w = this.makeWorker();
    const s: Slot = { w, windows: [], pending: new Map(), ack: null, fail: null };
    w.onmessage = (e: MessageEvent<FromTuneWorker>) => {
      const m = e.data;
      if (m.type === "ack") {
        this.heapPeakMb = Math.max(this.heapPeakMb, m.heapMb);
        s.ack?.();
        return;
      }
      if (m.type === "error") {
        const p = s.pending.get(m.id);
        if (p) p.no(new Error(m.message));
        else s.fail?.(new Error(m.message));
        s.pending.delete(m.id);
        return;
      }
      if ("heapMb" in m) this.heapPeakMb = Math.max(this.heapPeakMb, m.heapMb);
      s.pending.get(m.id)?.ok(m);
      s.pending.delete(m.id);
    };
    w.onerror = (e: ErrorEvent) => s.fail?.(new Error(e.message || "a tune worker failed"));
    return s;
  }

  async load(windows: WindowLoad[], perWindowBytes: number): Promise<void> {
    const n = TuneWorkers.workerCount(windows.length, perWindowBytes, this.maxWorkers);
    while (this.slots.length < n) this.slots.push(this.slot());
    while (this.slots.length > n) this.slots.pop()?.w.terminate();
    this.caps.clear();
    await Promise.all(
      this.slots.map((s, j) => {
        const mine = windows.filter((_, k) => k % n === j);
        s.windows = mine.map((w) => w.i);
        return new Promise<void>((ok, no) => {
          s.ack = ok;
          s.fail = no;
          s.w.postMessage({
            type: "load",
            runSession: this.runSession,
            windows: mine,
          } satisfies ToTuneWorker);
        });
      }),
    );
  }

  private ask<K extends FromTuneWorker["type"]>(
    s: Slot,
    msg: Asked,
    kind: K,
  ): Promise<Extract<FromTuneWorker, { type: K }>> {
    const id = this.nextId++;
    return new Promise((ok, no) => {
      s.pending.set(id, {
        ok: (m) =>
          m.type === kind
            ? ok(m as Extract<FromTuneWorker, { type: K }>)
            : no(new Error(`a ${m.type} reply to a ${kind} request`)),
        no,
      });
      s.w.postMessage({ ...msg, id } as ToTuneWorker);
    });
  }

  async trial(session: Uint8Array, checks: Checks, reference: boolean): Promise<SessionTrial> {
    const caps = reference || !this.caps.size ? "" : JSON.stringify(Object.fromEntries(this.caps));
    const began = performance.now();
    const replies = await Promise.all(
      this.slots.map((s) =>
        this.ask(s, { type: "trial", session, checks: JSON.stringify(checks), reference, caps }, "trial"),
      ),
    );
    const perWindow = replies.flatMap((r) => r.results).sort((a, b) => a.i - b.i);
    if (reference) {
      for (const { i, out } of perWindow) {
        this.caps.set(i, {
          seeds_pnr: out.counts.seeds_pnr ?? 0,
          seeds_merged: out.counts.seeds_merged ?? 0,
        });
      }
    }
    const counts: Record<string, number> = {};
    const score: Summary = {
      good: 0,
      flagged: 0,
      lost: 0,
      score: 0,
      low_pnr: 0,
      in_pieces: 0,
      bad_size: 0,
      duplicate: 0,
    };
    const sizes: number[] = [];
    let error: string | null = null;
    for (const { out } of perWindow) {
      for (const [k, v] of Object.entries(out.counts)) counts[k] = (counts[k] ?? 0) + v;
      if (out.error) error ??= out.error;
      if (out.score) for (const k of Object.keys(score) as (keyof Summary)[]) score[k] += out.score[k];
      sizes.push(...out.size10);
    }
    return {
      counts,
      score: error ? null : score,
      error,
      size10Median: error ? null : median(sizes),
      seconds: (performance.now() - began) / 1e3,
      perWindow,
    };
  }

  async units(): Promise<WindowUnits[]> {
    const replies = await Promise.all(this.slots.map((s) => this.ask(s, { type: "units" }, "units")));
    return replies.flatMap((r) => r.windows).sort((a, b) => a.i - b.i);
  }

  async contrastHist(seedsWnd: number, seedsStp: number, seedsMaxWnd: number, nWindows: number): Promise<number[]> {
    const loaded = this.slots.reduce((n, s) => n + s.windows.length, 0);
    const per = Math.max(1, Math.round(nWindows / Math.max(1, loaded)));
    const replies = await Promise.all(
      this.slots.map((s) =>
        this.ask(
          s,
          {
            type: "hist",
            seeds_wnd: seedsWnd,
            seeds_stp: seedsStp,
            seeds_max_wnd: seedsMaxWnd,
            n_windows: per,
          },
          "hist",
        ),
      ),
    );
    const hist: number[] = [];
    for (const r of replies) for (let c = 0; c < r.hist.length; c++) hist[c] = (hist[c] ?? 0) + r.hist[c];
    return hist;
  }

  close(): void {
    for (const s of this.slots) s.w.terminate();
    this.slots = [];
  }
}
