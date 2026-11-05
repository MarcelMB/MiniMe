/** `autotune.sweep.LocalSearch`: coordinate search from a data-driven start, one setting at a
 * time through a `TrialRunner`. Names and constants follow the Python module. */
import { DEFAULTS } from "../../stream/session";
import { CORR_KEYS, norm, type Params, SEARCHED, TUNE_FRAMES } from "./params";
import type { Grids, Trial, TrialRunner, Values } from "./types";

export type { Grids, Score, Trial, TrialRunner, Values } from "./types";

export const LOCAL_ORDER = [
  "sparse_snr",
  "seeds_diff_thres",
  "pnr_thres",
  "merge_corr",
  "inita_corr",
  "loop_merge_corr",
] as const;
export const LOCAL_SCALE = 1.4;
export const LOCAL_CORR = 0.05;
export const LOCAL_PASSES = 1;
export const LOCAL_MAX_STEPS = 5;
export const START_PNR = 2.0;
export const KNEE_RATIO = 1.5;
export const KNEE_WINDOWS = 16;
export const CORR_RANGE: [number, number] = [0.5, 0.98];
export const STEP_OF: Record<string, string> = {
  seeds_diff_thres: "seeds",
  pnr_thres: "seeds",
  merge_corr: "merge",
  init_merge_corr: "merge",
  inita_corr: "inita",
  sparse_snr: "sparse",
  loop_merge_corr: "loop_merge",
};
export { TUNE_FRAMES };

/** Python's `float(f"{v:.3g}")`. */
export function round3g(v: number): number {
  return Number(v.toPrecision(3));
}

function roundValue(v: number, integer: boolean): number {
  return integer ? Math.max(1, Math.round(v)) : round3g(v);
}

/** `value` one step up or down (null for a string or when clipping leaves it as is). */
export function stepValue(key: string, value: number | string, up: boolean, integer = false): number | string | null {
  if (typeof value === "string") return null;
  let next: number;
  if (CORR_KEYS.includes(key)) {
    const [lo, hi] = CORR_RANGE;
    next = Math.round(Math.min(hi, Math.max(lo, value + (up ? LOCAL_CORR : -LOCAL_CORR))) * 1000) / 1000;
  } else {
    next = roundValue(up ? value * LOCAL_SCALE : value / LOCAL_SCALE, integer);
    if (integer && next === value) next = Math.max(1, value + (up ? 1 : -1)); // small integers: at least one grey level
  }
  return next === value ? null : next;
}

/** Upper bound on the settings `run` tries: the run, two starts, each pass's lines (one step back
 * plus `maxSteps` on), the decide reruns (the run's and the best). */
export function localNTrials(maxSteps = LOCAL_MAX_STEPS, passes = LOCAL_PASSES, nDecide = 1): number {
  return 3 + passes * LOCAL_ORDER.length * (maxSteps + 1) + 1 + nDecide;
}

const nChanged = (values: Values, base: Params) => SEARCHED.filter((k) => norm(values[k]) !== norm(base[k])).length;
const key = (v: Values) => JSON.stringify(SEARCHED.map((k) => [k, String(v[k])]));

/** Successful trials, best first (score, then fewer changes, then order); distinct values. */
export function ranked(trials: Trial[], base: Params): Trial[] {
  const ok = trials.map((t, i) => [i, t] as const).filter(([, t]) => t.score !== null);
  ok.sort(
    ([i, a], [j, b]) =>
      (b.score?.score ?? 0) - (a.score?.score ?? 0) || nChanged(a.values, base) - nChanged(b.values, base) || i - j,
  );
  const seen = new Set<string>();
  const out: Trial[] = [];
  for (const [, t] of ok) {
    const k = key(t.values);
    if (!seen.has(k)) {
      seen.add(k);
      out.push(t);
    }
  }
  return out;
}

export const better = (t: Trial, than: Trial): boolean =>
  t.score !== null && (than.score === null || t.score.score > than.score.score);

/** The windows whose frames cover about `tuneFrames`, centred in the session (`middle_segment` on
 * window edges); null when `tuneFrames` is 0 or the session is not longer than it. */
export function middleWindows(
  windows: { t0: number; t1: number }[],
  tuneFrames: number,
): { from: number; to: number; segment: [number, number] } | null {
  const T = windows[windows.length - 1].t1;
  if (tuneFrames < 0) throw new Error(`tune_frames must be >= 0, got ${tuneFrames}`);
  if (tuneFrames === 0 || tuneFrames >= T) return null;
  const edges = [windows[0].t0, ...windows.map((w) => w.t1)];
  const nearest = (x: number) =>
    edges.reduce((best, e, i) => (Math.abs(e - x) < Math.abs(edges[best] - x) ? i : best), 0);
  const from = nearest(Math.floor((T - tuneFrames) / 2));
  const to = nearest(edges[from] + tuneFrames);
  return to > from ? { from, to, segment: [edges[from], edges[to]] } : null;
}

export type SearchOptions = {
  base: Params;
  runParams: Params;
  runValues: Values;
  integerDiff: boolean;
  passes?: number;
  maxSteps?: number;
  segment: boolean;
  nDecide?: number;
  onTrial?: (t: Trial) => void;
};

/** Strings first, then numbers ascending, each once (`sweep._unique`). */
export function unique(values: (number | string)[]): (number | string)[] {
  const strs = [...new Set(values.filter((v): v is string => typeof v === "string"))];
  const nums = [...new Set(values.filter((v): v is number => typeof v === "number"))].sort((a, b) => a - b);
  return [...strs, ...nums];
}

/** The search's own failures (Python's `RuntimeError`); anything else is a crash and propagates. */
export class SearchFailed extends Error {}

export class LocalSearch {
  trials: Trial[] = [];
  runTrial: Trial | null = null;
  final: Trial | null = null;
  betterThanRun = false;
  searchError: string | null = null;
  start: Values = {};
  grids: Grids = {};
  readonly passes: number;
  private maxSteps: number;
  private nDecide: number;

  constructor(
    private runner: TrialRunner,
    private o: SearchOptions,
  ) {
    this.passes = o.passes ?? LOCAL_PASSES;
    this.maxSteps = o.maxSteps ?? LOCAL_MAX_STEPS;
    this.nDecide = o.nDecide ?? 1;
    if (this.passes < 1) throw new Error(`passes must be >= 1, got ${this.passes}`);
  }

  nTrials(): number {
    return localNTrials(this.maxSteps, this.passes, this.nDecide);
  }

  private params(values: Values, base?: Params): Params {
    return { ...(base ?? this.o.base), ...values };
  }

  private async try(step: string, values: Values, base?: Params, reference = false): Promise<Trial> {
    const r = await this.runner.trial(this.params(values, base), reference);
    const t: Trial = {
      step,
      values: { ...values },
      counts: r.counts,
      score: r.score,
      size10_median: r.size10Median,
      seconds: r.seconds,
      error: r.error,
    };
    this.trials.push(t);
    this.o.onTrial?.(t);
    return t;
  }

  async startValues(runValues: Values): Promise<Values> {
    const v = { ...runValues };
    const run = this.o.runParams;
    if (norm(run.seeds_diff_thres) === norm(DEFAULTS.seeds_diff_thres)) {
      const knee = await this.runner.startDiffThres(this.o.base);
      if (knee !== null) v.seeds_diff_thres = knee;
    }
    if (norm(run.pnr_thres) === norm(DEFAULTS.pnr_thres)) v.pnr_thres = START_PNR;
    return v;
  }

  /** Best trial along `key` from `cur`: one step down, on down while the score does not drop (a tie
   * is crossed, only a gain is taken); else the same upward. */
  private async line(k: string, cur: Trial, tried: (number | string)[], integer: boolean): Promise<Trial> {
    const base = cur;
    for (const up of [false, true]) {
      let v = base.values;
      for (let i = 0; i < this.maxSteps; i++) {
        const next = stepValue(k, v[k], up, integer);
        if (next === null) break;
        tried.push(next);
        const t = await this.try(STEP_OF[k], { ...v, [k]: next });
        if (t.score === null || t.score.score < (cur.score?.score ?? Number.NEGATIVE_INFINITY)) break;
        v = t.values;
        if (better(t, cur)) cur = t;
      }
      if (cur !== base) break; // improved going down: no step up
    }
    return cur;
  }

  private async search(runValues: Values): Promise<Trial[]> {
    const integer = this.o.integerDiff;
    this.start = await this.startValues(runValues);
    let cur = await this.try("start", this.start);
    // the run's values when they score better: the data-driven start is a guess
    if (
      key(this.start) !== key(runValues) &&
      (cur.score === null ||
        this.runTrial?.score === null ||
        (this.runTrial?.score?.score ?? 0) >= (cur.score?.score ?? 0))
    ) {
      const other = await this.try("start", runValues);
      if (better(other, cur)) cur = other;
    }
    if (cur.score === null) throw new SearchFailed(`auto-tune: the start setting failed (${cur.error})`);
    const tried: Record<string, (number | string)[]> = Object.fromEntries(SEARCHED.map((k) => [k, [cur.values[k]]]));
    for (let pass = 0; pass < this.passes; pass++) {
      const from = cur;
      for (const k of LOCAL_ORDER) cur = await this.line(k, cur, tried[k], integer && k === "seeds_diff_thres");
      if (cur === from) break; // a further pass only when the last moved a key
    }
    for (const k of SEARCHED) this.grids[k] = unique(tried[k]);
    return ranked(
      this.trials.filter((t) => t.step !== "run" && t.step !== "decide"),
      this.o.base,
    );
  }

  /** Rerun the run's values and the segment's best on the whole session; the run's becomes
   * `runTrial`. Returns the best of the others there (null if all failed). */
  private async decide(runValues: Values, top: Trial[]): Promise<Trial | null> {
    await this.runner.use("all");
    this.runTrial = null; // no caps: the session has more seeds than the segment
    const run = await this.try("decide", runValues, this.o.runParams, true);
    const others: Trial[] = [];
    for (const t of top.filter((t) => key(t.values) !== key(runValues)).slice(0, this.nDecide)) {
      others.push(await this.try("decide", t.values));
    }
    this.runTrial = run;
    const r = ranked(others, this.o.base);
    if (run.score === null && !r.length) {
      throw new SearchFailed(`auto-tune: the run's parameters failed on the whole session (${run.error})`);
    }
    return r[0] ?? null;
  }

  /** Every step; returns the proposed values (`final` is their trial). */
  async run(): Promise<Values> {
    const runValues = this.o.runValues;
    await this.runner.use(this.o.segment ? "segment" : "all");
    this.runTrial = await this.try("run", runValues, this.o.runParams, true);
    let top: Trial[] = [];
    try {
      top = await this.search(runValues);
    } catch (e) {
      if (!(e instanceof SearchFailed) || this.runTrial.score === null) throw e;
      this.searchError = e.message;
    }
    const found = this.o.segment ? await this.decide(runValues, top) : (top[0] ?? null);
    const run = this.runTrial as Trial;
    if (found !== null && (run.score === null || (found.score?.score ?? 0) > run.score.score)) {
      this.final = found;
      this.betterThanRun = true;
      return { ...found.values };
    }
    this.final = run;
    this.betterThanRun = false;
    return { ...runValues };
  }
}
