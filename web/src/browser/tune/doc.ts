/** `sweep.csv` and `autotune.json` of an auto-tune (`autotune.run.sweep_table`, `best_per_value`). */
import type { CurvePoint, Proposal } from "../../types";
import { pyFloat } from "../num";
import { SEARCHED } from "./params";
import { STEP_OF } from "./search";
import type { Grids, Trial, Values } from "./types";

const SCORE_COLS = ["good", "flagged", "lost", "score", "low_pnr", "in_pieces", "bad_size", "duplicate"] as const;
const COUNT_COLS = ["seeds_init", "seeds_pnr", "seeds_merged", "units_init_merge", "units_final"] as const;

export type PairCounts = { n_near: number; n_near_above: number; n_closer: number; far_p99: number | null };
export type AutotuneDoc = Omit<Proposal, "curves"> & {
  strategy: "local";
  passes: number;
  start: Values;
  pairs: { run: PairCounts; proposal: PairCounts };
  grids: Grids;
  wasm_heap_peak_mb: number;
};

const cell = (v: unknown): string =>
  v === null || v === undefined ? "" : typeof v === "number" ? pyFloat(v).replace(/\.0$/, "") : String(v);

/** `autotune.run.sweep_table` as pandas writes it (`index=False`; integers without `.0`). */
export function sweepCsv(trials: Trial[]): string {
  const head = ["step", ...SEARCHED, ...SCORE_COLS, ...COUNT_COLS, "size10_median", "seconds", "error"];
  const rows = trials.map((t) => [
    t.step,
    ...SEARCHED.map((k) => cell(t.values[k])),
    ...SCORE_COLS.map((k) => cell(t.score?.[k] ?? null)),
    ...COUNT_COLS.map((k) => cell(t.counts[k] ?? null)),
    cell(t.size10_median),
    cell(Math.round(t.seconds * 100) / 100),
    t.error ?? "",
  ]);
  return `${[head, ...rows].map((r) => r.join(",")).join("\n")}\n`;
}

const isNum = (v: unknown) =>
  typeof v === "number" || (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v)));
const same = (a: unknown, b: unknown) => (isNum(a) && isNum(b) ? Number(a) === Number(b) : String(a) === String(b));

/** Per value of `key` in its step: the best-scoring successful trial, "auto" first then numbers ascending. */
export function bestPerValue(trials: Trial[], key: string): CurvePoint[] {
  const best: Trial[] = [];
  for (const t of trials) {
    if (t.step !== STEP_OF[key] || t.score === null) continue;
    const i = best.findIndex((b) => same(b.values[key], t.values[key]));
    if (i < 0) best.push(t);
    else if (t.score.score > (best[i].score?.score ?? 0)) best[i] = t;
  }
  const words = best.filter((t) => !isNum(t.values[key]));
  const nums = best.filter((t) => isNum(t.values[key])).sort((a, b) => Number(a.values[key]) - Number(b.values[key]));
  return [...words, ...nums].map((t) => ({
    value: t.values[key],
    good: t.score?.good ?? 0,
    flagged: t.score?.flagged ?? 0,
  }));
}

/** True if a numeric `value` is the lowest or highest number of a grid with more than one number. */
export function atEdge(value: unknown, grid: (number | string)[]): boolean {
  const nums = grid.filter((v): v is number => typeof v === "number").sort((a, b) => a - b);
  return typeof value === "number" && nums.length > 1 && (value === nums[0] || value === nums[nums.length - 1]);
}
