import type { Summary } from "../../stream/tuneProtocol";
import type { Params } from "./params";

/** Every SEARCHED key. */
export type Values = Record<string, number | string>;
export type Score = Summary;
export type Trial = {
  step: string;
  values: Values;
  counts: Record<string, number>;
  score: Score | null;
  size10_median: number | null;
  seconds: number;
  error: string | null;
};
/** The values tried per key (LocalSearch: start + each line's steps). */
export type Grids = Record<string, (number | string)[]>;
/** What the search runs settings on; TuneWorkers behind it in the app, a function in the tests. */
export interface TrialRunner {
  /** Run `params` (the full flat parameters) and score; `reference` sets the good units later trials are charged for losing. */
  trial(
    params: Params,
    reference: boolean,
  ): Promise<{
    counts: Record<string, number>;
    score: Score | null;
    error: string | null;
    size10Median: number | null;
    seconds: number;
  }>;
  /** Switch the windows the trials run on ("segment": the explored middle, "all": the session). */
  use(set: "segment" | "all"): Promise<void>;
  /** `start_diff_thres`: the knee of the seed-contrast histogram of the explored windows, or null. */
  startDiffThres(params: Params): Promise<number | null>;
}
