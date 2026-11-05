import { describe, expect, it } from "vitest";
import { DEFAULTS } from "../../stream/session";
import { derivedParams, SEARCHED } from "./params";
import { LocalSearch, localNTrials, middleWindows, ranked, stepValue, type TrialRunner } from "./search";
import type { Trial } from "./types";

const pick = (p: Record<string, unknown>) => Object.fromEntries(SEARCHED.map((k) => [k, p[k] as number | string]));

/** A runner whose score is a smooth function of the values, so the search has something to find. */
function runner(
  opts: {
    fail?: (p: Record<string, unknown>) => string | null;
    knee?: number | null;
    peak?: Record<string, number>;
  } = {},
) {
  const peak = {
    sparse_snr: 0.5,
    seeds_diff_thres: 6,
    merge_corr: 0.85,
    inita_corr: 0.8,
    loop_merge_corr: 0.8,
    ...opts.peak,
  };
  const calls: { params: Record<string, unknown>; reference: boolean; set: string }[] = [];
  let set = "segment";
  const r: TrialRunner = {
    async trial(params, reference) {
      calls.push({ params, reference, set });
      const err = opts.fail?.(params) ?? null;
      if (err) return { counts: {} as Record<string, number>, score: null, error: err, size10Median: null, seconds: 0 };
      let good = 100;
      for (const [k, v] of Object.entries(peak)) {
        const x = params[k] === "auto" ? 2 : Number(params[k]);
        good -= Math.round(10 * Math.abs(Math.log(x / v)));
      }
      if (set === "all") good += 50; // the whole session has more units
      return {
        counts: { seeds_pnr: 50, seeds_merged: 20, units_final: good },
        score: { good, flagged: 0, lost: 0, score: good, low_pnr: 0, in_pieces: 0, bad_size: 0, duplicate: 0 },
        error: null,
        size10Median: 25,
        seconds: 0.01,
      };
    },
    async use(s) {
      set = s;
    },
    async startDiffThres() {
      return opts.knee === undefined ? 6 : opts.knee;
    },
  };
  return { r, calls };
}

const base = derivedParams(DEFAULTS, 10);

describe("step values and bounds", () => {
  it("scales thresholds and shifts correlations as Python does", () => {
    expect(stepValue("sparse_snr", 1.0, true)).toBe(1.4);
    expect(stepValue("sparse_snr", 1.0, false)).toBe(0.714);
    expect(stepValue("merge_corr", 0.8, true)).toBe(0.85);
    expect(stepValue("merge_corr", 0.98, true)).toBeNull(); // clipped to CORR_RANGE
    expect(stepValue("seeds_diff_thres", 6, true, true)).toBe(8);
    expect(stepValue("seeds_diff_thres", 2, false, true)).toBe(1);
    expect(stepValue("seeds_diff_thres", 1, false, true)).toBeNull();
    expect(stepValue("pnr_thres", "auto", true)).toBeNull();
  });
  it("bounds the trials as local_n_trials", () => {
    expect(localNTrials()).toBe(3 + 1 * 6 * (5 + 1) + 1 + 1);
    expect(localNTrials(5, 2, 1)).toBe(3 + 2 * 6 * 6 + 2);
  });
  it("picks the middle windows covering tune_frames", () => {
    const w = Array.from({ length: 12 }, (_, i) => ({ t0: i * 3000, t1: (i + 1) * 3000 }));
    expect(middleWindows(w, 40000)).toBeNull(); // 36 000 frames: not longer
    expect(middleWindows(w, 0)).toBeNull();
    expect(middleWindows(w, 12000)).toEqual({ from: 4, to: 8, segment: [12000, 24000] });
    expect(middleWindows(w, 9000)).toEqual({ from: 4, to: 7, segment: [12000, 21000] });
  });
  it("ranks by score, then fewer changes, then order, distinct values", () => {
    const t = (score: number | null, v: number, i = 0): Trial => ({
      step: "sparse",
      values: { ...pick(base), sparse_snr: v },
      counts: {},
      score:
        score === null
          ? null
          : { good: score, flagged: 0, lost: 0, score, low_pnr: 0, in_pieces: 0, bad_size: 0, duplicate: 0 },
      size10_median: null,
      seconds: i,
      error: null,
    });
    const r = ranked([t(5, 0.7), t(8, 0.5), t(null, 0.3), t(8, 1.0), t(8, 0.5, 1)], base);
    expect(r.map((x) => x.values.sparse_snr)).toEqual([1.0, 0.5, 0.7]); // 1.0 is the run's value: fewer changes
  });
});

describe("LocalSearch", () => {
  it("starts from the knee and START_PNR, walks each key, never proposes worse", async () => {
    const { r, calls } = runner();
    const s = new LocalSearch(r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
    });
    const proposed = await s.run();
    expect(s.trials[0].step).toBe("run");
    expect(s.trials[1].step).toBe("start");
    expect(s.start).toMatchObject({ seeds_diff_thres: 6, pnr_thres: 2 });
    expect(s.trials.length).toBeLessThanOrEqual(s.nTrials());
    expect(calls[0]).toMatchObject({ reference: true, params: DEFAULTS });
    expect(s.betterThanRun).toBe(true);
    expect(s.final?.score?.score).toBeGreaterThan(s.runTrial?.score?.score ?? 0);
    expect(proposed.sparse_snr).toBeCloseTo(0.51, 1); // 1 / 1.4 / 1.4 = 0.51
    expect(proposed.merge_corr).toBe(0.85);
    for (const k of SEARCHED) expect(s.grids[k]).toContain(s.start[k]);
  });

  it("keeps the run's values when the start scores no better", async () => {
    const { r } = runner({
      peak: { sparse_snr: 1.0, seeds_diff_thres: 3, merge_corr: 0.8, inita_corr: 0.8, loop_merge_corr: 0.9 },
    });
    const s = new LocalSearch(r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
    });
    const proposed = await s.run();
    expect(s.betterThanRun).toBe(false);
    expect(proposed).toEqual(pick(DEFAULTS));
    expect(s.trials.filter((t) => t.step === "start").length).toBe(2); // the start and the run's values on the search base
  });

  it("decides on the whole session when it explored on a segment", async () => {
    const { r, calls } = runner();
    const s = new LocalSearch(r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: true,
    });
    await s.run();
    const decide = s.trials.filter((t) => t.step === "decide");
    expect(decide.length).toBe(2); // the run's values and the best
    expect(calls.filter((c) => c.set === "all").length).toBe(2);
    expect(s.runTrial?.step).toBe("decide");
    expect(s.final?.step).toBe("decide");
  });

  it("falls back to the run when the start fails, and raises when the run fails too", async () => {
    const bad = runner({ fail: (p) => (p.pnr_thres === 2 ? "ValueError: no seeds survived pnr_refine" : null) });
    const s = new LocalSearch(bad.r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
    });
    await s.run();
    expect(s.searchError).toBeNull(); // the run's values are tried as the start too and work
    const worse = runner({ fail: () => "ValueError: boom" });
    const s2 = new LocalSearch(worse.r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
    });
    await expect(s2.run()).rejects.toThrow("auto-tune: the start setting failed (ValueError: boom)");
  });

  it("a second pass runs only after a move", async () => {
    const { r } = runner();
    const one = new LocalSearch(r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
    });
    const two = new LocalSearch(runner().r, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
      passes: 2,
    });
    await one.run();
    await two.run();
    expect(one.passes).toBe(1);
    expect(two.trials.length).toBeGreaterThanOrEqual(one.trials.length);
    expect(two.trials.length).toBeLessThanOrEqual(two.nTrials());
    expect(two.final?.score?.score).toBeGreaterThanOrEqual(one.final?.score?.score ?? 0);
  });

  it("lets a crash propagate instead of recording it as a search error", async () => {
    const { r } = runner();
    let n = 0;
    const crashing: TrialRunner = {
      ...r,
      async trial(params, reference) {
        if (++n === 3) throw new Error("worker crashed");
        return r.trial(params, reference);
      },
    };
    const s = new LocalSearch(crashing, {
      base,
      runParams: DEFAULTS,
      runValues: pick(DEFAULTS),
      integerDiff: true,
      segment: false,
    });
    await expect(s.run()).rejects.toThrow("worker crashed");
    expect(s.searchError).toBeNull();
  });
});
