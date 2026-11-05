import { describe, expect, it } from "vitest";
import { atEdge, bestPerValue, sweepCsv } from "./doc";
import type { Trial } from "./types";

const trial = (
  step: string,
  values: Record<string, number | string>,
  score: number | null,
  error: string | null = null,
): Trial => ({
  step,
  values: {
    seeds_diff_thres: 3,
    pnr_thres: "auto",
    merge_corr: 0.8,
    init_merge_corr: 0.8,
    inita_corr: 0.8,
    sparse_snr: 1,
    loop_merge_corr: 0.9,
    ...values,
  },
  counts: { seeds_init: 10, seeds_pnr: 8, seeds_merged: 6, units_init_merge: 5, units_final: 4 },
  score:
    score === null
      ? null
      : {
          good: score,
          flagged: 1,
          lost: 0,
          score: score - 1,
          low_pnr: 1,
          in_pieces: 0,
          bad_size: 0,
          duplicate: 0,
        },
  size10_median: 25,
  seconds: 0.123456,
  error,
});

describe("the sweep table and curves", () => {
  it("writes sweep.csv with Python's columns, rounded seconds and blank Nones", () => {
    const csv = sweepCsv([trial("run", {}, 10), trial("seeds", { seeds_diff_thres: 4 }, null, "ValueError: x")]);
    const [head, r1, r2] = csv.trim().split("\n");
    expect(head).toBe(
      "step,seeds_diff_thres,pnr_thres,merge_corr,init_merge_corr,inita_corr,sparse_snr,loop_merge_corr,good,flagged,lost,score,low_pnr,in_pieces,bad_size,duplicate,seeds_init,seeds_pnr,seeds_merged,units_init_merge,units_final,size10_median,seconds,error",
    );
    expect(r1).toBe("run,3,auto,0.8,0.8,0.8,1,0.9,10,1,0,9,1,0,0,0,10,8,6,5,4,25,0.12,");
    expect(r2).toBe("seeds,4,auto,0.8,0.8,0.8,1,0.9,,,,,,,,,10,8,6,5,4,25,0.12,ValueError: x");
  });
  it("gives the best good/flagged per value in grid order", () => {
    const ts = [
      trial("sparse", { sparse_snr: 0.7 }, 5),
      trial("sparse", { sparse_snr: 0.5 }, 8),
      trial("sparse", { sparse_snr: 0.5 }, 3),
      trial("seeds", { pnr_thres: 2 }, 4),
      trial("seeds", { pnr_thres: "auto" }, 6),
    ];
    expect(bestPerValue(ts, "sparse_snr")).toEqual([
      { value: 0.5, good: 8, flagged: 1 },
      { value: 0.7, good: 5, flagged: 1 },
    ]);
    expect(bestPerValue(ts, "pnr_thres")).toEqual([
      { value: "auto", good: 6, flagged: 1 },
      { value: 2, good: 4, flagged: 1 },
    ]);
    expect(bestPerValue(ts, "merge_corr")).toEqual([]); // no merge step tried
  });
  it("marks a value at the edge of a grid with more than one number", () => {
    expect(atEdge(0.5, [0.5, 0.7, 1.0])).toBe(true);
    expect(atEdge(0.7, [0.5, 0.7, 1.0])).toBe(false);
    expect(atEdge("auto", ["auto", 1, 2])).toBe(false);
    expect(atEdge(1, [1])).toBe(false);
  });
});
