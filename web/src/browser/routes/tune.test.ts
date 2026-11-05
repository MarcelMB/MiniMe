import { describe, expect, it } from "vitest";
import { sweepCsv } from "../tune/doc";
import type { Trial } from "../tune/types";
import { readSweep } from "./tune";

describe("readSweep", () => {
  it("round-trips sweepCsv, also an error with a comma, a quote and a newline", () => {
    const score = { good: 5, flagged: 2, lost: 1, score: 4.5, low_pnr: 1, in_pieces: 0, bad_size: 1, duplicate: 0 };
    const values = {
      seeds_diff_thres: 3,
      pnr_thres: "auto",
      merge_corr: 0.8,
      init_merge_corr: 0.8,
      inita_corr: 0.8,
      sparse_snr: 0.5,
      loop_merge_corr: 0.9,
    };
    const trials: Trial[] = [
      { step: "seeds", values, counts: {}, score, size10_median: 12.5, seconds: 1.25, error: null },
      { step: "snr", values, counts: {}, score: null, size10_median: null, seconds: 2, error: 'boom, "x"\nline 2' },
    ];
    const got = readSweep(sweepCsv(trials));
    expect(got).toHaveLength(2);
    expect(got[0]).toMatchObject({ step: "seeds", values, score, size10_median: 12.5, seconds: 1.25, error: null });
    expect(got[1]).toMatchObject({ step: "snr", score: null, size10_median: null, error: 'boom, "x"\nline 2' });
  });
});
