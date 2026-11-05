import { describe, expect, it } from "vitest";
import { state } from "../state";
import { setupE2eRun } from "../testing/e2eRun";
import { checkDiameter, estimateDiameter, scoreRun } from "./score";

describe("scoring a run folder", () => {
  it("estimates the diameter from the footprints and flags the e2e run's units", async () => {
    const { rid } = await setupE2eRun();
    const r = await state.registry.get(rid);
    const d = await estimateDiameter(r);
    expect(d).toBeGreaterThan(4);
    expect(d).toBeLessThan(16);
    const s = await scoreRun(
      r,
      { diameter: 7, pnr_min: 12, piece_frac: 0.9, size_lo: 0.3, size_hi: 3, dup_corr: 0.6, keep_dist: 0.5 },
      0.4,
      [],
    );
    expect(s.flags.good + s.flags.flagged).toBe(15);
    expect(Object.keys(s.flags.reasons)).toEqual(["low_pnr", "in_pieces", "bad_size", "duplicate"]);
    expect(s.pairs.corr_bins).toHaveLength(29);
    expect(s.csv.split("\n")[0]).toContain("unit_id,pnr");
    expect(s.summary.lost).toBe(0);
    expect(s.good.length).toBe(s.flags.good);
    expect(() => checkDiameter(100, 64, 64)).toThrow(
      "diameter must be a finite number of px of at most 64 (the smaller field-of-view side), got 100",
    );
  });
});
