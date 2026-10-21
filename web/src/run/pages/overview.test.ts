import { describe, expect, it } from "vitest";
import { funnel, stageBars } from "./Overview";

describe("overview data", () => {
  it("builds the seed funnel in order", () => {
    const f = funnel({ seeds_init: 59504, seeds_pnr: 39769, seeds_merged: 582, units_final: 503 });
    expect(f.map((x) => x.value)).toEqual([59504, 39769, 582, 503]);
    expect(f[0].name).toBe("Seeds");
  });
  it("scales stage bars to the longest stage", () => {
    const b = stageBars([
      { stage: "pass1", seconds: 450 },
      { stage: "pass2", seconds: 503 },
      { stage: "probe", seconds: 0.5 },
    ]);
    expect(b.map((x) => x.pct)).toEqual([(450 / 503) * 100, 100, (0.5 / 503) * 100]);
  });
});
