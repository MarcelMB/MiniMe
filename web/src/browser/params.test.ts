import { describe, expect, it } from "vitest";
import { getJSON } from "../api";
import { derivedParams } from "./routes/params";

// Python: derived_params(Params(), d) -> (tophat_wnd/seeds_max_wnd/..., merge_min_dist, size_thres[0])
// 1.125 and 5.375 are where Python's round(0.4 * d, 1) differs from np.round (0.5 vs 0.4, 2.1 vs 2.2).
const CASES: [number, number, number, number][] = [
  [0.4, 1, 0.2, 1],
  [1.125, 1, 0.5, 1],
  [2.5, 2, 1.0, 1], // round(2.5) is 2: ties go to even
  [5.375, 5, 2.1, 6],
  [7.125, 7, 2.9, 10],
  [7.5, 8, 3.0, 11],
  [12.5, 12, 5.0, 31],
  [100, 100, 40.0, 1963],
];

describe("derivedParams", () => {
  it.each(CASES)("diameter %f matches derived_params", (diameter, d, minDist, lo) => {
    expect(derivedParams(diameter)).toEqual({
      tophat_wnd: d,
      seeds_max_wnd: d,
      inita_wnd: d,
      dl_wnd: d,
      merge_dist: d,
      merge_min_dist: minDist,
      size_thres: [lo, null],
    });
  });

  it("keeps main's key order", () => {
    expect(Object.keys(derivedParams(10))).toEqual([
      "tophat_wnd",
      "seeds_max_wnd",
      "inita_wnd",
      "dl_wnd",
      "merge_dist",
      "merge_min_dist",
      "size_thres",
    ]);
  });
});

describe("/params/derived", () => {
  it("answers 400 with main's message for a diameter that is not above zero", async () => {
    const bad = (d: string) => getJSON(`/params/derived?diameter=${d}`);
    await expect(bad("-1")).rejects.toMatchObject({
      status: 400,
      message: "diameter must be a number of px > 0, got -1.0",
    });
    await expect(bad("0")).rejects.toMatchObject({
      status: 400,
      message: "diameter must be a number of px > 0, got 0.0",
    });
    await expect(bad("-2.5")).rejects.toMatchObject({
      status: 400,
      message: "diameter must be a number of px > 0, got -2.5",
    });
    await expect(bad("nan")).rejects.toMatchObject({
      status: 400,
      message: "diameter must be a number of px > 0, got nan",
    });
  });

  it("answers 422 for a missing or non-numeric diameter, as main's request validation does", async () => {
    await expect(getJSON("/params/derived")).rejects.toMatchObject({ status: 422 });
    await expect(getJSON("/params/derived?diameter=abc")).rejects.toMatchObject({ status: 422 });
  });
});
