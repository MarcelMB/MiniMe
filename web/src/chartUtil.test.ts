import { describe, expect, it } from "vitest";
import { fullExtent, resolveSeries } from "./chartUtil";

describe("chart helpers", () => {
  it("resolves CSS variable names and keeps literal colours", () => {
    const out = resolveSeries([{ stroke: "--accent" }, { stroke: "#fff" }], (n) => `res(${n})`);
    expect(out.map((s) => s.stroke)).toEqual(["res(--accent)", "#fff"]);
  });
  it("gives the full x extent for resetting the zoom", () => {
    expect(fullExtent([Float64Array.from([2, 3, 9]), Float64Array.from([1, 1, 1])])).toEqual([2, 9]);
    expect(fullExtent([[], []])).toBeNull();
  });
});
