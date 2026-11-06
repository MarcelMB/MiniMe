import { describe, expect, it } from "vitest";
import { linkSegments, windowColor } from "./LiveCanvas";

describe("live canvas helpers", () => {
  it("draws a link between the centroids of two units across windows", () => {
    const windows = new Map([
      [0, { polygons: [[]], centroids: Float64Array.from([1, 2]) }],
      [1, { polygons: [[]], centroids: Float64Array.from([5, 6]) }],
    ]);
    expect(linkSegments(windows, Float64Array.from([0, 1]))).toEqual([[2, 1, 6, 5]]);
  });
  it("skips links to units it has no window for", () => {
    expect(linkSegments(new Map(), Float64Array.from([0, 1]))).toEqual([]);
  });
  it("gives windows distinct colours", () => {
    expect(windowColor(0)).not.toBe(windowColor(1));
  });
});
