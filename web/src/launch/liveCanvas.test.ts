import { describe, expect, it } from "vitest";
import { linkSegments, windowColor } from "./LiveCanvas";

describe("live canvas helpers", () => {
  it("draws a link between the centroids of two units across windows", () => {
    const windows = new Map([
      [0, { first: 0, polygons: [[]], centroids: Float64Array.from([1, 2]) }],
      [1, { first: 1, polygons: [[]], centroids: Float64Array.from([5, 6]) }],
    ]);
    expect(linkSegments(windows, Float64Array.from([0, 1]))).toEqual([[2, 1, 6, 5]]);
  });
  it("numbers units by each window's first index when earlier windows are missing", () => {
    // window 0 (units 0-2) never reached the page; window 1 holds units 3-4, window 2 units 5-6
    const windows = new Map([
      [1, { first: 3, polygons: [], centroids: Float64Array.from([1, 2, 3, 4]) }],
      [2, { first: 5, polygons: [], centroids: Float64Array.from([7, 8, 9, 10]) }],
    ]);
    const edges = Float64Array.from([4, 5, 0, 6, 3, 6]);
    expect(linkSegments(windows, edges)).toEqual([
      [4, 3, 8, 7],
      [2, 1, 10, 9],
    ]);
  });
  it("skips links to units it has no window for", () => {
    expect(linkSegments(new Map(), Float64Array.from([0, 1]))).toEqual([]);
  });
  it("gives windows distinct colours", () => {
    expect(windowColor(0)).not.toBe(windowColor(1));
  });
});
