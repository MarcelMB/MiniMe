import { describe, expect, it } from "vitest";
import type { Unit } from "../../types";
import { rankHistogram } from "./Compression";
import { sortUnits } from "./Units";

const u = (id: number, size: number, peak: number | null, c: [number, number]): Unit => ({
  id,
  size_px: size,
  peak,
  centroid: c,
  polygons: [],
});

describe("units table", () => {
  it("sorts by any column, nulls last", () => {
    const us = [u(0, 50, 2, [5, 9]), u(1, 80, null, [1, 3]), u(2, 20, 7, [9, 1])];
    expect(sortUnits(us, "size_px", true).map((x) => x.id)).toEqual([1, 0, 2]);
    expect(sortUnits(us, "peak", true).map((x) => x.id)).toEqual([2, 0, 1]);
    expect(sortUnits(us, "col", false).map((x) => x.id)).toEqual([2, 1, 0]);
  });

  it("orders two null peaks by id", () => {
    const us = [u(3, 1, null, [0, 0]), u(1, 1, null, [0, 0]), u(2, 1, 5, [0, 0])];
    expect(sortUnits(us, "peak", true).map((x) => x.id)).toEqual([2, 1, 3]);
    expect(sortUnits(us, "peak", false).map((x) => x.id)).toEqual([2, 1, 3]);
  });

  it("keeps a null last when sorting ascending", () => {
    const us = [u(0, 1, null, [0, 0]), u(1, 1, 9, [0, 0]), u(2, 1, 3, [0, 0])];
    expect(sortUnits(us, "peak", false).map((x) => x.id)).toEqual([2, 1, 0]);
  });

  it("breaks ties by id", () => {
    const us = [u(2, 10, 1, [0, 0]), u(0, 10, 1, [0, 0]), u(1, 10, 1, [0, 0])];
    expect(sortUnits(us, "size_px", true).map((x) => x.id)).toEqual([0, 1, 2]);
    expect(sortUnits(us, "size_px", false).map((x) => x.id)).toEqual([0, 1, 2]);
  });
});

describe("rank histogram", () => {
  it("counts tiles per rank bin", () => {
    const h = rankHistogram([1, 2, 2, 10], 3);
    expect(h.map((b) => b.n)).toEqual([3, 0, 1]);
    expect(h[0].lo).toBe(1);
    expect(h.at(-1)!.hi).toBe(10);
  });
});
