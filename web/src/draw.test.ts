import { describe, expect, it } from "vitest";
import { clientToPixel, fillImageData, percentile, pickUnit, pointInPolygon } from "./draw";
import { sourceOptions } from "./run/RunContext";
import type { Unit } from "./types";

const square = [
  [10, 10],
  [20, 10],
  [20, 20],
  [10, 20],
];
const unit = (id: number, poly: number[][], c: [number, number]): Unit => ({
  id,
  centroid: c,
  size_px: 1,
  peak: null,
  polygons: [poly],
});

describe("draw helpers", () => {
  it("maps lo..hi to 0..255 and clamps", () => {
    const out = fillImageData(
      { data: [0, 5, 10, 20], W: 4, H: 1, lo: 0, hi: 10 },
      { data: new Uint8ClampedArray(16), width: 4, height: 1 } as ImageData,
    );
    expect(Array.from(out.data.filter((_, i) => i % 4 === 0))).toEqual([0, 128, 255, 255]);
    expect(out.data[3]).toBe(255);
  });
  it("computes percentiles", () => {
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
    expect(percentile([1, 2, 3, 4, 5], 100)).toBe(5);
  });
  it("finds points in polygons and picks units", () => {
    expect(pointInPolygon(15, 15, square)).toBe(true);
    expect(pointInPolygon(25, 15, square)).toBe(false);
    const units = [unit(0, square, [15, 15]), unit(1, [[40, 40]], [40, 40])];
    expect(pickUnit(units, 12, 18)).toBe(0);
    expect(pickUnit(units, 43, 41)).toBe(1); // a one-point outline: nearest centroid
    expect(pickUnit(units, 100, 100)).toBeNull();
  });
  it("disables sources the run does not have, with the reason", () => {
    const opts = sourceOptions({ raw: false, pre: true, seeds_saved: true, init_saved: true });
    const raw = opts.find((o) => o.value === "raw")!;
    expect(raw.disabled).toBe(true);
    expect(raw.title).toMatch(/raw videos/);
    expect(opts.find((o) => o.value === "recon")!.disabled).toBeFalsy();
  });
});

describe("clientToPixel", () => {
  it("maps inside a wide, letterboxed box", () => {
    const rect = { left: 100, top: 0, width: 400, height: 100 }; // 100x100 image -> 100 px wide, spans x 250..350
    expect(clientToPixel(rect, 100, 100, 300, 50)).toEqual([49.5, 49.5]);
    expect(clientToPixel(rect, 100, 100, 250, 0)).toEqual([-0.5, -0.5]);
    expect(clientToPixel(rect, 100, 100, 200, 50)).toBeNull(); // left bar
    expect(clientToPixel(rect, 100, 100, 360, 50)).toBeNull(); // right bar
  });
  it("maps inside a tall box", () => {
    const rect = { left: 0, top: 50, width: 100, height: 300 };
    expect(clientToPixel(rect, 200, 100, 100, 200)).toEqual([199.5, 49.5]); // 100x50 image spans y 175..225
    expect(clientToPixel(rect, 200, 100, 50, 100)).toBeNull(); // top bar
  });
  it("returns null in the bar", () => {
    expect(clientToPixel({ left: 0, top: 0, width: 400, height: 100 }, 100, 100, 10, 10)).toBeNull();
  });
});
