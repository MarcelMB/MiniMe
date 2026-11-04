import { expect, it } from "vitest";
import { lowpassGain, overview, traceWindow, zooms } from "./traces";

it("gain is 1 below f/2, 0 from 1.5 f, cosine between (CaLab's curve)", () => {
  const g = lowpassGain(100, 30, 2);
  // rfftfreq(100, 1/30): f_k = 0.3 k
  expect(g[3]).toBe(1); // 0.9 Hz < 1 Hz
  expect(g[10]).toBe(0); // 3 Hz >= 3 Hz
  expect(g[5]).toBeCloseTo(0.5 * (1 + Math.cos((Math.PI * (1.5 - 1)) / 2)), 12);
});

it("short traces come back unchanged and windows check their bounds", () => {
  const x = new Float32Array([1, 2, 3]);
  expect(Array.from(traceWindow(x, 0, 3, 30, 1))).toEqual([1, 2, 3]);
  expect(() => traceWindow(x, 2, 2, null, null)).toThrow("window 2:2 outside 0:3");
  expect(() => traceWindow(new Float32Array(20), 0, 20, null, 1)).toThrow("frame rate");
});

it("overview gives min/max per span and handles 0 units", async () => {
  const rows = [new Float32Array([1, 5, 2, 0, 3])];
  const o = await overview(async (i) => rows[i], 1, 5, 2);
  // spans: linspace(0, 5, 3)[:-1] = [0, 2] -> [0,2) and [2,5)
  expect(Array.from(o)).toEqual([1, 0, 5, 3]);
  expect((await overview(async () => new Float32Array(0), 0, 5, 3)).length).toBe(0);
});

it("zooms: brightest units first, windows only with a frame rate", () => {
  expect(zooms(new Float32Array([1, 9, 3]), 100, null)).toEqual({ units: [1, 2, 0], windows: [] });
  const z = zooms(new Float32Array([1]), 30 * 3600, 30);
  expect(z.windows.slice(0, 2)).toEqual([
    { start: 0, stop: 9000 },
    { start: 36000, stop: 45000 },
  ]);
});

it("zooms with no units still lists the windows", () => {
  expect(zooms(new Float32Array(0), 600, 1)).toEqual({ units: [], windows: [{ start: 0, stop: 300 }] });
});
