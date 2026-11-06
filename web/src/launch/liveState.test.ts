import { describe, expect, it } from "vitest";
import { emptyLive, panelForStage, reduceLive, shiftSeries, shownPanel } from "./liveState";

const motion = (at: number, n: number) => ({
  panel: "motion" as const,
  frame: new Uint8Array(4),
  h: 2,
  w: 2,
  f: 1,
  at,
  shifts: Float32Array.from({ length: 2 * n }, (_, i) => i),
});

describe("reduceLive", () => {
  it("keeps panel order and the latest motion frame, and merges shift chunks", () => {
    let s = reduceLive(emptyLive(), motion(0, 3));
    s = reduceLive(s, motion(100, 2));
    s = reduceLive(s, motion(0, 4)); // the same chunk grown
    expect(s.order).toEqual(["motion"]);
    expect(s.motion?.at).toBe(0);
    const [x, a, b] = shiftSeries(s.shifts);
    expect([...x]).toEqual([0, 1, 2, 3, 100, 101]);
    expect([...a]).toEqual([0, 2, 4, 6, 0, 2]);
    expect([...b]).toEqual([1, 3, 5, 7, 1, 3]);
  });
  it("reduces pass2 with unknown earlier tiles", () => {
    const ranks = Int32Array.from([-1, -1, 7]);
    const s = reduceLive(emptyLive(), {
      panel: "pass2",
      label: "window 1/1",
      grid: { h: 4, w: 6, tile: 2 },
      ranks,
      k: 2,
      comps: [],
      th: 2,
      tw: 2,
      traces: [],
    });
    expect(s.pass2?.ranks[0]).toBe(-1);
    expect(s.order).toEqual(["pass2"]);
  });
  it("collects stitch windows and the final links", () => {
    let s = reduceLive(emptyLive(), {
      panel: "stitch",
      window: 0,
      first: 0,
      polygons: [],
      centroids: new Float64Array(2),
    });
    s = reduceLive(s, { panel: "stitch", window: 1, first: 1, polygons: [], centroids: new Float64Array(4) });
    s = reduceLive(s, { panel: "stitch", window: -1, edges: Float64Array.from([0, 2]), units: 2 });
    expect([...s.stitch.windows.keys()]).toEqual([0, 1]);
    expect(s.stitch.windows.get(1)?.first).toBe(1);
    expect(s.stitch.units).toBe(2);
    expect(s.order).toEqual(["stitch"]);
  });
  it("keeps only the newest 8 stitch windows", () => {
    let s = emptyLive();
    for (let w = 0; w < 12; w++)
      s = reduceLive(s, { panel: "stitch", window: w, first: 2 * w, polygons: [], centroids: new Float64Array(4) });
    expect([...s.stitch.windows.keys()].sort((a, b) => a - b)).toEqual([4, 5, 6, 7, 8, 9, 10, 11]);
    expect(s.stitch.windows.get(11)?.first).toBe(22);
  });
  it("decimates a long shift trace", () => {
    const m = new Map([[0, new Float32Array(2 * 10_000)]]);
    expect(shiftSeries(m, 2000)[0].length).toBe(2000);
  });
});

describe("auto-follow", () => {
  it("maps each running stage to its panel", () => {
    expect(
      ["motion", "pass1", "pass2", "seeds", "init", "cnmf", "stitch", "stream", undefined].map(panelForStage),
    ).toEqual(["motion", "motion", "pass2", "seeds", "units", "units", "stitch", null, null]);
  });
  it("shows the pick, else the running stage's panel when it has data, else the newest", () => {
    const order = ["motion", "pass2", "stitch"] as const;
    expect(shownPanel([...order], null, "pass2")).toBe("pass2");
    expect(shownPanel([...order], null, "units")).toBe("stitch"); // no units preview yet
    expect(shownPanel([...order], "motion", "pass2")).toBe("motion");
    expect(shownPanel([], null, "motion")).toBeNull();
  });
});
