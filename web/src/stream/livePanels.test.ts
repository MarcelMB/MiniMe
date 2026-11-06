import { describe, expect, it } from "vitest";
import { tileGrid } from "../browser/derived/tiles";
import { writeNpy } from "../browser/npy";
import {
  guard,
  motionPanel,
  Pass2Live,
  seedsPanel,
  stitchFinalPanel,
  stitchWindowPanel,
  unitsPanel,
} from "./livePanels";

const i64 = (xs: number[]) => new BigInt64Array(xs.map(BigInt));
/** One unit at pixel 0 and one at pixels 4,5 of a 2x3 frame. */
const csr = (name: string) =>
  new Map([
    [`${name}_indptr.npy`, writeNpy([3], i64([0, 1, 3]))],
    [`${name}_indices.npy`, writeNpy([3], i64([0, 4, 5]))],
    [`${name}_data.npy`, writeNpy([3], Float32Array.from([1, 1, 1]))],
  ]);

describe("live panels", () => {
  it("motion: the block's last frame and the chunk's shifts", () => {
    const grid = tileGrid(2, 2, 2);
    const p = motionPanel(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]), 2, grid, 2, 2, new Float32Array(4), 50);
    expect(p.panel).toBe("motion");
    if (p.panel !== "motion") return;
    expect(p.at).toBe(50);
    expect(p.frame.length).toBe(4);
  });
  it("pass2: ranks of tiles seen, unknown before", () => {
    const live = new Pass2Live("window 1/1", { h: 2, w: 4, tile: 2 }, 2);
    const bytes = new Uint8Array(32 + 4 * (4 + 3 + 1 + 4 + 4)); // rank 1, P 4, T 3
    new DataView(bytes.buffer).setUint32(0, 1, true);
    new DataView(bytes.buffer).setUint32(8, 4, true);
    new DataView(bytes.buffer).setUint32(12, 3, true);
    new DataView(bytes.buffer).setUint32(16, 1, true);
    new DataView(bytes.buffer).setUint32(20, 4, true);
    new DataView(bytes.buffer).setUint32(24, 4, true);
    const p = live.tile(1, 1, bytes); // last tile: always sent
    expect(p?.panel).toBe("pass2");
    if (p?.panel !== "pass2") return;
    expect([...p.ranks]).toEqual([-1, 1]);
    expect(p.comps.length).toBe(1);
    expect(p.traces[0].length).toBe(3);
  });
  it("seeds: all seeds and the merged ones as (row, col)", () => {
    const out = new Map([
      ["seeds_init.npy", writeNpy([2, 3], i64([0, 1, 5, 1, 2, 3]))],
      ["seeds_merged.npy", writeNpy([1, 2], i64([1, 2]))],
    ]);
    const p = seedsPanel("window 1/1", out, new Uint8Array(6), 2, 3);
    if (p.panel !== "seeds") throw new Error("panel");
    expect([...p.seeds]).toEqual([0, 1, 1, 2]);
    expect([...p.merged]).toEqual([1, 2]);
  });
  it("units: outlines, count and brightest traces after init", () => {
    const out = csr("A_merge");
    out.set("C_merge.npy", writeNpy([2, 3], Float32Array.from([0, 1, 0, 0, 9, 0])));
    const p = unitsPanel("window 1/1", "init", out, new Float32Array(6), 2, 3);
    if (p?.panel !== "units") throw new Error("panel");
    expect(p.count).toBe(2);
    expect(p.ids[0]).toBe(1);
    expect(p.polygons.length).toBe(2);
  });
  it("units: null when the step's files are missing", () => {
    expect(unitsPanel("w", "cnmf", new Map(), new Float32Array(6), 2, 3)).toBeNull();
  });
  it("stitch: a window's outlines and centroids, then the links", () => {
    const w = stitchWindowPanel(0, csr("A"), 2, 3);
    if (w?.panel !== "stitch" || "edges" in w) throw new Error("panel");
    expect(w.centroids.length).toBe(4);
    const out = csr("A");
    out.set("edges.npy", writeNpy([1, 2], i64([0, 1])));
    const f = stitchFinalPanel(out);
    if (f?.panel !== "stitch" || !("edges" in f)) throw new Error("panel");
    expect([...f.edges]).toEqual([0, 1]);
    expect(f.units).toBe(2);
  });
  it("guard turns a failing panel off after one log line", () => {
    const logs: string[] = [];
    const posted: unknown[] = [];
    const bad = () => {
      throw new Error("boom");
    };
    guard(
      "test-panel",
      bad,
      (p) => posted.push(p),
      (m) => logs.push(m),
    );
    guard(
      "test-panel",
      bad,
      (p) => posted.push(p),
      (m) => logs.push(m),
    );
    expect(logs).toEqual(["live view: test-panel: boom"]);
    expect(posted).toEqual([]);
  });
});
