import { describe, expect, it } from "vitest";
import { BATCH, batches } from "./gpuStages";

describe("batches", () => {
  it("cuts a chunk into GPU batches that never cross a tile-store block", () => {
    const b = batches(0, 250, 0);
    expect(b.every((n) => n <= BATCH)).toBe(true);
    expect(b.reduce((x, y) => x + y, 0)).toBe(250);
    // the running count of buffered frames hits 100 exactly, as the CPU path's flush does
    let c = 0;
    const hits: number[] = [];
    for (const n of b) {
      c += n;
      if (c % 100 === 0) hits.push(c);
      expect(Math.floor((c - n) / 100)).toBe(Math.floor((c - 1) / 100));
    }
    expect(hits).toEqual([100, 200]);
    // a chunk resumed with 96 frames buffered first fills the block
    expect(batches(10, 30, 96)[0]).toBe(4);
    expect(batches(5, 5, 0)).toEqual([]);
  });
});
