import { describe, expect, it } from "vitest";
import { chunkBounds } from "./poolStages";
import { FLUSH, pieces, readTile } from "./slabs";

/** A slab as `Pass1Engine.take_tiles()` blocks lay it out: per block of up to FLUSH frames, tile after tile,
 * each `frames x pixels`; the value of frame `f` of tile `k` pixel `p` of chunk `c` is a function of all four. */
function slab(c: number, n: number, pixels: number[]): Uint8Array {
  const out: number[] = [];
  for (let f0 = 0; f0 < n; f0 += FLUSH) {
    const nb = Math.min(FLUSH, n - f0);
    pixels.forEach((px, k) => {
      for (let f = f0; f < f0 + nb; f++) for (let p = 0; p < px; p++) out.push((c * 31 + k * 17 + f * 7 + p) & 255);
    });
  }
  return Uint8Array.from(out);
}

const reader = (files: Uint8Array[]) =>
  ({
    readAsArrayBuffer: (b: { c: number; at: number; end: number }) => files[b.c].slice(b.at, b.end).buffer,
  }) as unknown as FileReaderSync;

describe("slabs", () => {
  const pixels = [6, 4, 9];
  const frames = [250, 100, 37];

  it("gives tile k's frames in order, chunk after chunk, whatever the block boundaries", () => {
    const files = frames.map((n, c) => slab(c, n, pixels));
    const slabs = files.map((_, c) => ({ slice: (at: number, end: number) => ({ c, at, end }) })) as unknown as File[];
    for (let k = 0; k < pixels.length; k++) {
      const got = readTile(slabs, frames, pixels, k, reader(files));
      const want: number[] = [];
      frames.forEach((n, c) => {
        for (let f = 0; f < n; f++) for (let p = 0; p < pixels[k]; p++) want.push((c * 31 + k * 17 + f * 7 + p) & 255);
      });
      expect(Array.from(got)).toEqual(want);
    }
  });

  it("finds a tile's pieces: one per block, inside the slab", () => {
    const ps = pieces(250, pixels, 1);
    expect(ps.map((p) => p.len)).toEqual([100 * 4, 100 * 4, 50 * 4]);
    expect(ps[0].at).toBe(100 * 6);
    expect(ps[1].at).toBe(100 * 19 + 100 * 6);
    expect(ps[2].at).toBe(200 * 19 + 50 * 6);
    expect(slab(0, 250, pixels).length).toBe(250 * 19);
  });
});

describe("chunkBounds", () => {
  it("cuts a window into about two chunks per worker, in multiples of 100 frames", () => {
    expect(chunkBounds(3000, 8)).toEqual([
      0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000,
    ]);
    expect(chunkBounds(3000, 1)).toEqual([0, 1500, 3000]);
    expect(chunkBounds(1000, 4)).toEqual([0, 200, 400, 600, 800, 1000]);
    expect(chunkBounds(50, 8)).toEqual([0, 50]);
  });

  it("with `even` gives each worker one chunk of the same size", () => {
    expect(chunkBounds(3000, 8, true)).toEqual([0, 375, 750, 1125, 1500, 1875, 2250, 2625, 3000]);
    expect(chunkBounds(1001, 4, true)).toEqual([0, 251, 502, 753, 1001]);
  });

  it("never gives more chunks than the tile parts allow", () => {
    for (const [t, n] of [
      [3000, 8],
      [20000, 32],
      [300, 32],
      [123456, 12],
    ])
      expect(chunkBounds(t, n).length - 1).toBeLessThanOrEqual(2 * n);
  });
});
