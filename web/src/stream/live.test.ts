import { describe, expect, it } from "vitest";
import { tileGrid } from "../browser/derived/tiles";
import { writeNpy } from "../browser/npy";
import {
  brightest,
  csrFromFiles,
  decimate,
  frameFromBlock,
  makeThrottle,
  meanImage,
  parseTileFactors,
  previewImage,
  tileOf,
} from "./live";

/** TileFactors bytes as `poolio.rs` `to_bytes` writes them: 8 u32 lengths, five f32 arrays, energy f64. */
function tileBytes(rank: number, P: number, T: number, mean: number[]): Uint8Array {
  const w = Float32Array.from({ length: P * rank }, (_, i) => i);
  const v = Float32Array.from({ length: T * rank }, (_, i) => -i);
  const s = Float32Array.from({ length: rank }, (_, i) => rank - i);
  const parts = [w, v, s, Float32Array.from(mean), new Float32Array(P)];
  const head = new Uint32Array([rank, rank, w.length, v.length, s.length, P, P, 0]);
  const out = new Uint8Array(32 + 4 * parts.reduce((n, a) => n + a.length, 0));
  out.set(new Uint8Array(head.buffer), 0);
  let at = 32;
  for (const a of parts) {
    out.set(new Uint8Array(a.buffer), at);
    at += a.byteLength;
  }
  return out;
}

describe("live helpers", () => {
  it("throttles to hz but lets a forced message through", () => {
    let t = 0;
    const ok = makeThrottle(4, () => t);
    expect(ok()).toBe(true);
    t = 100;
    expect(ok()).toBe(false);
    expect(ok(true)).toBe(true);
    t = 400;
    expect(ok()).toBe(true);
  });

  it("downsamples to at most max px and stretches to u8", () => {
    const src = Float32Array.from({ length: 600 * 600 }, (_, i) => i % 600);
    const p = previewImage(src, 600, 600, 256);
    expect([p.f, p.h, p.w]).toEqual([3, 200, 200]);
    expect(p.image.length).toBe(200 * 200);
    expect(p.image[0]).toBe(0);
    expect(p.image[199]).toBe(255);
  });

  it("rebuilds a frame from a tile-major block", () => {
    const [h, w, n] = [3, 5, 2];
    const grid = tileGrid(h, w, 2);
    const pixels = grid.map((t) => (t.y1 - t.y0) * (t.x1 - t.x0));
    const block = new Uint8Array(n * pixels.reduce((a, b) => a + b, 0));
    let o = 0;
    for (const t of grid) {
      for (let f = 0; f < n; f++)
        for (let y = t.y0; y < t.y1; y++) for (let x = t.x0; x < t.x1; x++) block[o++] = 10 * f + y * w + x;
    }
    const frame = frameFromBlock(block, n, 1, grid, w);
    expect([...frame]).toEqual(Array.from({ length: h * w }, (_, p) => 10 + p));
  });

  it("parses tile factors and assembles the mean image", () => {
    const grid = tileGrid(2, 3, 2);
    const a = tileBytes(2, 4, 5, [1, 2, 3, 4]);
    const b = tileBytes(1, 2, 5, [9, 8]);
    const t = parseTileFactors(a);
    expect(t.rank).toBe(2);
    expect(t.w.length).toBe(8);
    expect(t.v.length).toBe(10);
    expect([...meanImage([a, b], grid, 2, 3)]).toEqual([1, 2, 9, 3, 4, 8]);
  });

  it("decimates to at most max points", () => {
    expect(
      decimate(
        Float32Array.from({ length: 1000 }, (_, i) => i),
        500,
      ).length,
    ).toBe(500);
    expect(decimate([1, 2, 3], 500).length).toBe(3);
  });

  it("reads a CSR from npy files, or null when absent", () => {
    const files = new Map([
      ["A_indptr.npy", writeNpy([3], new BigInt64Array([0n, 1n, 3n]))],
      ["A_indices.npy", writeNpy([3], new BigInt64Array([4n, 0n, 5n]))],
      ["A_data.npy", writeNpy([3], Float32Array.from([1, 2, 3]))],
    ]);
    const a = csrFromFiles(files, "A", 6)!;
    expect(a.shape).toEqual([2, 6]);
    expect([...a.indices]).toEqual([4, 0, 5]);
    expect(csrFromFiles(files, "A_merge", 6)).toBeNull();
  });

  it("picks the brightest traces by peak", () => {
    const c = Float32Array.from([0, 1, 0, 5, 0, 2]);
    expect(brightest(c, 3, 2, 2)).toEqual([1, 2]);
  });

  it("reads the tile size from pass1.json", () => {
    const files = new Map([["pass1.json", new TextEncoder().encode(JSON.stringify({ tile: 64 }))]]);
    expect(tileOf(files)).toBe(64);
  });
});
