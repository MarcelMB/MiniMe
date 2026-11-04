import { existsSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { getArray } from "../api";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { settled } from "./testing/settled";

// The views of the run the browser pipeline wrote (`e2e/stream.spec.ts` with STREAM_KEEP_STORE=1
// saves it to e2e/.data/browser-run/run, tile store included): the frames rebuilt from the
// window factors and read from the tile store, the motion residual and the noise image. Main's
// routes cannot open these factors, so there is no contract to compare with: the checks are
// what the views must satisfy. Skipped without the export (the preprocessed frames also need
// its tile store).
const RUN = join(DATA, "browser-run/run");
const have = existsSync(join(RUN, "report.json"));
const haveTiles = existsSync(join(RUN, "tiles/w00/tile_000.part00"));

/** Pearson correlation of two equally long arrays. */
function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  const [ma, mb] = [mean(a), mean(b)];
  let [ab, aa, bb] = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const [x, y] = [a[i] - ma, b[i] - mb];
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return ab / Math.sqrt(aa * bb);
}
const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length;

describe.skipIf(!have)("the views of a run the browser wrote, from its window factors", () => {
  let rid = "";
  beforeAll(async () => {
    rid = (await setupE2eRun(RUN)).rid;
  });
  const frame = (t: number, source: string) => settled(() => getArray(`/runs/${rid}/frame/${t}?source=${source}`));

  it("answers recon frames as (H, W) uint8", async () => {
    for (const t of [0, 150, 299]) {
      const a = await frame(t, "recon");
      expect([a.shape, a.dtype]).toEqual([[64, 64], "uint8"]);
      expect(new Set(a.data).size).toBeGreaterThan(10); // a picture, not a constant
    }
  });

  it.skipIf(!haveTiles)("answers pre frames as (H, W) uint8", async () => {
    for (const t of [0, 150, 299]) {
      const a = await frame(t, "pre");
      expect([a.shape, a.dtype]).toEqual([[64, 64], "uint8"]);
      expect(new Set(a.data).size).toBeGreaterThan(10);
    }
  });

  it.skipIf(!haveTiles)("rebuilds frames that follow the preprocessed ones: Pearson > 0.9 over 5 frames", async () => {
    const corr: number[] = [];
    for (const t of [0, 75, 150, 225, 299]) {
      const [recon, pre] = [await frame(t, "recon"), await frame(t, "pre")];
      corr.push(pearson(recon.data, pre.data));
    }
    console.log(`recon vs pre Pearson over 5 frames: ${corr.map((c) => c.toFixed(4)).join(" ")}`);
    for (const c of corr) expect(c).toBeGreaterThan(0.9);
  });

  it("answers the motion residual as (n_blocks, 3) with no leftover shift on a session without drift", async () => {
    const a = await settled(() => getArray(`/runs/${rid}/motion/residual`));
    expect(a.dtype).toBe("float32");
    expect(a.shape[1]).toBe(3);
    expect(a.shape[0]).toBeGreaterThan(0);
    for (let b = 0; b < a.shape[0]; b++) {
      const [dy, dx, c] = [a.data[3 * b], a.data[3 * b + 1], a.data[3 * b + 2]];
      console.log(`block ${b}: dy ${dy.toFixed(3)} dx ${dx.toFixed(3)} corr ${c.toFixed(4)}`);
      expect(Math.abs(dy)).toBeLessThan(1);
      expect(Math.abs(dx)).toBeLessThan(1);
      expect(c).toBeGreaterThan(0.5);
    }
  });

  it("answers the noise image as (H, W) float32", async () => {
    const a = await getArray(`/runs/${rid}/image/noise`);
    expect([a.shape, a.dtype]).toEqual([[64, 64], "float32"]);
  });
});
