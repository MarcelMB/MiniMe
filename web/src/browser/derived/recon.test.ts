import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { writeNpy } from "../npy";
import type { RunInfo } from "../runs";
import { NodeDir } from "../testing/nodeDir";
import { windowDir } from "../writer";
import { preFrame, reconFrame, windowOf } from "./recon";
import { tileGrid } from "./tiles";

let counter = 0;
/** A `RunInfo` on a fresh folder (a new id each time: the readers cache per run and stamp). */
function run(meta: Partial<RunInfo["meta"]> = {}): { r: RunInfo; path: string } {
  const path = mkdtempSync(join(tmpdir(), "recon-"));
  const r: RunInfo = {
    id: `recon-${counter++}`,
    vpath: "x",
    dir: new NodeDir(path),
    params: {},
    report: {},
    meta: { H: 64, W: 64, T: 300, tile: 32, ranks: [1, 1, 1, 1], windows: [{ t0: 0, t1: 300 }], ...meta },
    mtime: 0,
    stamp: "s",
  };
  return { r, path };
}

const put = (path: string, rel: string, bytes: Uint8Array) => {
  const full = join(path, rel);
  mkdirSync(join(full, ".."), { recursive: true });
  writeFileSync(full, bytes);
};

/** One tile's factors: `w(p, k)` (P, r) and `v(t, k)` (T, r) row-major, `mean(p)`. */
type TileSpec = {
  P: number;
  rank: number;
  w: (p: number, k: number) => number;
  v: (t: number, k: number) => number;
  mean: (p: number) => number;
};

/** The files `compress::outputs` writes for a window of `T` frames: tiles concatenated in grid order. */
function writeFactors(path: string, w: number, T: number, tiles: TileSpec[]): void {
  const [W, V, mean] = [[] as number[], [] as number[], [] as number[]];
  for (const t of tiles) {
    for (let p = 0; p < t.P; p++) for (let k = 0; k < t.rank; k++) W.push(t.w(p, k));
    for (let i = 0; i < T; i++) for (let k = 0; k < t.rank; k++) V.push(t.v(i, k));
    for (let p = 0; p < t.P; p++) mean.push(t.mean(p));
  }
  const d = `factors/${windowDir(w)}`;
  put(path, `${d}/W.npy`, writeNpy([W.length], Float32Array.from(W)));
  put(path, `${d}/V.npy`, writeNpy([V.length], Float32Array.from(V)));
  put(path, `${d}/mean.npy`, writeNpy([mean.length], Float32Array.from(mean)));
  put(path, `${d}/rank.npy`, writeNpy([tiles.length], BigInt64Array.from(tiles.map((t) => BigInt(t.rank)))));
}

describe("windowOf", () => {
  const meta = (windows?: { t0: number; t1: number }[]) => run({ T: 400, windows }).r.meta;
  const three = [
    { t0: 0, t1: 150 },
    { t0: 150, t1: 300 },
    { t0: 300, t1: 400 },
  ];

  it("gives the window of a frame and its place in the window", () => {
    expect(windowOf(meta(three), 0)).toEqual({ w: 0, local: 0 });
    expect(windowOf(meta(three), 149)).toEqual({ w: 0, local: 149 });
    expect(windowOf(meta(three), 399)).toEqual({ w: 2, local: 99 });
  });

  it("starts the next window at its t0", () => {
    expect(windowOf(meta(three), 150)).toEqual({ w: 1, local: 0 });
    expect(windowOf(meta(three), 300)).toEqual({ w: 2, local: 0 });
  });

  it("refuses a frame no window holds", () => {
    expect(() => windowOf(meta(three), 400)).toThrow("frame 400 is in no window");
    expect(() => windowOf(meta(three), -1)).toThrow("frame -1 is in no window");
  });

  it("takes a run without window records as one window over every frame", () => {
    expect(windowOf(meta(undefined), 399)).toEqual({ w: 0, local: 399 });
    expect(() => windowOf(meta(undefined), 400)).toThrow("in no window");
  });
});

describe("reconFrame", () => {
  it("is mean + W V[t] on every tile: rank 1, W = 1, V[t] = t, mean = 10 gives 13 at t = 3", async () => {
    const { r, path } = run();
    const one: TileSpec = { P: 1024, rank: 1, w: () => 1, v: (t) => t, mean: () => 10 };
    writeFactors(path, 0, 300, [one, one, one, one]);
    const f = await reconFrame(r, 3);
    expect(f).toBeInstanceOf(Float32Array);
    expect(f).toHaveLength(64 * 64);
    expect(new Set(f)).toEqual(new Set([13]));
    expect(new Set(await reconFrame(r, 299))).toEqual(new Set([309]));
  });

  it("puts every tile in its place with its own rank, over partial tiles at the edges", async () => {
    // 5 x 7 frame, 3-px tiles: six tiles of 9, 9, 3, 6, 6, 2 pixels
    const [H, W, tile, T] = [5, 7, 3, 4];
    const grid = tileGrid(H, W, tile);
    const ranks = [1, 2, 3, 1, 2, 3];
    const tiles: TileSpec[] = grid.map((g, b) => ({
      P: (g.y1 - g.y0) * (g.x1 - g.x0),
      rank: ranks[b],
      w: (p, k) => (b + 1) * 0.5 + p * 0.25 - k,
      v: (t, k) => Math.sin(1 + 0.7 * t + k + b),
      mean: (p) => 100 * b + p,
    }));
    const { r, path } = run({ H, W, T, tile, ranks, windows: [{ t0: 0, t1: T }] });
    writeFactors(path, 0, T, tiles);
    for (const t of [0, 2, 3]) {
      const f = await reconFrame(r, t);
      grid.forEach((g, b) => {
        const wd = g.x1 - g.x0;
        for (let y = g.y0; y < g.y1; y++)
          for (let x = g.x0; x < g.x1; x++) {
            const p = (y - g.y0) * wd + (x - g.x0);
            let want = tiles[b].mean(p);
            for (let k = 0; k < ranks[b]; k++) want += tiles[b].w(p, k) * tiles[b].v(t, k);
            expect(f[y * W + x], `t ${t} tile ${b} (${y}, ${x})`).toBeCloseTo(want, 4);
          }
      });
    }
  });

  it("reads frame t of its own window at its local position", async () => {
    const { r, path } = run({
      T: 7,
      windows: [
        { t0: 0, t1: 3 },
        { t0: 3, t1: 7 },
      ],
      ranks: [1, 1, 1, 1],
    });
    // window 0: V[t] = t (frames 0-2); window 1: V[t] = 100 + t (frames 3-6 are local 0-3)
    const spec = (base: number): TileSpec => ({ P: 1024, rank: 1, w: () => 1, v: (t) => base + t, mean: () => 0 });
    writeFactors(path, 0, 3, Array(4).fill(spec(0)));
    writeFactors(path, 1, 4, Array(4).fill(spec(100)));
    expect(new Set(await reconFrame(r, 2))).toEqual(new Set([2]));
    expect(new Set(await reconFrame(r, 3))).toEqual(new Set([100]));
    expect(new Set(await reconFrame(r, 6))).toEqual(new Set([103]));
    expect(new Set(await reconFrame(r, 0))).toEqual(new Set([0])); // back to the first window
  });

  it("refuses factors that do not hold the run's tiles", async () => {
    const { r, path } = run();
    const one: TileSpec = { P: 1024, rank: 1, w: () => 1, v: (t) => t, mean: () => 10 };
    writeFactors(path, 0, 300, [one, one, one]); // three tiles for a four-tile grid
    await expect(reconFrame(r, 0)).rejects.toMatchObject({ status: 500, message: expect.stringContaining("tiles") });
  });
});

/** The movie the tile stores below are cut from: pixel (y, x) of frame t. */
const px = (t: number, y: number, x: number) => (t * 31 + y * 7 + x * 3) % 251;

/** `tile_KKK.partPP` of the frames `[t0, t1)`: each tile (n, th, tw) row-major. */
function writeTiles(
  path: string,
  dir: string,
  part: number,
  t0: number,
  t1: number,
  H: number,
  W: number,
  tile: number,
) {
  tileGrid(H, W, tile).forEach((g, k) => {
    const bytes: number[] = [];
    for (let t = t0; t < t1; t++)
      for (let y = g.y0; y < g.y1; y++) for (let x = g.x0; x < g.x1; x++) bytes.push(px(t, y, x));
    put(path, `${dir}/tile_${String(k).padStart(3, "0")}.part${String(part).padStart(2, "0")}`, Uint8Array.from(bytes));
  });
}

const movie = (H: number, W: number, t: number) =>
  Float32Array.from({ length: H * W }, (_, i) => px(t, Math.floor(i / W), i % W));

describe("preFrame", () => {
  const [H, W, tile] = [5, 7, 3];

  it("reads a browser run's tile store window by window, at the frame's place in its window", async () => {
    const { r, path } = run({
      H,
      W,
      T: 6,
      tile,
      ranks: Array(6).fill(1),
      windows: [
        { t0: 0, t1: 4 },
        { t0: 4, t1: 6 },
      ],
    });
    writeTiles(path, `tiles/${windowDir(0)}`, 0, 0, 4, H, W, tile);
    writeTiles(path, `tiles/${windowDir(1)}`, 0, 4, 6, H, W, tile);
    for (const t of [0, 3, 4, 5]) expect(await preFrame(r, t), `frame ${t}`).toEqual(movie(H, W, t));
  });

  it("reads a Python run's flat tile store, global frame numbers across its part files", async () => {
    // 6 frames in two parts of 4 and 2 frames: the frame count per part is tile 0's size / 9 pixels
    const { r, path } = run({ H, W, T: 6, tile, ranks: Array(6).fill(1), windows: undefined });
    writeTiles(path, "tiles", 0, 0, 4, H, W, tile);
    writeTiles(path, "tiles", 1, 4, 6, H, W, tile);
    for (const t of [0, 3, 4, 5]) expect(await preFrame(r, t), `frame ${t}`).toEqual(movie(H, W, t));
  });

  it("reads the flat store of a run that also has window records, whatever the windows say", async () => {
    // a Python run is one stream of frames: a flat store is read by global t
    const { r, path } = run({ H, W, T: 6, tile, ranks: Array(6).fill(1), windows: [{ t0: 0, t1: 6 }] });
    writeTiles(path, "tiles", 0, 0, 6, H, W, tile);
    expect(await preFrame(r, 5)).toEqual(movie(H, W, 5));
  });

  it("says 404 'not available for this run' when neither layout is there, 500 for a store that is short", async () => {
    const { r, path } = run({ H, W, T: 6, tile, ranks: Array(6).fill(1), windows: [{ t0: 0, t1: 6 }] });
    await expect(preFrame(r, 0)).rejects.toMatchObject({
      status: 404,
      message: expect.stringContaining("pre frames is not available for this run"),
    });
    writeTiles(path, `tiles/${windowDir(0)}`, 0, 0, 4, H, W, tile); // 4 frames for a 6-frame window
    expect(await preFrame(r, 3)).toEqual(movie(H, W, 3));
    await expect(preFrame(r, 4)).rejects.toMatchObject({ status: 500, message: expect.stringContaining("4 frames") });
  });
});
