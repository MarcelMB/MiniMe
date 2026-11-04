// The frame-parallel stages on a pool of frame workers, put together by the run's own worker so that
// every output equals the one-worker run's, bit for bit (see `rust/minime-core/src/pool.rs`):
// - pass 1: chunks of consecutive window frames, each worker writing its own part of every tile
//   file; maxima, shifts and quality fold into the window's engine;
// - pass 2: tiles, read from the parts in order;
// - the streaming pass: workers preprocess and solve chunks of frames, this thread folds the solved
//   frames into the pass's sums in frame order.

import type { Engine, Pass1Engine, Pass2Engine } from "./pkg/minime_core.js";
import type { Pool } from "./pool";
import type { FileMap } from "./poolProtocol";
import { readTile, slabName } from "./slabs";

const STREAM_CHUNK = 16; // frames per streaming task: a worker's 0.2 s, 6 MB of preprocessed pixels

/** What the pool stages need from the run's worker (its stage rows, log and scratch folder). */
export type Ctx = {
  progress: (done: number, frames: number, seconds: number, stateMb?: number) => void;
  log: (what: string) => void;
};

const pick = (files: FileMap, keep: (name: string) => boolean): FileMap =>
  new Map([...files].filter(([name]) => keep(name)));

/** Log how busy the workers were over a stage that began at `began` (`busy` ms of their time). */
function utilisation(pool: Pool, ctx: Ctx, what: string, began: number, busy: number, tasks: number): void {
  const wall = performance.now() - began;
  ctx.log(`${what}: ${tasks} tasks on ${pool.size} workers, ${((100 * busy) / (wall * pool.size)).toFixed(0)}% busy`);
}

/** Chunk starts (and the end) of a window of `t` frames for `n` workers: about two chunks per
 * worker (the slowest worker then holds up the stage less), in multiples of 100 frames, the
 * tile buffering. */
export function chunkBounds(t: number, n: number): number[] {
  const size = Math.max(100, Math.ceil(t / (2 * n) / 100) * 100);
  const bounds: number[] = [];
  for (let a = 0; a < t; a += size) bounds.push(a);
  bounds.push(t);
  return bounds;
}

/** The tile store a pooled pass 1 left: its slabs' frame counts and the tiles' sizes (`slabs.ts`). */
export type Store = { frames: number[]; pixels: number[]; bytes: number };

/** Pass 1 of a window of `t` frames on the pool: `engine` (the window's, with no frame pushed) takes
 * the chunks' results. */
export async function pass1Pooled(
  pool: Pool,
  engine: Pass1Engine,
  files: FileMap,
  t: number,
  ctx: Ctx,
): Promise<Store> {
  const bounds = chunkBounds(t, pool.size);
  const parts = bounds.length - 1;
  const stage = pick(files, (n) => n === "params.json" || n === "template.npy" || n === "shifts.npy");
  const live = new Array<number>(parts).fill(0);
  const began = performance.now();
  let bytes = 0;
  let lastPost = 0;
  const busy0 = pool.busyMs.p1;
  const report = (force = false) => {
    const now = performance.now();
    if (!force && now - lastPost < 500) return;
    lastPost = now;
    ctx.progress(
      live.reduce((a, b) => a + b, 0),
      t,
      (now - began) / 1e3,
    );
  };
  await pool.run("p1", parts, (i) => ({ type: "p1", files: stage, a: bounds[i], b: bounds[i + 1], chunk: i }), {
    onProgress: (i, done) => {
      live[i] = done;
      report();
    },
    onResult: (i, r) => {
      engine.merge_chunk(r.a, r.n, r.maxProj, r.blockMax, r.shifts, r.shifts64, r.quality);
      live[i] = r.n;
      bytes += r.bytes;
      report();
    },
  });
  report(true);
  utilisation(pool, ctx, "pass 1", began, pool.busyMs.p1 - busy0, parts);
  return {
    frames: bounds.slice(1).map((b, i) => b - bounds[i]),
    pixels: Array.from({ length: engine.n_tiles() }, (_, k) => engine.tile_pixels(k)),
    bytes,
  };
}

/** Pass 2 on the tiles of the window's tile store: the workers compress them in any order, `engine`
 * takes the factors in tile order. Returns the components in all. */
export async function pass2Pooled(
  pool: Pool,
  engine: Pass2Engine,
  files: FileMap,
  store: Store,
  ctx: Ctx,
): Promise<{ rankTotal: number; readMs: number }> {
  const n = engine.n_tiles();
  await pool.broadcast({
    type: "p2ctx",
    files: pick(files, (f) => f === "pass2.json" || f.startsWith("omega_")),
    chunks: store.frames,
  });
  const began = performance.now();
  const busy0 = pool.busyMs.p2;
  let rankTotal = 0;
  let readMs = 0;
  await pool.run("p2", n, (k) => ({ type: "p2", k }), {
    ordered: true,
    onResult: (k, r) => {
      rankTotal += engine.set_tile(k, r.factors);
      readMs += r.readMs;
      ctx.progress(k + 1, n, (performance.now() - began) / 1e3);
    },
  });
  utilisation(pool, ctx, "pass 2", began, pool.busyMs.p2 - busy0, n);
  await pool.broadcast({ type: "free" });
  return { rankTotal, readMs };
}

/** The streaming pass with the workers' preprocessing and least squares and this thread's sums:
 * `engine` is the pass (`Engine`, built from the same `files`); `after(t)` is called with the frames
 * folded in so far after every chunk (the caller drains the traces to C.npy). */
export async function streamPooled(
  pool: Pool,
  engine: Engine,
  files: FileMap,
  after: (t: number) => void,
  ctx: Ctx,
): Promise<void> {
  const T = engine.frames();
  await pool.broadcast({ type: "s0", files });
  const chunks = Math.ceil(T / STREAM_CHUNK);
  const began = performance.now();
  const busy0 = pool.busyMs.s1;
  let t = 0;
  let lastPost = 0;
  const live = new Map<number, number>();
  await pool.run("s1", chunks, (i) => ({ type: "s1", a: i * STREAM_CHUNK, b: Math.min(T, (i + 1) * STREAM_CHUNK) }), {
    ordered: true,
    maxAhead: 3 * pool.size,
    onProgress: (i, done) => live.set(i, done),
    onResult: (i, r) => {
      live.delete(i);
      engine.push_solved(r.y8, r.cc, r.f);
      t += r.f.length;
      after(t);
      const now = performance.now();
      if (now - lastPost > 500) {
        lastPost = now;
        let running = 0;
        for (const d of live.values()) running += d;
        ctx.progress(t + running, T, (now - began) / 1e3, engine.state_mb());
      }
    },
  });
  utilisation(pool, ctx, "streaming pass", began, pool.busyMs.s1 - busy0, chunks);
  await pool.broadcast({ type: "free" });
}

/** The slabs of a pooled tile store, as read-only snapshots (pass 1 has closed them). */
async function slabFiles(dir: FileSystemDirectoryHandle, store: Store): Promise<File[]> {
  return Promise.all(store.frames.map(async (_, c) => (await dir.getFileHandle(slabName(c))).getFile()));
}

/** CRC-32 of every tile of a pooled store as one file (the pass 1 check). */
export async function tileCrcs(
  dir: FileSystemDirectoryHandle,
  store: Store,
  crc32: (bytes: Uint8Array, crc: number) => number,
): Promise<Uint32Array> {
  const slabs = await slabFiles(dir, store);
  const reader = new FileReaderSync();
  return Uint32Array.from({ length: store.pixels.length }, (_, k) =>
    crc32(readTile(slabs, store.frames, store.pixels, k, reader), 0),
  );
}

/** Hand the pooled store's tiles to `write(name, bytes)` as the `tile_KKK.part00` files a one-worker run keeps. */
export async function tilesAsFiles(
  dir: FileSystemDirectoryHandle,
  store: Store,
  write: (name: string, bytes: Uint8Array) => Promise<void>,
): Promise<void> {
  const slabs = await slabFiles(dir, store);
  const reader = new FileReaderSync();
  for (let k = 0; k < store.pixels.length; k++)
    await write(`tile_${String(k).padStart(3, "0")}.part00`, readTile(slabs, store.frames, store.pixels, k, reader));
}
