/// <reference lib="webworker" />
// A frame worker of the pool (`pool.ts`): its own wasm instance, its own AVI index, and one task at
// a time from the run's worker. It never holds more than a chunk of frames or one tile.

import { forFrames, indexSegments, type Segment } from "./frames";
import init, { Pass1Engine, Pass2Engine, StreamSolver, TemplateWorker } from "./pkg/minime_core.js";
import type { FromFrameWorker, ToFrameWorker } from "./poolProtocol";
import { FLUSH, readTile, slabName } from "./slabs";

const post = (m: FromFrameWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);

let memory: WebAssembly.Memory | null = null;
let segs: Segment[] = [];
let scratch = "";
let solver: StreamSolver | null = null;
let pass2: Pass2Engine | null = null;
let template: TemplateWorker | null = null;
/** Pass 2's tile store: the slab snapshots, frames per chunk and the tiles' sizes. */
let store: { slabs: File[]; frames: number[]; pixels: number[]; reader: FileReaderSync } | null = null;

const heapMb = () => (memory?.buffer.byteLength ?? 0) / 2 ** 20;
async function tileDir(name: string): Promise<FileSystemDirectoryHandle> {
  const root = await (await navigator.storage.getDirectory()).getDirectoryHandle("scratch", { create: true });
  return (await root.getDirectoryHandle(scratch, { create: true })).getDirectoryHandle(name, { create: true });
}

async function handle(m: ToFrameWorker): Promise<void> {
  switch (m.type) {
    case "init": {
      memory = (await init()).memory;
      scratch = m.scratch;
      segs = indexSegments(m.avis);
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "s0": {
      solver?.free();
      solver = new StreamSolver(m.files);
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "p2ctx": {
      pass2?.free();
      pass2 = new Pass2Engine(m.files);
      const dir = await tileDir(m.tiles);
      // snapshots of the slabs, taken once pass 1 has closed them
      const slabs = await Promise.all(m.chunks.map(async (_, c) => (await dir.getFileHandle(slabName(c))).getFile()));
      const pixels = Array.from({ length: pass2.n_tiles() }, (_, k) => (pass2 as Pass2Engine).tile_pixels(k));
      store = { slabs, frames: m.chunks, pixels, reader: new FileReaderSync() };
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "t0": {
      template?.free();
      template = new TemplateWorker(m.params);
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "tfirst": {
      if (!template) throw new Error("the template worker is not built");
      template.set_first(m.image);
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "free": {
      solver?.free();
      pass2?.free();
      template?.free();
      solver = pass2 = template = null;
      store = null;
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "p1":
      return pass1Chunk(m);
    case "s1":
      return streamChunk(m);
    case "p2":
      return tile(m);
    case "t1":
    case "t2":
      return templateFrames(m);
  }
}

/** Pass 1 on window frames `[a, b)`: the chunk's slab of the tile store, and its maxima, shifts and
 * quality for the window's engine to fold in. */
async function pass1Chunk(m: Extract<ToFrameWorker, { type: "p1" }>): Promise<void> {
  const began = performance.now();
  const engine = new Pass1Engine(m.files);
  const [H, W, t0] = [engine.height(), engine.width(), engine.t0()];
  for (const { file, avi } of segs)
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
  engine.start_at(m.a);
  const dir = await tileDir(m.tiles);
  const slab = await (await dir.getFileHandle(slabName(m.chunk), { create: true })).createSyncAccessHandle();
  try {
    let bytes = 0;
    const flush = () => {
      if (!engine.buffered()) return;
      const block = engine.take_tiles();
      slab.write(block, { at: bytes });
      bytes += block.length;
    };
    let lastPost = 0;
    forFrames(
      segs,
      t0 + m.a,
      t0 + m.b,
      (s) => engine.set_stride(s),
      (c) => {
        engine.push_raw(c);
        if (engine.buffered() === FLUSH) flush();
        const now = performance.now();
        if (now - lastPost > 500) {
          lastPost = now;
          post({ type: "progress", id: m.id, done: engine.done() - m.a });
        }
      },
    );
    flush();
    slab.close();
    const r = {
      a: m.a,
      n: m.b - m.a,
      maxProj: engine.chunk_max_proj(),
      blockMax: engine.chunk_block_max(),
      shifts: engine.chunk_shifts(),
      shifts64: engine.chunk_shifts64(),
      quality: engine.chunk_quality(),
      bytes,
    };
    engine.free();
    post({ type: "done", kind: "p1", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r }, [
      r.maxProj.buffer,
      r.blockMax.buffer,
      r.shifts.buffer,
      r.shifts64.buffer,
      r.quality.buffer,
    ]);
  } finally {
    try {
      slab.close();
    } catch {
      // closed above
    }
  }
}

/** Preprocess and solve session frames `[a, b)`. */
function streamChunk(m: Extract<ToFrameWorker, { type: "s1" }>): void {
  const began = performance.now();
  if (!solver) throw new Error("the streaming solver is not built");
  const s = solver;
  let t = m.a;
  let lastPost = 0;
  forFrames(
    segs,
    m.a,
    m.b,
    (stride) => s.set_stride(stride),
    (c) => {
      s.solve_frame(t++, c);
      const now = performance.now();
      if (now - lastPost > 500) {
        lastPost = now;
        post({ type: "progress", id: m.id, done: t - m.a });
      }
    },
  );
  const r = { y8: s.take_y8(), cc: s.take_cc(), f: s.take_f() };
  post({ type: "done", kind: "s1", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r }, [
    r.y8.buffer,
    r.cc.buffer,
    r.f.buffer,
  ]);
}

/** Compress tile `k`: read its pieces of every slab in order, then `compress_at`. */
function tile(m: Extract<ToFrameWorker, { type: "p2" }>): void {
  const began = performance.now();
  if (!pass2 || !store) throw new Error("pass 2 is not set up");
  const buf = readTile(store.slabs, store.frames, store.pixels, m.k, store.reader);
  const readMs = performance.now() - began;
  const factors = pass2.compress_at(m.k, buf);
  post({ type: "done", kind: "p2", id: m.id, ms: performance.now() - began, heapMb: heapMb(), factors, readMs }, [
    factors.buffer,
  ]);
}

/** Template frames `[a, b)` (all of segment 0): prepared for round 1 or round 2. */
function templateFrames(m: Extract<ToFrameWorker, { type: "t1" | "t2" }>): void {
  const began = performance.now();
  if (!template) throw new Error("the template worker is not built");
  const t = template;
  const preps: Float32Array[] = [];
  forFrames(
    segs,
    m.a,
    m.b,
    (s) => t.set_stride(s),
    (c) => {
      preps.push(m.type === "t1" ? t.round1(c) : t.round2(c));
    },
  );
  const all = new Float32Array(preps.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of preps) {
    all.set(p, at);
    at += p.length;
  }
  post({ type: "done", kind: m.type, id: m.id, ms: performance.now() - began, heapMb: heapMb(), preps: all }, [
    all.buffer,
  ]);
}

// one message at a time, in order: a worker may be dealt several tasks up front
let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent<ToFrameWorker>) => {
  const id = "id" in e.data ? e.data.id : -1;
  queue = queue
    .then(() => handle(e.data))
    .catch((err: unknown) => {
      post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
    });
};
