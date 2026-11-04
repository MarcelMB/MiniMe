/// <reference lib="webworker" />
// A frame worker of the pool (`pool.ts`): its own wasm instance, its own AVI index, and one task at
// a time from the run's worker. It never holds more than a chunk of frames or one tile.

import { forFrames, indexSegments, type Segment } from "./frames";
import init, { Pass1Engine, Pass2Engine, StreamSolver } from "./pkg/minime_core.js";
import type { FromFrameWorker, ToFrameWorker } from "./poolProtocol";

const TILE_DIR = "pass1_tiles";
const FLUSH = 100; // frames buffered before they are appended to the tile parts
const post = (m: FromFrameWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);

let memory: WebAssembly.Memory | null = null;
let segs: Segment[] = [];
let scratch = "";
let solver: StreamSolver | null = null;
let pass2: Pass2Engine | null = null;
let pass2Parts = 0;

const heapMb = () => (memory?.buffer.byteLength ?? 0) / 2 ** 20;
const part = (k: number, p: number) => `tile_${String(k).padStart(3, "0")}.part${String(p).padStart(2, "0")}`;
async function tileDir(): Promise<FileSystemDirectoryHandle> {
  const root = await (await navigator.storage.getDirectory()).getDirectoryHandle("scratch", { create: true });
  return (await root.getDirectoryHandle(scratch, { create: true })).getDirectoryHandle(TILE_DIR, { create: true });
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
      pass2Parts = m.parts;
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "free": {
      solver?.free();
      pass2?.free();
      solver = pass2 = null;
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "p1":
      return pass1Chunk(m);
    case "s1":
      return streamChunk(m);
    case "p2":
      return tile(m);
  }
}

/** Pass 1 on window frames `[a, b)`: its own part of every tile file, and the chunk's maxima,
 * shifts and quality for the window's engine to fold in. */
async function pass1Chunk(m: Extract<ToFrameWorker, { type: "p1" }>): Promise<void> {
  const began = performance.now();
  const engine = new Pass1Engine(m.files);
  const [H, W, t0, nTiles] = [engine.height(), engine.width(), engine.t0(), engine.n_tiles()];
  for (const { file, avi } of segs)
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
  engine.start_at(m.a);
  const dir = await tileDir();
  const handles: FileSystemSyncAccessHandle[] = [];
  try {
    for (let k = 0; k < nTiles; k++)
      handles.push(await (await dir.getFileHandle(part(k, m.part), { create: true })).createSyncAccessHandle());
    const tileBytes = Array.from({ length: nTiles }, (_, k) => engine.tile_pixels(k));
    let bytes = 0;
    const flush = () => {
      const n = engine.buffered();
      const packed = engine.take_tiles();
      let o = 0;
      for (let k = 0; k < nTiles; k++) {
        const len = n * tileBytes[k];
        handles[k].write(packed.subarray(o, o + len), { at: handles[k].getSize() });
        o += len;
      }
      bytes += o;
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
    for (const h of handles) h.close();
    handles.length = 0;
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
    for (const h of handles) h.close();
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

/** Compress tile `k`: read its parts in order, then `compress_at`. */
async function tile(m: Extract<ToFrameWorker, { type: "p2" }>): Promise<void> {
  const began = performance.now();
  if (!pass2) throw new Error("pass 2 is not set up");
  const dir = await tileDir();
  const handles: FileSystemSyncAccessHandle[] = [];
  try {
    // the parts of tile k, in frame order
    for (let p = 0; p < pass2Parts; p++)
      handles.push(await (await dir.getFileHandle(part(m.k, p))).createSyncAccessHandle());
    const buf = new Uint8Array(handles.reduce((n, h) => n + h.getSize(), 0));
    let at = 0;
    for (const h of handles) {
      const n = h.getSize();
      h.read(buf.subarray(at, at + n), { at: 0 });
      at += n;
    }
    for (const h of handles) h.close();
    handles.length = 0;
    const readMs = performance.now() - began;
    const factors = pass2.compress_at(m.k, buf);
    post({ type: "done", kind: "p2", id: m.id, ms: performance.now() - began, heapMb: heapMb(), factors, readMs }, [
      factors.buffer,
    ]);
  } finally {
    for (const h of handles) h.close();
  }
}

self.onmessage = (e: MessageEvent<ToFrameWorker>) => {
  const id = "id" in e.data ? e.data.id : -1;
  handle(e.data).catch((err: unknown) => {
    post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  });
};
