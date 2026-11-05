/// <reference lib="webworker" />
// A frame worker of the pool (`pool.ts`): its own wasm instance, its own AVI index, and one task at
// a time from the run's worker. It never holds more than a chunk of frames or one tile.

import { forFrames, indexSegments, type Segment } from "./frames";
import { benchPass1, pass1Gpu, selfTest, streamGpu, templateGpu, WorkerGpu } from "./gpuStages";
import init, {
  CnmfEngine,
  InitEngine,
  Pass1Engine,
  Pass2Engine,
  SeedsEngine,
  SpreadWorker,
  StreamLane,
  StreamSolver,
  TemplateWorker,
} from "./pkg/minime_core.js";
import type { FromFrameWorker, ToFrameWorker } from "./poolProtocol";
import { FLUSH, readTile, slabName } from "./slabs";
import { spreadFrames } from "./spread";

const post = (m: FromFrameWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);

let memory: WebAssembly.Memory | null = null;
let segs: Segment[] = [];
let scratch = "";
let solver: StreamSolver | null = null;
/** This worker's lane of the streaming pass's fold, and where every lane's pixels begin (and end). */
let lane: StreamLane | null = null;
let laneBounds: Uint32Array | null = null;
let pass2: Pass2Engine | null = null;
let template: TemplateWorker | null = null;
/** This worker's share of a spread template (`spread.ts`; on the CPU). */
let spread: SpreadWorker | null = null;
/** This worker's GPU when the run turned it on (`g0`); its per-frame tasks then run their kernels there. */
let gpu: WorkerGpu | null = null;
/** Say once in the run's log that this worker's GPU went off, and why. */
let offSaid = false;
const sayOff = () => {
  if (gpu?.off && !offSaid) {
    offSaid = true;
    post({ type: "note", message: `a frame worker's GPU went off (${gpu.off}); its frames go through the CPU path` });
  }
};
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
    case "u0": {
      spread?.free();
      spread = new SpreadWorker(m.params);
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "utmpl": {
      if (!spread) throw new Error("the spread template worker is not built");
      spread.set_template(m.image, m.search);
      return post({ type: "ack", heapMb: heapMb() });
    }
    case "u1": {
      const began = performance.now();
      if (!spread) throw new Error("the spread template worker is not built");
      const r = spreadFrames(spread, segs, m);
      const bufs = [r.preps, r.est, r.fill].filter((b) => b !== undefined).map((b) => b.buffer);
      return post({ type: "done", kind: "u1", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r }, bufs);
    }
    case "f0": {
      const began = performance.now();
      solver?.free();
      lane?.free();
      solver = new StreamSolver(m.files);
      lane = new StreamLane(m.files, m.lane, m.lanes);
      laneBounds = lane.pixel_bounds();
      return post({ type: "done", kind: "f0", id: m.id, ms: performance.now() - began, heapMb: heapMb() });
    }
    case "f1": {
      const began = performance.now();
      if (!lane) throw new Error("the fold lane is not built");
      lane.fold(m.n, m.y8, m.cc);
      return post({ type: "done", kind: "f1", id: m.id, ms: performance.now() - began, heapMb: heapMb() });
    }
    case "f2": {
      if (!lane) throw new Error("the fold lane is not built");
      const r = { range: lane.range(), done: lane.done_flat(), seg: lane.seg_flat() };
      return post({ type: "done", kind: "f2", id: m.id, ms: 0, heapMb: heapMb(), ...r }, [
        r.range.buffer,
        r.done.buffer,
        r.seg.buffer,
      ]);
    }
    case "free": {
      solver?.free();
      pass2?.free();
      template?.free();
      spread?.free();
      lane?.free();
      solver = pass2 = template = spread = lane = null;
      laneBounds = null;
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
    case "w1":
      return windowUnits(m);
    case "g0": {
      const began = performance.now();
      const keep = m.on && gpu && !gpu.off && testHook === (m.test ?? "");
      testHook = m.test ?? "";
      if (!keep) {
        gpu?.close();
        gpu = m.on ? new WorkerGpu(testHook) : null;
      }
      offSaid = false;
      const off = gpu ? (keep ? "" : await gpu.open()) : "off";
      if (gpu && off) gpu = null;
      const r = { name: gpu?.name ?? "", off };
      return post({ type: "done", kind: "g0", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r });
    }
    case "gt": {
      const began = performance.now();
      const g = gpu ?? new WorkerGpu();
      const off = gpu ? "" : await g.open();
      const failure = off || (await selfTest(g, segs, m.params, m.at, m.shifts, testHook));
      const r = { failure, name: g.name };
      if (g !== gpu) g.close();
      return post({ type: "done", kind: "gt", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r });
    }
    case "gb": {
      const engine = new Pass1Engine(m.files);
      try {
        engine.start_at(0);
        const ms = await benchPass1(engine, gpu, segs, m.a, m.b);
        sayOff();
        const r = { frames: m.b - m.a, gpuFrames: gpu && !gpu.off ? m.b - m.a : 0 };
        return post({ type: "done", kind: "gb", id: m.id, ms, heapMb: heapMb(), ...r });
      } finally {
        engine.free();
      }
    }
  }
}

/** The tests' hook of the run (`g0`'s `test`), for the self-test too. */
let testHook = "";

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
    const tick = (done: number) => {
      const now = performance.now();
      if (now - lastPost > 500) {
        lastPost = now;
        post({ type: "progress", id: m.id, done });
      }
    };
    // the GPU path first (to the end, or to where its GPU failed), then the CPU path for the rest
    const from = gpu ? await pass1Gpu(engine, gpu, segs, m.a, m.b, flush, tick) : m.a;
    sayOff();
    forFrames(
      segs,
      t0 + from,
      t0 + m.b,
      (s) => engine.set_stride(s),
      (c) => {
        engine.push_raw(c);
        if (engine.buffered() === FLUSH) flush();
        tick(engine.done() - m.a);
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
      gpuFrames: from - m.a,
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

/** Preprocess and solve session frames `[a, b)`: on the GPU when it is on (to where it got), then the CPU. */
async function streamChunk(m: Extract<ToFrameWorker, { type: "s1" }>): Promise<void> {
  const began = performance.now();
  if (!solver) throw new Error("the streaming solver is not built");
  const s = solver;
  let lastPost = 0;
  const tick = (done: number) => {
    const now = performance.now();
    if (now - lastPost > 500) {
      lastPost = now;
      post({ type: "progress", id: m.id, done });
    }
  };
  const [H, W] = [segs[0].avi.height(), segs[0].avi.width()];
  const from = gpu ? await streamGpu(s, gpu, segs, H, W, m.a, m.b, tick) : m.a;
  sayOff();
  let t = from;
  forFrames(
    segs,
    from,
    m.b,
    (stride) => s.set_stride(stride),
    (c) => {
      s.solve_frame(t++, c);
      tick(t - m.a);
    },
  );
  const y8 = s.take_y8();
  const r = { parts: cutForLanes(y8, m.b - m.a), cc: s.take_cc(), f: s.take_f(), gpuFrames: from - m.a };
  post({ type: "done", kind: "s1", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r }, [
    ...r.parts.map((p) => p.buffer),
    r.cc.buffer,
    r.f.buffer,
  ]);
}

/** `n` frames of pixels cut into the fold lanes' pixel ranges, each frame after frame (one part, the
 * whole frames, before the lanes are built). */
function cutForLanes(y8: Uint8Array, n: number): Uint8Array[] {
  if (!laneBounds) return [y8];
  const b = laneBounds;
  const p = b[b.length - 1];
  return Array.from({ length: b.length - 1 }, (_, j) => {
    const np = b[j + 1] - b[j];
    const part = new Uint8Array(n * np);
    for (let f = 0; f < n; f++) part.set(y8.subarray(f * p + b[j], f * p + b[j + 1]), f * np);
    return part;
  });
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

/** Template frames `[a, b)` (all of segment 0): prepared for round 1 or round 2, on the GPU when it is on. */
async function templateFrames(m: Extract<ToFrameWorker, { type: "t1" | "t2" }>): Promise<void> {
  const began = performance.now();
  if (!template) throw new Error("the template worker is not built");
  const t = template;
  const preps: Float32Array[] = [];
  const from = gpu ? await templateGpu(t, gpu, segs, m.type === "t1" ? 1 : 2, m.a, m.b, preps) : m.a;
  sayOff();
  forFrames(
    segs,
    from,
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
  const r = { preps: all, gpuFrames: from - m.a };
  post({ type: "done", kind: m.type, id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r }, [all.buffer]);
}

// one message at a time, in order: a worker may be dealt several tasks up front
let queue: Promise<void> = Promise.resolve();
/** A window's seeds, init and CNMF loop, as the run's worker chains them (`runSeeds`, `runInit`,
 * `runCnmf`), on the window's pass 1 maxima and pass 2 factors. */
function windowUnits(m: Extract<ToFrameWorker, { type: "w1" }>): void {
  const began = performance.now();
  const files = new Map<string, Uint8Array>();
  const take = (out: Map<string, Uint8Array>) => {
    for (const [name, bytes] of out) files.set(name, bytes);
  };
  const p1 = new Pass1Engine(m.files);
  const p2 = new Pass2Engine(m.files);
  const engines: { free(): void }[] = [p1, p2];
  try {
    p1.set_maxima(m.maxProj, m.blockMax);
    for (const [k, b] of m.tiles.entries()) p2.set_tile(k, b);
    const seconds: number[] = [];
    const notes: string[] = [];
    let t = performance.now();
    const step = (note: string, i: number) => {
      const now = performance.now();
      seconds.push((now - t) / 1e3);
      notes.push(note);
      t = now;
      if (i < 3) post({ type: "progress", id: m.id, done: i, note, seconds: seconds[i - 1] });
    };
    post({ type: "progress", id: m.id, done: 0 });
    const seeds = new SeedsEngine(m.files, p1, p2);
    engines.push(seeds);
    const merged = seeds.run();
    step(`${merged} merged seeds`, 1);
    take(seeds.finish(JSON.stringify({ where: "browser", seeds: seconds[0], wasm_heap_mb: heapMb() })));
    const initE = new InitEngine(m.files);
    engines.push(initE);
    const units = initE.run(seeds);
    step(`${units} units after the merge`, 2);
    take(initE.finish(JSON.stringify({ where: "browser", init: seconds[1], wasm_heap_mb: heapMb() })));
    const cnmf = new CnmfEngine(m.files);
    engines.push(cnmf);
    const final = cnmf.run(seeds, initE);
    step(`${final} units`, 3);
    take(cnmf.finish(JSON.stringify({ where: "browser", cnmf: seconds[2], wasm_heap_mb: heapMb() })));
    const window = cnmf.window_bytes();
    const r = { files, window, seconds, notes };
    post({ type: "done", kind: "w1", id: m.id, ms: performance.now() - began, heapMb: heapMb(), ...r }, [
      ...[...files.values()].map((b) => b.buffer),
      window.buffer,
    ]);
  } finally {
    for (const e of engines.reverse()) e.free();
  }
}

self.onmessage = (e: MessageEvent<ToFrameWorker>) => {
  const id = "id" in e.data ? e.data.id : -1;
  queue = queue
    .then(() => handle(e.data))
    .catch((err: unknown) => {
      post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
    });
};
