/// <reference lib="webworker" />

import { orderAvis } from "./order";
import init, {
  AviFile,
  CnmfEngine,
  crc32,
  Engine,
  InitEngine,
  MotionEngine,
  Pass1Engine,
  Pass2Engine,
  SeedsEngine,
  StitchEngine,
  time_windows,
  window_inputs,
} from "./pkg/minime_core.js";
import type { FromWorker, ToWorker } from "./protocol";

const post = (m: FromWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const BLOCK = 64; // frames per File.slice read
const TRACE_DIR = "stream_out";
const DRAIN = 4096; // frames of traces held in wasm before they go to C.npy in OPFS

async function run({ avis, inputs }: ToWorker): Promise<void> {
  const tl = performance.now();
  const log = (what: string) =>
    post({ type: "log", message: `${what} at ${((performance.now() - tl) / 1e3).toFixed(1)} s` });
  const wasm = await init();
  log("wasm ready");
  const files = new Map<string, Uint8Array>();
  for (const f of inputs) files.set(f.name, new Uint8Array(await f.arrayBuffer()));
  log(`read ${files.size} input files`);
  // a session folder runs every window and the stitch (ref_stitch.py --session-inputs);
  // stitch.json alone stitches ref_stitch.py's window units; an inputs folder with a
  // template is ref_pass1.py's, one without footprints ref_motion.py's
  if (files.has("session.json")) return runWindows(avis, files, log, wasm.memory);
  if (files.has("stitch.json")) return runStitch(files, log, wasm.memory);
  if (files.has("template.npy")) return runPass1(avis, files, log, wasm.memory);
  if (!files.has("A_data.npy")) return runMotion(avis, files, log, wasm.memory);
  const engine = new Engine(files);
  log("engine built");
  const [T, H, W] = [engine.frames(), engine.height(), engine.width()];
  const K = engine.units();
  // traces go to C.npy ((K, T) float32) in OPFS as the pass goes: they grow with T
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(TRACE_DIR, { recursive: true }).catch(() => undefined);
  const cFile = await (await root.getDirectoryHandle(TRACE_DIR, { create: true })).getFileHandle("C.npy", {
    create: true,
  });
  const cOut = await cFile.createSyncAccessHandle();
  const header = engine.traces_header();
  cOut.truncate(header.length + 4 * K * T);
  cOut.write(header, { at: 0 });
  post({ type: "start", stage: "stream", frames: T, units: K, height: H, width: W });
  const reader = new FileReaderSync();
  const t0 = performance.now();
  let t = 0;
  let written = 0;
  const drain = () => {
    const n = t - written;
    if (!n) return;
    const c = engine.take_traces();
    for (let i = 0; i < K; i++)
      cOut.write(c.subarray(i * n, (i + 1) * n), { at: header.length + 4 * (i * T + written) });
    written = t;
  };
  let lastPost = 0;
  for (const file of orderAvis(avis)) {
    if (t === T) break;
    const read = (off: number, len: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(off, off + len)));
    const avi = AviFile.parse(read, file.size);
    log(`${file.name}: ${avi.n_frames()} frames indexed`);
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
    const n = avi.n_frames();
    if (n) engine.set_stride(avi.size(0) === H * W ? W : (W + 3) & ~3);
    for (let i = 0; i < n && t < T; i += BLOCK) {
      const j = Math.min(i + BLOCK, n, i + (T - t));
      const start = avi.offset(i);
      const buf = read(start, avi.offset(j - 1) + avi.size(j - 1) - start);
      for (let k = i; k < j; k++) {
        const o = avi.offset(k) - start;
        engine.push_raw(buf.subarray(o, o + avi.size(k)));
        t++;
      }
      if (t - written >= DRAIN) drain();
      const now = performance.now();
      if (now - lastPost > 500) {
        lastPost = now;
        post({ type: "progress", done: t, frames: T, seconds: (now - t0) / 1e3, stateMb: engine.state_mb() });
      }
    }
    avi.free();
  }
  if (t !== T) throw new Error(`read ${t} frames, inputs say ${T}`);
  drain();
  cOut.flush();
  cOut.close();
  const passSeconds = (performance.now() - t0) / 1e3;
  const heapMb = wasm.memory.buffer.byteLength / 2 ** 20;
  log(`pass done, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { pass: passSeconds, where: "browser", wasm_heap_mb: heapMb };
  const out: Map<string, Uint8Array | Blob> = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  out.set("C.npy", await cFile.getFile());
  post(
    { type: "done", files: out, seconds: (performance.now() - t0) / 1e3 },
    [...out.values()].flatMap((v) => (v instanceof Uint8Array ? [v.buffer] : [])),
  );
}

type Segment = { file: File; avi: AviFile; read: (off: number, len: number) => Uint8Array };

/** Index every AVI (numeric order), reading through `File.slice`. */
function indexSegments(avis: File[]): Segment[] {
  const reader = new FileReaderSync();
  return orderAvis(avis).map((file) => {
    const read = (off: number, len: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(off, off + len)));
    return { file, read, avi: AviFile.parse(read, file.size) };
  });
}

/** What a window run keeps besides its outputs: tile CRCs (the pass 1 check reads the whole
 * tile store back) and the CNMF loop's engine, handed over before it is freed. */
type WindowOpts = { crc: boolean; onCnmf?: (engine: CnmfEngine) => void };
type Out = { files: Map<string, Uint8Array>; seconds: number };

/** Calls `push` on raw frames `[from, to)` of the session, reading `BLOCK` frames per slice. */
function forFrames(
  segs: Segment[],
  from: number,
  to: number,
  setStride: (s: number) => void,
  push: (chunk: Uint8Array) => void,
): void {
  let s0 = 0;
  for (const { avi, read } of segs) {
    const n = avi.n_frames();
    const [lo, hi] = [Math.max(from - s0, 0), Math.min(to - s0, n)];
    if (lo < hi) {
      setStride(avi.size(0) === avi.height() * avi.width() ? avi.width() : (avi.width() + 3) & ~3);
      for (let i = lo; i < hi; i += BLOCK) {
        const j = Math.min(i + BLOCK, hi);
        const start = avi.offset(i);
        const buf = read(start, avi.offset(j - 1) + avi.size(j - 1) - start);
        for (let k = i; k < j; k++) {
          const o = avi.offset(k) - start;
          push(buf.subarray(o, o + avi.size(k)));
        }
      }
    }
    s0 += n;
    if (s0 >= to) return;
  }
  if (s0 < to) throw new Error(`frames ${from}-${to} requested, the files hold ${s0}`);
}

/** Motion stage (`minime-core motion`): session template, then shift and quality of `[t0, t1)`. */
function runMotion(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
): void {
  const segs = indexSegments(avis);
  if (!segs.length) throw new Error("no AVIs");
  const engine = new MotionEngine(files, segs[0].avi.n_frames());
  const [H, W, t0, t1, n] = [engine.height(), engine.width(), engine.t0(), engine.t1(), engine.template_n()];
  for (const { file, avi } of segs)
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
  log(`${segs.length} files indexed; template from ${n} frames (read twice), then frames ${t0}-${t1}`);
  const total = 2 * n + (t1 - t0);
  post({ type: "start", stage: "motion", frames: total, units: 0, height: H, width: W });
  const start = performance.now();
  let done = 0;
  let lastPost = 0;
  const tick = () => {
    done++;
    const now = performance.now();
    if (now - lastPost > 500) {
      lastPost = now;
      post({ type: "progress", done, frames: total, seconds: (now - start) / 1e3, stateMb: 0 });
    }
  };
  const setStride = (s: number) => engine.set_stride(s);
  for (let round = 0; round < 2; round++)
    forFrames(segs, 0, n, setStride, (c) => {
      engine.push_template_raw(c);
      tick();
    });
  engine.build_template();
  const tTemplate = (performance.now() - start) / 1e3;
  log(`template built`);
  forFrames(segs, t0, t1, setStride, (c) => {
    engine.push_raw(c);
    tick();
  });
  const seconds = (performance.now() - start) / 1e3;
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`motion done, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { where: "browser", template: tTemplate, total: seconds, wasm_heap_mb: heapMb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  for (const s of segs) s.avi.free();
  post(
    { type: "done", files: out, seconds },
    [...out.values()].map((v) => v.buffer),
  );
}

const TILE_DIR = "pass1_tiles";
const FLUSH = 100; // frames buffered before they are appended to the tile files

/** Pass 1 and what follows on one window (`ref_pass1.py`'s ... `ref_cnmf.py`'s inputs). */
async function runPass1(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
): Promise<void> {
  const segs = indexSegments(avis);
  const { files: out, seconds } = await windowPass1(segs, files, log, memory, { crc: true });
  for (const sg of segs) sg.avi.free();
  post(
    { type: "done", files: out, seconds },
    [...out.values()].map((v) => v.buffer),
  );
}

/** Every window of the session (`session.json`: window length and stage parameters,
 * `shifts.npy`: the session's motion, `template.npy`), pass 1 to the CNMF loop each, keeping
 * only each window's final units (its tile store is cleared before the next window); then
 * the stitch. `omega_<t0>_<k>.npy` files replay numpy's draws in the window starting at t0. */
async function runWindows(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
): Promise<void> {
  const segs = indexSegments(avis);
  const starts = [0];
  for (const { avi } of segs) starts.push(starts[starts.length - 1] + avi.n_frames());
  const T = starts[starts.length - 1];
  const session = files.get("session.json");
  const shifts = files.get("shifts.npy");
  if (!session || !shifts) throw new Error("the session folder needs session.json and shifts.npy");
  const length = Number(JSON.parse(new TextDecoder().decode(session)).window_frames);
  const flat = time_windows(T, length, Uint32Array.from(starts));
  const n = flat.length / 2;
  log(`${segs.length} files, ${T} frames: ${n} windows of about ${length} frames`);
  const stitch = new StitchEngine(files);
  const out = new Map<string, Uint8Array>();
  const root = await navigator.storage.getDirectory();
  let total = 0;
  for (let i = 0; i < n; i++) {
    const [t0, t1] = [flat[2 * i], flat[2 * i + 1]];
    const wf = new Map(files);
    for (const [name, bytes] of window_inputs(session, shifts, t0, t1) as Map<string, Uint8Array>) wf.set(name, bytes);
    const prefix = `omega_${String(t0).padStart(6, "0")}_`;
    for (const [name, bytes] of files) if (name.startsWith(prefix)) wf.set(`omega_${name.slice(prefix.length)}`, bytes);
    log(`window ${i + 1}/${n}: frames ${t0}-${t1}`);
    const w = await windowPass1(segs, wf, log, memory, {
      crc: false,
      onCnmf: (engine) => stitch.add_cnmf(engine, t0, t1),
    });
    total += w.seconds;
    const tag = `win_${String(i).padStart(2, "0")}_`;
    for (const [name, bytes] of w.files) if (name.startsWith("summary")) out.set(tag + name, bytes);
    await root.removeEntry(TILE_DIR, { recursive: true }).catch(() => undefined);
  }
  for (const sg of segs) sg.avi.free();
  const st = runStitchEngine(stitch, log, memory);
  for (const [name, bytes] of st.files) out.set(name, bytes);
  post(
    { type: "done", files: out, seconds: total + st.seconds },
    [...out.values()].map((v) => v.buffer),
  );
}

/** The stitch (`minime-core stitch`) of `ref_stitch.py`'s window units (`stitch.json`,
 * `win_*`). */
function runStitch(files: Map<string, Uint8Array>, log: (what: string) => void, memory: WebAssembly.Memory): void {
  const engine = new StitchEngine(files);
  let units = 0;
  for (let i = 0; i < engine.n_inputs(); i++) units += engine.add_inputs(files, i);
  log(`${engine.n_inputs()} windows read, ${units} units`);
  const { files: out, seconds } = runStitchEngine(engine, log, memory);
  post(
    { type: "done", files: out, seconds },
    [...out.values()].map((v) => v.buffer),
  );
}

/** Run a filled `StitchEngine`, then free it. */
function runStitchEngine(engine: StitchEngine, log: (what: string) => void, memory: WebAssembly.Memory): Out {
  post({ type: "start", stage: "stitch", frames: 1, units: 0, height: 0, width: 0 });
  const start = performance.now();
  const units = engine.run();
  const seconds = (performance.now() - start) / 1e3;
  post({ type: "progress", done: 1, frames: 1, seconds, stateMb: 0 });
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`stitch done, ${units} units, wasm heap ${heapMb.toFixed(0)} MB`);
  const out = engine.finish(JSON.stringify({ where: "browser", stitch: seconds, wasm_heap_mb: heapMb })) as Map<
    string,
    Uint8Array
  >;
  engine.free();
  return { files: out, seconds };
}

/** Pass 1 (`minime-core pass1`) on `[t0, t1)`, with the tile store in OPFS: one file per
 * tile, written through sync access handles, read back at the end for its CRC-32s (with
 * `opts.crc`). With `pass2.json` in the inputs, pass 2 follows on the same tile files (and
 * with `seeds.json` the seed stage on the factors, and so on to the CNMF loop). */
async function windowPass1(
  segs: Segment[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  opts: WindowOpts,
): Promise<Out> {
  const engine = new Pass1Engine(files);
  const [H, W, t0, t1, nTiles] = [engine.height(), engine.width(), engine.t0(), engine.t1(), engine.n_tiles()];
  for (const { file, avi } of segs)
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(TILE_DIR, { recursive: true }).catch(() => undefined);
  const dir = await root.getDirectoryHandle(TILE_DIR, { create: true });
  const handles: FileSystemSyncAccessHandle[] = [];
  for (let k = 0; k < nTiles; k++) {
    const name = `tile_${String(k).padStart(3, "0")}.part00`;
    handles.push(await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle());
  }
  const tileBytes = Array.from({ length: nTiles }, (_, k) => engine.tile_pixels(k));
  log(`${segs.length} files indexed; ${nTiles} tile files in OPFS; frames ${t0}-${t1}`);
  post({ type: "start", stage: "pass1", frames: t1 - t0, units: 0, height: H, width: W });
  const start = performance.now();
  let tWrite = 0;
  const flush = () => {
    const s = performance.now();
    const n = engine.buffered();
    const packed = engine.take_tiles();
    let o = 0;
    for (let k = 0; k < nTiles; k++) {
      const len = n * tileBytes[k];
      handles[k].write(packed.subarray(o, o + len), { at: handles[k].getSize() });
      o += len;
    }
    tWrite += performance.now() - s;
  };
  let lastPost = 0;
  forFrames(
    segs,
    t0,
    t1,
    (s) => engine.set_stride(s),
    (c) => {
      engine.push_raw(c);
      if (engine.buffered() === FLUSH) flush();
      const now = performance.now();
      if (now - lastPost > 500) {
        lastPost = now;
        post({ type: "progress", done: engine.done(), frames: t1 - t0, seconds: (now - start) / 1e3, stateMb: 0 });
      }
    },
  );
  flush();
  const passSeconds = (performance.now() - start) / 1e3;
  const s = performance.now();
  const crc = new Uint32Array(opts.crc ? nTiles : 0);
  let storeBytes = 0;
  for (let k = 0; k < nTiles; k++) {
    const h = handles[k];
    storeBytes += h.getSize();
    if (!opts.crc) continue;
    const buf = new Uint8Array(h.getSize());
    h.read(buf, { at: 0 });
    crc[k] = crc32(buf, 0);
  }
  const tCrc = (performance.now() - s) / 1e3;
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`pass 1 done, ${(storeBytes / 2 ** 20).toFixed(0)} MB in OPFS, wasm heap ${heapMb.toFixed(0)} MB`);
  const p2 = files.has("pass2.json") ? runPass2(handles, files, log, memory, engine, opts) : null;
  for (const h of handles) h.close();
  const timing = {
    where: "browser",
    pass: passSeconds,
    write_tiles: tWrite / 1e3,
    read_back_crc: tCrc,
    store_mb: storeBytes / 2 ** 20,
    wasm_heap_mb: heapMb,
  };
  const out = engine.finish(JSON.stringify(timing), crc) as Map<string, Uint8Array>;
  engine.free();
  if (p2) for (const [name, bytes] of p2.files) out.set(name, bytes);
  return { files: out, seconds: passSeconds + (p2?.seconds ?? 0) };
}

/** Pass 2 (`minime-core pass2`) on pass 1's open tile files, one tile at a time; then the
 * seed stage when the inputs hold `seeds.json` (and init when they hold `init.json`). */
function runPass2(
  handles: FileSystemSyncAccessHandle[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  pass1: Pass1Engine,
  opts: WindowOpts,
): Out {
  const engine = new Pass2Engine(files);
  const n = engine.n_tiles();
  if (n !== handles.length) throw new Error(`pass2.json has ${n} tiles, the tile store ${handles.length}`);
  log(`pass 2 on ${n} tiles, ${engine.replayed() ? "replaying numpy's draws" : "own draws"}`);
  post({ type: "start", stage: "pass2", frames: n, units: 0, height: 0, width: 0 });
  const start = performance.now();
  let [tRead, rankTotal] = [0, 0];
  for (let k = 0; k < n; k++) {
    const s = performance.now();
    const buf = new Uint8Array(handles[k].getSize());
    handles[k].read(buf, { at: 0 });
    tRead += performance.now() - s;
    rankTotal += engine.compress(buf);
    post({ type: "progress", done: k + 1, frames: n, seconds: (performance.now() - start) / 1e3, stateMb: 0 });
  }
  const seconds = (performance.now() - start) / 1e3;
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`pass 2 done, ${rankTotal} components, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { where: "browser", pass2: seconds, read_tiles: tRead / 1e3, wasm_heap_mb: heapMb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  let total = seconds;
  if (files.has("seeds.json")) {
    const s = runSeeds(files, pass1, engine, log, memory, opts);
    for (const [name, bytes] of s.files) out.set(name, bytes);
    total += s.seconds;
  }
  engine.free();
  return { files: out, seconds: total };
}

/** Seeds, PNR/GMM filter and seed merge (`minime-core seeds`) on the window's factors; then
 * init on the merged seeds when the inputs hold `init.json`. */
function runSeeds(
  files: Map<string, Uint8Array>,
  pass1: Pass1Engine,
  pass2: Pass2Engine,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  opts: WindowOpts,
): Out {
  post({ type: "start", stage: "seeds", frames: 1, units: 0, height: 0, width: 0 });
  const start = performance.now();
  const engine = new SeedsEngine(files, pass1, pass2);
  const tBuild = (performance.now() - start) / 1e3;
  const merged = engine.run();
  const seconds = (performance.now() - start) / 1e3;
  post({ type: "progress", done: 1, frames: 1, seconds, stateMb: 0 });
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`seeds done, ${merged} merged seeds, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { where: "browser", seeds: seconds, build_factors: tBuild, wasm_heap_mb: heapMb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  let total = seconds;
  if (files.has("init.json")) {
    const i = runInit(files, engine, log, memory, opts);
    for (const [name, bytes] of i.files) out.set(name, bytes);
    total += i.seconds;
  }
  engine.free();
  return { files: out, seconds: total };
}

/** initA, initC and the initial unit merge (`minime-core init`) on the merged seeds; then the
 * CNMF loop when the inputs hold `cnmf.json`. */
function runInit(
  files: Map<string, Uint8Array>,
  seeds: SeedsEngine,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  opts: WindowOpts,
): Out {
  post({ type: "start", stage: "init", frames: 1, units: 0, height: 0, width: 0 });
  const start = performance.now();
  const engine = new InitEngine(files);
  const units = engine.run(seeds);
  const seconds = (performance.now() - start) / 1e3;
  post({ type: "progress", done: 1, frames: 1, seconds, stateMb: 0 });
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`init done, ${units} units after the merge, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { where: "browser", init: seconds, wasm_heap_mb: heapMb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  let total = seconds;
  if (files.has("cnmf.json")) {
    const c = runCnmf(files, seeds, engine, log, memory, opts);
    for (const [name, bytes] of c.files) out.set(name, bytes);
    total += c.seconds;
  }
  engine.free();
  return { files: out, seconds: total };
}

/** The CNMF loop (`minime-core cnmf`: spatial update, background, traces, loop merge) on
 * init's merged units. */
function runCnmf(
  files: Map<string, Uint8Array>,
  seeds: SeedsEngine,
  init: InitEngine,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  opts: WindowOpts,
): Out {
  post({ type: "start", stage: "cnmf", frames: 1, units: 0, height: 0, width: 0 });
  const start = performance.now();
  const engine = new CnmfEngine(files);
  const units = engine.run(seeds, init);
  opts.onCnmf?.(engine);
  const seconds = (performance.now() - start) / 1e3;
  post({ type: "progress", done: 1, frames: 1, seconds, stateMb: 0 });
  const heapMb = memory.buffer.byteLength / 2 ** 20;
  log(`CNMF loop done, ${units} units, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { where: "browser", cnmf: seconds, wasm_heap_mb: heapMb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  return { files: out, seconds };
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  run(e.data).catch((err: unknown) =>
    post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
};
