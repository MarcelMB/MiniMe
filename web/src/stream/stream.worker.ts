/// <reference lib="webworker" />

import { HandleDir, walk } from "../browser/dir";
import { checkTraceRoom } from "../browser/support";
import {
  foldMaxProj,
  keepForSession,
  ProgressLog,
  type WindowParts,
  windowDir,
  writeRun,
  writeWindow,
} from "../browser/writer";
import { readNpy } from "./npy";
import { orderAvis } from "./order";
import init, {
  AviFile,
  CnmfEngine,
  crc32,
  Engine,
  InitEngine,
  MotionEngine,
  npy_f32,
  npy_shifts,
  Pass1Engine,
  Pass2Engine,
  SeedsEngine,
  StitchEngine,
  stream_params,
  time_windows,
  window_inputs,
} from "./pkg/minime_core.js";
import { Pool, workerCount } from "./pool";
import {
  pass1Pooled,
  pass2Pooled,
  type Store,
  streamPooled,
  templatePooled,
  tileCrcs,
  tilesAsFiles,
} from "./poolStages";
import type { FromWorker, Stage, ToWorker } from "./protocol";
import { DEFAULTS, MOTION_PARAMS, sessionFor } from "./session";

const post = (m: FromWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const BLOCK = 64; // frames per File.slice read
const TRACE_DIR = "stream_out";
const SCRATCH = "scratch";
/** The run's own folder in OPFS for its working files (the pass-1 tile store, the streaming pass's
 * traces), `scratch/<id>/` with the job's id: runs in two tabs never share one, and the job manager
 * removes it when the run ends however it ends. */
let scratchId = "";
const scratchParent = async () =>
  (await navigator.storage.getDirectory()).getDirectoryHandle(SCRATCH, { create: true });
const scratchDir = async () => (await scratchParent()).getDirectoryHandle(scratchId, { create: true });
const DRAIN = 4096; // frames of traces held in wasm before they go to C.npy in OPFS

let wasmMemory: WebAssembly.Memory | null = null;
/** The run's frame workers (`pool.ts`) when it has more than one worker. */
let pool: Pool | undefined;
/** The tile store of the window just through pass 1 when the pool ran it (slabs, `slabs.ts`). */
let slabStore: Store | undefined;
const heapMb = () => (wasmMemory?.buffer.byteLength ?? 0) / 2 ** 20 + (pool?.heapMb() ?? 0);
let heapPeakMb = 0;
/** Where the running stage sits in the run ("window 2/3"); shown with every stage start. */
let stageLabel = "";
/** The stage running now, when it began, and the seconds each stage took over the run so
 * far (a per-window stage adds up over the windows), for the stage events and `report.json`. */
let current: Stage | null = null;
let currentBegan = 0;
const stageSeconds = new Map<string, number>();
/** Every stage event as a `progress.jsonl` line (one JSON object, as `minime serve` reads them),
 * written to the run folder as the run goes (`run` gives it the folder). */
let progressLog = new ProgressLog(null);
const now = () => Date.now() / 1e3;
const stageEvent = (e: Extract<FromWorker, { type: "stage" }>) => {
  const { type: _, ...event } = e;
  progressLog.add(event);
  post(e);
};
const start = (stage: Stage, frames: number, units: number, height: number, width: number) => {
  current = stage;
  currentBegan = performance.now();
  post({ type: "start", stage, label: stageLabel, frames, units, height, width });
  stageEvent({ type: "stage", stage, event: "start", t: now(), note: stageLabel || undefined });
};
/** End the running stage; returns its seconds. */
const end = (note?: string, error = false): number => {
  if (!current) return 0;
  const seconds = (performance.now() - currentBegan) / 1e3;
  const total = (stageSeconds.get(current) ?? 0) + seconds;
  stageSeconds.set(current, total);
  stageEvent({ type: "stage", stage: current, event: error ? "error" : "end", t: now(), seconds: total, note });
  current = null;
  return seconds;
};
const progress = (done: number, frames: number, seconds: number, stateMb = 0) => {
  const heap = heapMb();
  heapPeakMb = Math.max(heapPeakMb, heap);
  if (current) post({ type: "progress", stage: current, done, frames, seconds, stateMb, heapMb: heap });
};
const encode = (v: unknown) => new TextEncoder().encode(JSON.stringify(v, null, 1));
const decode = (b: Uint8Array) => JSON.parse(new TextDecoder().decode(b)) as Record<string, unknown>;

async function run({
  avis,
  inputs,
  out: given,
  outPath,
  params,
  rawDir,
  fps,
  keepStore,
  scratch,
  browserStorage,
}: ToWorker): Promise<void> {
  const tl = performance.now();
  let dest = given;
  if (outPath) {
    dest = await navigator.storage.getDirectory();
    for (const p of outPath) dest = await dest.getDirectoryHandle(p);
  }
  scratchId = scratch ?? crypto.randomUUID();
  await (await scratchParent()).removeEntry(scratchId, { recursive: true }).catch(() => undefined); // an earlier run's
  progressLog = new ProgressLog(dest ? new HandleDir(dest) : null);
  const log = (what: string) =>
    post({ type: "log", message: `${what} at ${((performance.now() - tl) / 1e3).toFixed(1)} s` });
  const wasm = await init();
  wasmMemory = wasm.memory;
  log("wasm ready");
  const files = new Map<string, Uint8Array>();
  for (const f of inputs) files.set(f.name, new Uint8Array(await f.arrayBuffer()));
  if (files.size) log(`read ${files.size} input files`);
  // AVIs alone run the whole pipeline with the default parameters; a session folder
  // (ref_stitch.py --session-inputs) overrides them and, with shifts.npy and template.npy,
  // the motion; stitch.json alone stitches ref_stitch.py's window units; an inputs folder
  // with a template is ref_pass1.py's, one without footprints ref_motion.py's
  if (files.has("session.json") || !files.size)
    return runWindows(avis, files, log, wasm.memory, tl, {
      out: dest,
      params: params ?? {},
      rawDir,
      fps,
      keepStore,
      browserStorage,
    });
  if (files.has("stitch.json")) return runStitch(files, log, wasm.memory);
  if (files.has("template.npy")) return runPass1(avis, files, log, wasm.memory);
  if (!files.has("A_data.npy")) return runMotion(avis, files, log, wasm.memory);
  const { files: out, seconds } = await streamPass(avis, files, log, wasm.memory);
  post({ type: "done", files: out, seconds }, transferable(out));
}

const transferable = (out: Map<string, Uint8Array | Blob>) =>
  [...out.values()].flatMap((v) => (v instanceof Uint8Array ? [v.buffer] : []));

/** The streaming pass (`minime-core`) with fixed footprints over every frame of `avis`, then
 * one footprint update and merge from its sums. */
async function streamPass(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
): Promise<{ files: Map<string, Uint8Array | Blob>; seconds: number }> {
  const engine = new Engine(files);
  log("engine built");
  const [T, H, W] = [engine.frames(), engine.height(), engine.width()];
  const K = engine.units();
  // traces go to C.npy ((K, T) float32) in OPFS as the pass goes: they grow with T
  const root = await scratchDir();
  await root.removeEntry(TRACE_DIR, { recursive: true }).catch(() => undefined);
  const cFile = await (await root.getDirectoryHandle(TRACE_DIR, { create: true })).getFileHandle("C.npy", {
    create: true,
  });
  const cOut = await cFile.createSyncAccessHandle();
  const header = engine.traces_header();
  cOut.truncate(header.length + 4 * K * T);
  cOut.write(header, { at: 0 });
  start("stream", T, K, H, W);
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
  // with a pool the workers solve the frames and this thread folds them in, in frame order
  if (pool)
    await streamPooled(
      pool,
      engine,
      files,
      (n) => {
        t = n;
        if (t - written >= DRAIN) drain();
      },
      { progress, log },
    );
  for (const file of pool ? [] : orderAvis(avis)) {
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
        progress(t, T, (now - t0) / 1e3, engine.state_mb());
      }
    }
    avi.free();
  }
  if (t !== T) throw new Error(`read ${t} frames, inputs say ${T}`);
  drain();
  cOut.flush();
  cOut.close();
  const passSeconds = (performance.now() - t0) / 1e3;
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`pass done, wasm heap ${mb.toFixed(0)} MB`);
  const timing = { pass: passSeconds, where: "browser", wasm_heap_mb: mb };
  const out: Map<string, Uint8Array | Blob> = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  out.set("C.npy", await cFile.getFile());
  const summary = decode(out.get("summary.json") as Uint8Array);
  end(
    `${T} frames, ${(T / passSeconds).toFixed(0)} fps; ${String(summary.kept)} units kept, ${String(summary.merged)} after the merge`,
  );
  return { files: out, seconds: (performance.now() - t0) / 1e3 };
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
type WindowOpts = {
  crc: boolean;
  /** The window's tile store folder in the scratch folder (default `TILE_DIR`). */
  tiles?: string;
  /** The window's pass 1, started ahead on the pool (`beginPass1`). */
  ahead?: Ahead;
  onPass2?: (engine: Pass2Engine) => void;
  onCnmf?: (engine: CnmfEngine) => void;
};
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

/** Motion stage (`minime-core motion`, `ref_motion.py`'s inputs): session template, then
 * shift and quality of `[t0, t1)`. */
async function runMotion(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
): Promise<void> {
  const segs = indexSegments(avis);
  const { files: out, seconds } = await motionStage(segs, files, log, memory);
  for (const s of segs) s.avi.free();
  post(
    { type: "done", files: out, seconds },
    [...out.values()].map((v) => v.buffer),
  );
}

/** The template from the first `template_frames` frames of segment 0 (read twice), then the
 * shift and quality of every frame of `params.json`'s `[t0, t1)`. Returns the engine's
 * outputs (`template.npy`, `shifts.npy` ((n, 2) float64), `quality.npy`, `summary.json`). */
async function motionStage(
  segs: Segment[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
): Promise<Out> {
  if (!segs.length) throw new Error("no AVIs");
  const engine = new MotionEngine(files, segs[0].avi.n_frames());
  const [H, W, t0, t1, n] = [engine.height(), engine.width(), engine.t0(), engine.t1(), engine.template_n()];
  for (const { file, avi } of segs)
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
  log(`${segs.length} files indexed; template from ${n} frames (read twice), then frames ${t0}-${t1}`);
  const total = 2 * n + (t1 - t0);
  start("motion", total, 0, H, W);
  const began = performance.now();
  let done = 0;
  let lastPost = 0;
  const tick = () => {
    done++;
    const now = performance.now();
    if (now - lastPost > 500) {
      lastPost = now;
      progress(done, total, (now - began) / 1e3);
    }
  };
  const setStride = (s: number) => engine.set_stride(s);
  // with a pool its workers prepare the template frames and the engine adds them in frame order
  if (pool)
    await templatePooled(pool, engine, files.get("params.json") as Uint8Array, n, total, began, { progress, log });
  else
    for (let round = 0; round < 2; round++)
      forFrames(segs, 0, n, setStride, (c) => {
        engine.push_template_raw(c);
        tick();
      });
  engine.build_template();
  const tTemplate = (performance.now() - began) / 1e3;
  log("template built");
  forFrames(segs, t0, t1, setStride, (c) => {
    engine.push_raw(c);
    tick();
  });
  const seconds = (performance.now() - began) / 1e3;
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`motion done, wasm heap ${mb.toFixed(0)} MB`);
  const timing = { where: "browser", template: tTemplate, total: seconds, wasm_heap_mb: mb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  const maxShift = Number(decode(out.get("summary.json") as Uint8Array).max_abs_shift);
  end(
    t1 > t0
      ? `${t1 - t0} frames, ${((t1 - t0) / (seconds - tTemplate)).toFixed(0)} fps; largest shift ${maxShift.toFixed(1)} px`
      : `template from ${n} frames`,
  );
  return { files: out, seconds };
}

const TILE_DIR = "pass1_tiles";
/** Window `i`'s tile store folder: two windows' stores exist at once when the next window's pass 1 runs ahead. */
const tilesDir = (i: number) => `${TILE_DIR}_${i}`;
const FLUSH = 100; // frames buffered before they are appended to the tile files

/** Copy window `i`'s tile store (`TILE_DIR` in the scratch folder, one `tile_KKK.part00` per tile) into the run
 * folder's `tiles/wNN/`, file by file as a stream: a window's store is `frames * H * W` bytes, too
 * much to hold in memory twice. Called before the store is removed for the next window. */
async function keepTiles(root: FileSystemDirectoryHandle, to: HandleDir, i: number): Promise<void> {
  const from = await root.getDirectoryHandle(tilesDir(i));
  const dest = (await walk(to, `tiles/${windowDir(i)}`, { create: true })) as HandleDir;
  if (slabStore) {
    // a pooled run's store is slabs: write the tiles as the one-worker files
    await tilesAsFiles(from, slabStore, async (name, bytes) => {
      const out = await (await dest.handle.getFileHandle(name, { create: true })).createWritable();
      await out.write(bytes);
      await out.close();
    });
    return;
  }
  // biome-ignore lint/suspicious/noExplicitAny: entries() is missing from TS 5.6's DOM lib
  for await (const [name, h] of (from as any).entries() as AsyncIterable<[string, FileSystemHandle]>) {
    if (h.kind !== "file" || !name.endsWith(".part00")) continue;
    const file = await (h as FileSystemFileHandle).getFile();
    const out = await (await dest.handle.getFileHandle(name, { create: true })).createWritable();
    await file.stream().pipeTo(out); // closes `out` when the file is through
  }
}

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

/** What a run folder needs besides the pipeline's outputs (`ToWorker`'s `out`, `params`, `rawDir`,
 * `fps` and `keepStore`). */
type RunOpts = {
  out?: FileSystemDirectoryHandle;
  params: Record<string, unknown>;
  rawDir?: string;
  fps?: number | null;
  keepStore?: boolean;
  browserStorage?: boolean;
};

/** The whole pipeline on the session: its motion (unless the inputs hold `shifts.npy` and
 * `template.npy`, the session's own), then every window (`session.json`'s window length and
 * stage parameters, or the given parameters over MiniMe's defaults), pass 1 to the CNMF loop
 * each, keeping only each window's final units (its tile store is cleared before the next
 * window); then the stitch and, with a `stream` object (the default has one), the streaming
 * pass on the stitched footprints. `omega_<t0>_<k>.npy` files replay numpy's draws in the
 * window starting at t0. With `opts.out` the outputs are written there as a run folder. */
async function runWindows(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  tl: number,
  opts: RunOpts,
): Promise<void> {
  const n = workerCount(opts.params);
  if (n > 1) {
    try {
      pool = await Pool.create(n, avis, scratchId);
      log(`${n} frame workers`);
    } catch (e) {
      log(`no frame workers (${e instanceof Error ? e.message : String(e)}), one worker`);
    }
  }
  try {
    return await runSession(avis, files, log, memory, tl, opts);
  } finally {
    pool?.close();
    pool = undefined;
  }
}

async function runSession(
  avis: File[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  tl: number,
  opts: RunOpts,
): Promise<void> {
  const segs = indexSegments(avis);
  if (!segs.length) throw new Error("no AVIs");
  const starts = [0];
  for (const { avi } of segs) starts.push(starts[starts.length - 1] + avi.n_frames());
  const T = starts[starts.length - 1];
  const [H, W] = [segs[0].avi.height(), segs[0].avi.width()];
  const session = files.get("session.json") ?? encode(sessionFor(H, W, opts.params));
  const sessionObj = decode(session);
  const out = new Map<string, Uint8Array>([["session.json", session]]);
  let total = 0;
  let shifts = files.get("shifts.npy");
  // the session's own motion: the template from the first frames of segment 0 is a short stage of its
  // own; every frame's shift is then estimated by its window's pass 1 (same code, same template) and
  // the windows' shifts and qualities are put together after the loop
  const ownMotion = !shifts || !files.has("template.npy");
  if (ownMotion) {
    stageLabel = "motion";
    const mp = encode({ ...(sessionObj.pass1 as Record<string, unknown>), t0: 0, t1: 0 });
    const m = await motionStage(segs, new Map([["params.json", mp]]), log, memory);
    total += m.seconds;
    const template = m.files.get("template.npy");
    const source = m.files.get("template_src.npy");
    if (!template || !source) throw new Error("the template stage gave no template");
    files = new Map(files);
    files.set("template.npy", source); // pass 1 rebuilds the stage's own motion from this
    out.set("template.npy", template);
  }
  const ownShifts: Float64Array[] = [];
  const ownQuality: Float32Array[] = [];
  const length = Number(sessionObj.window_frames);
  const flat = time_windows(T, length, Uint32Array.from(starts));
  const n = flat.length / 2;
  log(`${segs.length} files, ${T} frames: ${n} windows of about ${length} frames`);
  const stitch = new StitchEngine(new Map([["session.json", session]]));
  const root = await scratchDir();
  const windows: Record<string, unknown>[] = [];
  // with `opts.out` each window's factors go to the run folder as it finishes; only what the
  // session-wide steps at the end read stays in memory (`kept`, `maxProj`)
  const to = opts.out ? new HandleDir(opts.out) : null;
  const kept: WindowParts[] = [];
  let maxProj: Float32Array | null = null;
  /** The stage files of window `i`. */
  const windowFiles = (i: number): Map<string, Uint8Array> => {
    const [t0, t1] = [flat[2 * i], flat[2 * i + 1]];
    const wf = new Map(files);
    for (const [name, bytes] of window_inputs(session, shifts ?? new Uint8Array(0), t0, t1) as Map<string, Uint8Array>)
      wf.set(name, bytes);
    const prefix = `omega_${String(t0).padStart(6, "0")}_`;
    for (const [name, bytes] of files) if (name.startsWith(prefix)) wf.set(`omega_${name.slice(prefix.length)}`, bytes);
    return wf;
  };
  let ahead: Ahead | undefined;
  let dropStore: Promise<void> | undefined;
  for (let i = 0; i < n; i++) {
    const [t0, t1] = [flat[2 * i], flat[2 * i + 1]];
    const before = new Map(stageSeconds);
    stageLabel = `window ${i + 1}/${n}`;
    log(`window ${i + 1}/${n}: frames ${t0}-${t1}`);
    const mine = ahead;
    ahead = undefined;
    const w = await windowPass1(segs, windowFiles(i), log, memory, {
      crc: Boolean(opts.params.tile_crc), // the tile store's checksums (the e2e check), read back at some cost
      tiles: tilesDir(i),
      ahead: mine,
      onPass2: (engine) => {
        stitch.add_pass2(engine, t0, t1);
        // The next window's pass 1 starts now on the pool, under this window's seeds, init and CNMF. This
        // window's tile store is not needed any more (unless it is kept): it goes first, so that two
        // stores never take the room at once (a private window has only a couple of GB).
        if (pool && i + 1 < n && !opts.keepStore) {
          dropStore = root.removeEntry(tilesDir(i), { recursive: true }).catch(() => undefined);
          ahead = beginPass1(pool, windowFiles(i + 1), tilesDir(i + 1), log);
        }
      },
      onCnmf: (engine) => stitch.add_cnmf(engine, t0, t1),
    });
    total += w.seconds;
    if (ownMotion) {
      ownShifts.push(readNpy(w.files.get("shifts64.npy")!).data as Float64Array);
      ownQuality.push(readNpy(w.files.get("quality.npy")!).data as Float32Array);
    }
    const tag = `win_${String(i).padStart(2, "0")}_`;
    for (const [name, bytes] of w.files)
      if (name.startsWith("summary") || (opts.params.tile_crc && name === "tile_crc.npy")) out.set(tag + name, bytes);
    windows.push(windowReport(t0, t1, w.files, before));
    if (to) {
      await writeWindow(to, i, w.files);
      maxProj = foldMaxProj(maxProj, w.files);
      kept.push({ t0, t1, files: keepForSession(w.files) });
      if (opts.keepStore) await keepTiles(root, to, i);
    }
    await dropStore;
    await root.removeEntry(tilesDir(i), { recursive: true }).catch(() => undefined);
  }
  for (const sg of segs) sg.avi.free();
  stageLabel = "";
  if (ownMotion) {
    const cat = <A extends Float32Array | Float64Array>(parts: A[]): A => {
      const all = new (parts[0].constructor as new (n: number) => A)(parts.reduce((n, a) => n + a.length, 0));
      let at = 0;
      for (const a of parts) {
        all.set(a, at);
        at += a.length;
      }
      return all;
    };
    const all = cat(ownShifts);
    shifts = npy_shifts(all);
    out.set("shifts.npy", shifts);
    out.set("quality.npy", npy_f32(cat(ownQuality)));
    const maxShift = all.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
    out.set(
      "summary_motion.json",
      encode({ where: "browser", stage: "motion", frames: all.length / 2, max_abs_shift: maxShift }),
    );
  }
  const st = runStitchEngine(stitch, log, memory);
  for (const [name, bytes] of st.files) out.set(name, bytes);
  total += st.seconds;
  const res: Map<string, Uint8Array | Blob> = out;
  // with a "stream" object the streaming pass follows on the stitched footprints, the
  // session's shifts and the windows' mean image and noise
  if (sessionObj.stream) {
    // the units are known now: their traces have to fit in browser storage before the pass allocates them
    const indptr = st.files.get("A_indptr.npy");
    if (indptr) await checkTraceRoom(readNpy(indptr).shape[0] - 1, T, !!opts.browserStorage);
    const sf = new Map<string, Uint8Array>([
      ["params.json", stream_params(session, T)],
      ["shifts.npy", shifts as Uint8Array],
    ]);
    for (const name of ["A_data.npy", "A_indices.npy", "A_indptr.npy", "ymean.npy", "sn.npy"]) {
      const b = st.files.get(name);
      if (!b) throw new Error(`the stitch gave no ${name}`);
      sf.set(name, b);
    }
    const sp = await streamPass(avis, sf, log, memory);
    for (const [name, bytes] of sp.files) res.set(name, bytes);
    total += sp.seconds;
  }
  const rep = report(out, windows, T, H, W, (performance.now() - tl) / 1e3);
  res.set("report.json", encode(rep));
  if (opts.out && maxProj) {
    const parts = { H, W, T, windows: kept, maxProj, stitch: st.files, stream: res, report: rep };
    // the traces now sit in the run folder; the scratch copy goes
    res.set("C.npy", await saveRun(opts.out, opts, sessionObj, shifts as Uint8Array, out, parts));
  }
  post({ type: "done", files: res, seconds: total }, transferable(res));
}

type RunData = {
  H: number;
  W: number;
  T: number;
  windows: WindowParts[];
  maxProj: Float32Array;
  stitch: Map<string, Uint8Array>;
  stream: Map<string, Uint8Array | Blob>;
  report: Record<string, unknown>;
};

/** Write the run folder into `dir` (`browser/writer.ts`), then delete the run's scratch folder
 * in OPFS; returns `cnmf/C.npy` of the run folder. `motion` holds the motion stage's outputs
 * (`template.npy`, `quality.npy`), which only a run that estimated its own motion has. */
async function saveRun(
  dir: FileSystemDirectoryHandle,
  opts: RunOpts,
  session: Record<string, unknown>,
  shifts: Uint8Array,
  motion: Map<string, Uint8Array>,
  d: RunData,
): Promise<File> {
  const bytes = (name: string) => {
    const b = name === "shifts.npy" ? shifts : motion.get(name);
    if (!b)
      throw new Error(`a run folder needs ${name}: run from the AVIs alone, not with an inputs folder's own motion`);
    return b;
  };
  const traces = d.stream.get("C.npy");
  if (!(traces instanceof Blob))
    throw new Error("a run folder needs the streaming pass: the session has no stream object");
  const pass1 = session.pass1 as Record<string, unknown>;
  const to = new HandleDir(dir);
  await progressLog.close(); // writeRun writes it once more, then report.json last
  await writeRun(to, {
    session,
    params: { ...DEFAULTS, ...opts.params, raw_dir: opts.rawDir, fps: opts.fps ?? undefined },
    report: d.report,
    H: d.H,
    W: d.W,
    T: d.T,
    tile: Number(pass1.tile),
    windows: d.windows,
    maxProj: d.maxProj,
    motion: {
      shifts: bytes("shifts.npy"),
      template: bytes("template.npy"),
      quality: bytes("quality.npy"),
      params: Object.fromEntries(MOTION_PARAMS.map((k) => [k, pass1[k]])),
    },
    stitch: d.stitch,
    stream: d.stream,
    traces,
    progress: progressLog.lines,
  });
  await (await scratchParent()).removeEntry(scratchId, { recursive: true }).catch(() => undefined);
  return (await to.dir("cnmf")).file("C.npy");
}

/** One window's entry of `report.json`: its frames, the seconds of each of its stages
 * (`before`: the run's stage totals when it began) and its unit counts. */
function windowReport(
  t0: number,
  t1: number,
  files: Map<string, Uint8Array>,
  before: Map<string, number>,
): Record<string, unknown> {
  const stages: Record<string, number> = {};
  for (const [stage, seconds] of stageSeconds) {
    const d = seconds - (before.get(stage) ?? 0);
    if (d > 0) stages[stage] = d;
  }
  const get = (name: string) => (files.has(name) ? decode(files.get(name)!) : {});
  const [seeds, init, cnmf, pass2] = [
    get("summary_seeds.json"),
    get("summary_init.json"),
    get("summary_cnmf.json"),
    get("summary_pass2.json"),
  ];
  return {
    t0,
    t1,
    stages,
    counts: {
      rank_total: pass2.rank_total,
      seeds_init: seeds.seeds_init,
      seeds_pnr: seeds.seeds_pnr,
      seeds_merged: seeds.seeds_merged,
      units_initA: init.units_initA,
      units_init_merge: init.units_init_merge,
      units_final: cnmf.units_final,
    },
  };
}

/** `report.json` of a whole-pipeline run, with MiniMe's `report.json` fields where they
 * exist (`stages`, `counts`, `motion`, `wall_total`, `workers`) plus the windows. */
function report(
  out: Map<string, Uint8Array>,
  windows: Record<string, unknown>[],
  T: number,
  H: number,
  W: number,
  wallTotal: number,
): Record<string, unknown> {
  const get = (name: string) => (out.has(name) ? decode(out.get(name)!) : {});
  const [motion, stitch, stream] = [get("summary_motion.json"), get("summary_stitch.json"), get("summary.json")];
  const quality = out.has("quality.npy") ? Array.from(readNpy(out.get("quality.npy")!).data as Float32Array) : null;
  const median = (v: number[]) => {
    const s = [...v].sort((a, b) => a - b);
    return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : null;
  };
  // every window finds its own seeds: the run's seed counts (the Overview's funnel, as MiniMe's
  // `counts` has them) are the sums over the windows; the stitch then joins the window units
  const sum = (key: string) => windows.reduce((n, w) => n + Number((w.counts as Record<string, unknown>)[key] ?? 0), 0);
  return {
    stages: [...stageSeconds].map(([stage, seconds]) => ({ stage, seconds })),
    counts: {
      seeds_init: sum("seeds_init"),
      seeds_pnr: sum("seeds_pnr"),
      seeds_merged: sum("seeds_merged"),
      windows: windows.length,
      units_windows: stitch.units_in,
      units_stitched: stitch.units_out,
      units_kept: stream.kept,
      units_merged: stream.merged ?? null,
      // the final units are the stitched ones: cnmf/A.npz and C.npy hold them (the streaming
      // pass's update and merge is cnmf/Amerged.npz)
      units_final: stitch.units_out,
    },
    motion: {
      sidecar_reused: !out.has("summary_motion.json"),
      quality_median: quality ? median(quality) : null,
      max_abs_shift_px: motion.max_abs_shift ?? null,
    },
    frames: T,
    H,
    W,
    windows,
    pass1_cpu_seconds: pool ? pool.busyMs.p1 / 1e3 : (stageSeconds.get("pass1") ?? null), // one thread: the wall time
    wasm_heap_peak_mb: heapPeakMb,
    wall_total: wallTotal,
    workers: pool?.size ?? 1,
    where: "browser",
  };
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
  start("stitch", 1, 0, 0, 0);
  const began = performance.now();
  const units = engine.run();
  const seconds = (performance.now() - began) / 1e3;
  progress(1, 1, seconds);
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`stitch done, ${units} units, wasm heap ${mb.toFixed(0)} MB`);
  end(`${units} units`);
  const out = engine.finish(JSON.stringify({ where: "browser", stitch: seconds, wasm_heap_mb: mb })) as Map<
    string,
    Uint8Array
  >;
  engine.free();
  return { files: out, seconds };
}

/** A window's pass 1 started ahead on the pool: its engine, the chunks' result, when it began and where
 * its progress goes once the window's turn comes. */
type Ahead = { engine: Pass1Engine; done: Promise<Store>; began: number; sink: { progress?: typeof progress } };

/** Start a window's pass 1 on the pool without waiting for it: the chunks are dealt to the workers
 * now, so they go on while this thread is busy with the previous window's seeds, init and CNMF. */
function beginPass1(p: Pool, files: Map<string, Uint8Array>, tiles: string, log: (what: string) => void): Ahead {
  const engine = new Pass1Engine(files);
  const sink: Ahead["sink"] = {};
  const ctx = { progress: (...a: Parameters<typeof progress>) => sink.progress?.(...a), log };
  const done = pass1Pooled(p, engine, files, engine.t1() - engine.t0(), ctx, tiles, true);
  done.catch(() => undefined); // an error surfaces where the window waits for it
  return { engine, done, began: performance.now(), sink };
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
  const tiles = opts.tiles ?? TILE_DIR;
  const engine = opts.ahead?.engine ?? new Pass1Engine(files);
  const [H, W, t0, t1, nTiles] = [engine.height(), engine.width(), engine.t0(), engine.t1(), engine.n_tiles()];
  for (const { file, avi } of segs)
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
  const root = await scratchDir();
  if (!opts.ahead) await root.removeEntry(tiles, { recursive: true }).catch(() => undefined);
  const dir = await root.getDirectoryHandle(tiles, { create: true });
  const handles: FileSystemSyncAccessHandle[] = [];
  // with a pool each worker writes its own parts of the tile files, so none is open here
  for (let k = 0; k < (pool ? 0 : nTiles); k++) {
    const name = `tile_${String(k).padStart(3, "0")}.part00`;
    handles.push(await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle());
  }
  const tileBytes = Array.from({ length: nTiles }, (_, k) => engine.tile_pixels(k));
  log(`${segs.length} files indexed; ${nTiles} tile files in OPFS; frames ${t0}-${t1}`);
  start("pass1", t1 - t0, 0, H, W);
  const began = performance.now();
  // a pass 1 that ran ahead began when its chunks were dealt: the stage row counts from there, while the
  // seconds the run adds up (`passSeconds`) only count from now, as the time before overlapped another stage
  const stageBegan = opts.ahead?.began ?? began;
  currentBegan = stageBegan;
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
  let pooled: Store = { tiles, frames: [], pixels: [], bytes: 0 };
  slabStore = undefined;
  if (pool) {
    if (opts.ahead) opts.ahead.sink.progress = progress;
    slabStore = pooled = await (opts.ahead?.done ??
      pass1Pooled(pool, engine, files, t1 - t0, { progress, log }, tiles));
  } else {
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
          progress(engine.done(), t1 - t0, (now - began) / 1e3);
        }
      },
    );
    flush();
  }
  const passSeconds = (performance.now() - began) / 1e3;
  const s = performance.now();
  const crc = new Uint32Array(opts.crc ? nTiles : 0);
  let storeBytes = 0;
  for (let k = 0; k < handles.length; k++) {
    const h = handles[k];
    storeBytes += h.getSize();
    if (!opts.crc) continue;
    const buf = new Uint8Array(h.getSize());
    h.read(buf, { at: 0 });
    crc[k] = crc32(buf, 0);
  }
  if (pool) {
    storeBytes = pooled.bytes;
    if (opts.crc) crc.set(await tileCrcs(dir, pooled, crc32));
  }
  const tCrc = (performance.now() - s) / 1e3;
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`pass 1 done, ${(storeBytes / 2 ** 20).toFixed(0)} MB in OPFS, wasm heap ${mb.toFixed(0)} MB`);
  end(
    `${t1 - t0} frames, ${((t1 - t0) / ((performance.now() - stageBegan) / 1e3)).toFixed(0)} fps; ${(storeBytes / 2 ** 20).toFixed(0)} MB of tiles`,
  );
  const p2 = files.has("pass2.json") ? await runPass2(handles, files, log, memory, engine, opts, pooled) : null;
  for (const h of handles) h.close();
  const timing = {
    where: "browser",
    pass: passSeconds,
    write_tiles: tWrite / 1e3,
    read_back_crc: tCrc,
    store_mb: storeBytes / 2 ** 20,
    wasm_heap_mb: mb,
  };
  const out = engine.finish(JSON.stringify(timing), crc) as Map<string, Uint8Array>;
  engine.free();
  if (p2) for (const [name, bytes] of p2.files) out.set(name, bytes);
  return { files: out, seconds: passSeconds + (p2?.seconds ?? 0) };
}

/** Pass 2 (`minime-core pass2`) on pass 1's open tile files, one tile at a time; then the
 * seed stage when the inputs hold `seeds.json` (and init when they hold `init.json`). */
async function runPass2(
  handles: FileSystemSyncAccessHandle[],
  files: Map<string, Uint8Array>,
  log: (what: string) => void,
  memory: WebAssembly.Memory,
  pass1: Pass1Engine,
  opts: WindowOpts,
  store: Store,
): Promise<Out> {
  const engine = new Pass2Engine(files);
  const n = engine.n_tiles();
  if (!pool && n !== handles.length) throw new Error(`pass2.json has ${n} tiles, the tile store ${handles.length}`);
  log(`pass 2 on ${n} tiles, ${engine.replayed() ? "replaying numpy's draws" : "own draws"}`);
  start("pass2", n, 0, 0, 0);
  const began = performance.now();
  let [tRead, rankTotal] = [0, 0];
  if (pool) {
    const r = await pass2Pooled(pool, engine, files, store, { progress, log });
    [tRead, rankTotal] = [r.readMs, r.rankTotal];
  } else
    for (let k = 0; k < n; k++) {
      const s = performance.now();
      const buf = new Uint8Array(handles[k].getSize());
      handles[k].read(buf, { at: 0 });
      tRead += performance.now() - s;
      rankTotal += engine.compress(buf);
      progress(k + 1, n, (performance.now() - began) / 1e3);
    }
  const seconds = (performance.now() - began) / 1e3;
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`pass 2 done, ${rankTotal} components, wasm heap ${mb.toFixed(0)} MB`);
  end(`${n} tiles, ${(n / seconds).toFixed(1)} tiles/s; ${rankTotal} components`);
  const timing = { where: "browser", pass2: seconds, read_tiles: tRead / 1e3, wasm_heap_mb: mb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  opts.onPass2?.(engine);
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
  start("seeds", 1, 0, 0, 0);
  const began = performance.now();
  const engine = new SeedsEngine(files, pass1, pass2);
  const tBuild = (performance.now() - began) / 1e3;
  const merged = engine.run();
  const seconds = (performance.now() - began) / 1e3;
  progress(1, 1, seconds);
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`seeds done, ${merged} merged seeds, wasm heap ${mb.toFixed(0)} MB`);
  end(`${merged} merged seeds`);
  const timing = { where: "browser", seeds: seconds, build_factors: tBuild, wasm_heap_mb: mb };
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
  start("init", 1, 0, 0, 0);
  const began = performance.now();
  const engine = new InitEngine(files);
  const units = engine.run(seeds);
  const seconds = (performance.now() - began) / 1e3;
  progress(1, 1, seconds);
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`init done, ${units} units after the merge, wasm heap ${mb.toFixed(0)} MB`);
  end(`${units} units after the merge`);
  const timing = { where: "browser", init: seconds, wasm_heap_mb: mb };
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
  start("cnmf", 1, 0, 0, 0);
  const began = performance.now();
  const engine = new CnmfEngine(files);
  const units = engine.run(seeds, init);
  opts.onCnmf?.(engine);
  const seconds = (performance.now() - began) / 1e3;
  progress(1, 1, seconds);
  const mb = memory.buffer.byteLength / 2 ** 20;
  log(`CNMF loop done, ${units} units, wasm heap ${mb.toFixed(0)} MB`);
  end(`${units} units`);
  const timing = { where: "browser", cnmf: seconds, wasm_heap_mb: mb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  return { files: out, seconds };
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  run(e.data).catch(async (err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    end(message, true);
    await progressLog.close(); // the failed stage reaches the run folder before the worker is stopped
    post({ type: "error", message });
  });
};
