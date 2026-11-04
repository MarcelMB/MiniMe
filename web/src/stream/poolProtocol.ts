// Messages between the run's worker (`stream.worker.ts`, which owns the stages that cannot be
// split) and its frame workers (`frame.worker.ts`: one wasm instance each).
//
// A frame worker takes one message at a time: a context (`init`, `s0`, `p2ctx`: no reply but
// `ack`) or a task (`p1`, `s1`, `p2`: replies with `done` and the task's `id`). While a task runs
// it may post `progress`. Results carry `ms`, the milliseconds the task kept the worker busy, and
// `heapMb`, the worker's wasm heap.

/** Files a stage's engine is built from (`params.json`, `template.npy`, ...). */
export type FileMap = Map<string, Uint8Array>;

export type ToFrameWorker =
  | { type: "init"; avis: File[]; scratch: string }
  /** Build the streaming pass's solver (`StreamSolver`) from its inputs. */
  | { type: "s0"; files: FileMap }
  /** Build pass 2's engine for a window (`pass2.json`, `omega_*.npy`) and open its tile store, whose
   * slabs hold `chunks[c]` frames each (`slabs.ts`). */
  | { type: "p2ctx"; files: FileMap; chunks: number[]; tiles: string }
  /** The motion template's workers (`TemplateWorker`) from the stage's `params.json`. */
  | { type: "t0"; params: Uint8Array }
  /** Round 1's mean image, which round 2 aligns to. */
  | { type: "tfirst"; image: Float32Array }
  /** Prepare template frames `[a, b)` of the session, as round 1 or round 2 averages them. */
  | { type: "t1"; id: number; a: number; b: number }
  | { type: "t2"; id: number; a: number; b: number }
  /** Free the solver and pass 2's engine (their memory stays with the wasm instance, but is reusable). */
  | { type: "free" }
  /** Pass 1 on frames `[a, b)` of the window `files` describe (`params.json`, `template.npy`,
   * optionally `shifts.npy`), its tiles written to slab `chunk`. */
  | { type: "p1"; id: number; files: FileMap; a: number; b: number; chunk: number; tiles: string }
  /** Preprocess and solve frames `[a, b)` of the session (pixels cut for the fold lanes once `f0` ran). */
  | { type: "s1"; id: number; a: number; b: number }
  /** The streaming pass on this worker: its solver and fold lane `lane` of `lanes` (`StreamLane`). */
  | { type: "f0"; id: number; files: FileMap; lane: number; lanes: number }
  /** Fold `n` solved frames (in frame order, chunk `id`): this lane's pixels of them and all their traces. */
  | { type: "f1"; id: number; n: number; y8: Uint8Array; cc: Float64Array }
  /** This lane's sums. */
  | { type: "f2"; id: number }
  /** Compress tile `k` of the window's tile store. */
  | { type: "p2"; id: number; k: number };

export type P1Result = {
  a: number;
  n: number;
  maxProj: Uint8Array;
  blockMax: Uint8Array;
  shifts: Float32Array;
  shifts64: Float64Array;
  quality: Float32Array;
  /** Bytes written to the slab. */
  bytes: number;
};
/** `parts[j]`: the frames' pixels of fold lane `j`, frame after frame. */
export type S1Result = { parts: Uint8Array[]; cc: Float64Array; f: Float32Array };
/** A fold lane's range (`StreamLane.range`) and sums (`done_flat`, `seg_flat`). */
export type F2Result = { range: Uint32Array; done: Float64Array; seg: Float64Array };
export type P2Result = { factors: Uint8Array; readMs: number };
/** The prepared template frames `a..b`, one after the other. */
export type TResult = { preps: Float32Array };

type Meta = { id: number; ms: number; heapMb: number };
export type FromFrameWorker =
  | { type: "ack"; heapMb: number }
  | { type: "progress"; id: number; done: number }
  | { type: "error"; id: number; message: string }
  | ({ type: "done"; kind: "p1" } & Meta & P1Result)
  | ({ type: "done"; kind: "s1" } & Meta & S1Result)
  | ({ type: "done"; kind: "p2" } & Meta & P2Result)
  | ({ type: "done"; kind: "t1" | "t2" } & Meta & TResult)
  | ({ type: "done"; kind: "f0" | "f1" } & Meta)
  | ({ type: "done"; kind: "f2" } & Meta & F2Result);
