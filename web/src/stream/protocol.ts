import type { Device, SpeedTest } from "../gpu/plan";
import type { ProgressEvent, ProgressUnit } from "../launch/params";

/** Messages between the stream page and its worker. With `out` (a writable folder) the whole
 * pipeline writes its run folder there (`browser/writer.ts`); without it the outputs only come
 * back in `done` (the reference-folder e2e runs). `params` are the changed parameters over
 * MiniMe's defaults, `rawDir` the raw folder's path for `params.json` and `fps` the frame rate. */
export type ToWorker = {
  avis: File[];
  inputs: File[];
  out?: FileSystemDirectoryHandle;
  /** The run folder by its path in OPFS (Browser storage), in place of `out`: Safari cannot send
   * a directory handle to a worker, so the worker opens it itself. */
  outPath?: string[];
  params?: Record<string, unknown>;
  keepStore?: boolean;
  rawDir?: string;
  fps?: number | null;
  /** The run's own folder for its working files, OPFS `scratch/<scratch>/` (a job's id); a new
   * one when not given. */
  scratch?: string;
  /** The run folder is in Browser storage (OPFS): the traces take that room twice. */
  browserStorage?: boolean;
  /** Where the per-frame stages run (default auto: the fastest the speed test measured). */
  device?: Device;
  /** Only the GPU self-test and speed test on these videos (the New run page's Speed panel): `force`
   * runs the speed test again over a cached one. Answered with `speed`. */
  speedTest?: { force?: boolean };
  /** The tests' GPU hooks: "nogpu", "mismatch" (the self-test fails), "lost" / "lost:N" (each frame
   * worker's device is lost after its 3rd / N-th GPU batch). */
  gpuTest?: string;
};

/** What the speed test found (`speed`): the test (or null without a usable GPU), the self-test's
 * verdict ("pass" or what differed) and why the GPU cannot be used ("" when it can). */
export type SpeedAnswer = { test: SpeedTest | null; selfTest: string; reason: string; gpu: string; cached: boolean };

export type Stage = "stream" | "motion" | "pass1" | "pass2" | "seeds" | "init" | "cnmf" | "stitch";

/** The job page's live-view switch: while on, the worker posts previews of each stage (`live`). */
export type LiveToggle = { type: "live"; on: boolean };
/** The job page's `liveon` event: the switch's state, null when there is no switch any more (the run's
 * worker went away while the job goes on, as when a run chains its auto-tune). */
export type LiveOnEvent = { on: boolean | null };
/** A tile grid in full-resolution pixels. */
export type LiveGrid = { h: number; w: number; tile: number };
/** Stitch windows the job manager and the page keep (the newest; the final links stay too). */
export const LIVE_STITCH_WINDOWS = 8;
/** Image previews are u8, `h` x `w` after downsampling by `f`; coordinates (polygons, points,
 * centroids) stay in full-resolution pixels, the page divides them by `f`. Polygons are [x, y]
 * points; centroids and seeds are (row, col) pairs. A motion preview with `last` is a chunk's
 * final one (never throttled away: its shifts are the chunk's complete ones). A stitch window's
 * `first` is its first unit's index over all windows' units (as `edges` numbers them). */
export type LivePayload =
  | {
      panel: "motion";
      frame: Uint8Array;
      h: number;
      w: number;
      f: number;
      at: number;
      shifts: Float32Array;
      last?: boolean;
    }
  | {
      panel: "pass2";
      label: string;
      grid: LiveGrid;
      ranks: Int32Array;
      k: number;
      comps: Uint8Array[];
      th: number;
      tw: number;
      traces: Float32Array[];
    }
  | {
      panel: "seeds";
      label: string;
      image: Uint8Array;
      h: number;
      w: number;
      f: number;
      seeds: Float32Array;
      merged: Float32Array;
    }
  | {
      panel: "units";
      label: string;
      step: "init" | "cnmf";
      image: Uint8Array;
      h: number;
      w: number;
      f: number;
      polygons: number[][][][];
      count: number;
      ids: number[];
      traces: Float32Array[];
    }
  | { panel: "stitch"; window: number; first: number; polygons: number[][][][]; centroids: Float64Array }
  | { panel: "stitch"; window: -1; edges: Float64Array; units: number };
export type LivePanel = LivePayload["panel"];

export type FromWorker =
  | {
      type: "start";
      stage: Stage;
      /** Where the stage sits in the run ("window 2/3"), empty for a lone stage. */
      label: string;
      frames: number;
      units: number;
      height: number;
      width: number;
    }
  /** A stage began or ended (`progress.jsonl`'s events, as `minime serve` shows them); a
   * stage that runs once per window ends with the seconds of all its windows so far. */
  | ({ type: "stage"; note?: string } & ProgressEvent)
  /** The running stage's counters over the whole run: a stage that runs once per window counts
   * every window (`done` of `frames` units, `seconds` of its own time in all windows so far). */
  | {
      type: "progress";
      stage: Stage;
      done: number;
      frames: number;
      seconds: number;
      unit: ProgressUnit;
      stateMb: number;
      heapMb: number;
    }
  /** The run's frames and windows, once the windows are known (for the whole run's time left). */
  | { type: "plan"; frames: number; windows: number }
  // large outputs (the traces) come as an OPFS file instead of bytes
  | { type: "done"; files: Map<string, Uint8Array | Blob>; seconds: number }
  | { type: "log"; message: string }
  | ({ type: "speed" } & SpeedAnswer)
  /** A live-view preview of a stage (only while the page turned the switch on). */
  | ({ type: "live" } & LivePayload)
  | { type: "error"; message: string };
