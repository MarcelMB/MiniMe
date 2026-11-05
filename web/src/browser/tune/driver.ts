/** Reruns and auto-tune of a browser run (`window2.Window2Session.run`, `autotune.run.autotune`):
 * the windows on tune workers, the stitch in this thread, the streaming pass on the run worker in
 * its inputs mode, the outputs into `reruns/<name>/`. */
import type { ProgressEvent, StageProgress } from "../../launch/params";
import type { FromWorker, ToWorker } from "../../stream/protocol";
import { DEFAULTS, sessionFor } from "../../stream/session";
import type { Checks, Pairs } from "../../types";
import { type Dir, fileAt, walk } from "../dir";
import type { Folders } from "../folders";
import { readNpz } from "../npz";
import { HttpError } from "../router";
import type { Registry, RunInfo } from "../runs";
import { wasm } from "../wasm";
import { type AutotuneDoc, atEdge, type PairCounts, sweepCsv } from "./doc";
import { loadWindowFiles, windowBytes, windowsOf } from "./factors";
import {
  checkWindow2,
  DERIVED,
  derivedKeys,
  derivedParams,
  MIN_DIST_FRAC,
  norm,
  type Params,
  SEARCHED,
  TUNE_FRAMES,
  type TuneArgs,
} from "./params";
import { checkDiameter, estimateDiameter, scoreRun } from "./score";
import { KNEE_RATIO, KNEE_WINDOWS, LocalSearch, middleWindows } from "./search";
import type { Trial, TrialRunner, Values } from "./types";
import { TuneWorkers } from "./windows";
import { type RerunParts, rerunReport, writeRerun } from "./writer";

export const AUTO = "auto";
export class RerunCancelled extends Error {
  constructor() {
    super("cancelled");
  }
}
export type Progress = {
  stage(e: ProgressEvent): void;
  live(p: StageProgress): void;
  log(m: string): void;
  cancelled(): boolean;
};
export type Stitch = {
  add_window(b: Uint8Array, t0: number, t1: number): number;
  add_mean_noise(m: Uint8Array, s: Uint8Array, tile: number, t0: number, t1: number): void;
  run(): number;
  finish(json: string): Map<string, Uint8Array>;
  free(): void;
};
export type Ctx = {
  registry: Registry;
  folders: Folders;
  makeTuneWorker: () => Worker;
  makeStreamWorker: () => Worker;
  maxWorkers: number;
  scratch: string;
  stitch?: (session: Uint8Array) => Stitch;
};

const now = () => Date.now() / 1e3;
const enc = (v: unknown) => new TextEncoder().encode(JSON.stringify(v));
const dec = (b: Uint8Array) => JSON.parse(new TextDecoder().decode(b)) as Record<string, unknown>;

/** `pipeline.check_run_name` and `autotune.run.check_not_reserved`. */
export function checkRerunName(name: string): void {
  if (name === "" || name === "." || name === ".." || /[/\\]/.test(name))
    throw new HttpError(400, `invalid run name '${name}': use a single, plain directory name`);
  if (name === AUTO) throw new HttpError(400, `run name '${AUTO}' is reserved for auto-tune; pick another name`);
}

/** The run's flat parameters (`params.json` without the stage objects and bookkeeping keys). */
export function flatParams(r: RunInfo): Params {
  const skip = new Set([
    "pass1",
    "pass2",
    "seeds",
    "init",
    "cnmf",
    "stitch",
    "stream",
    "window_frames",
    "source",
    "raw_dir",
    "fps",
    "source_run",
  ]);
  return Object.fromEntries(Object.entries(r.params).filter(([k]) => !skip.has(k)));
}

/** A fresh `reruns/<name>/` (an existing one of that name is replaced, as main's does). */
export async function resetRerunDir(run: RunInfo, name: string): Promise<Dir> {
  const reruns = await walk(run.dir, "reruns", { create: true });
  await reruns.remove(name, { recursive: true }).catch(() => undefined);
  return reruns.dir(name, { create: true });
}

export class Stages {
  readonly rows: { stage: string; seconds: number }[] = [];
  readonly lines: string[] = [];
  private open: { stage: string; t: number; note?: string; n?: number } | null = null;
  constructor(
    private p: Progress,
    private out: Dir | null,
  ) {}
  private emit(e: ProgressEvent) {
    this.lines.push(JSON.stringify(e));
    this.p.stage(e);
    void this.out?.write("progress.jsonl", `${this.lines.join("\n")}\n`).catch(() => undefined);
  }
  start(stage: string, note?: string) {
    this.end();
    this.open = { stage, t: now(), note };
    this.emit({ stage, event: "start", t: this.open.t, note });
  }
  /** A note and a count for the open row, carried on its end event. */
  note(text: string, n: number) {
    if (this.open) {
      this.open.note = text;
      this.open.n = n;
    }
  }
  /** A row spanning other rows: its start event now, its end event when the returned function is
   * called; it does not use the open-row slot, so rows started in between are unaffected. */
  span(stage: string): (error?: boolean) => void {
    const t = now();
    this.emit({ stage, event: "start", t });
    return (error = false) => {
      const seconds = now() - t;
      this.rows.push({ stage, seconds });
      this.emit({ stage, event: error ? "error" : "end", t: now(), seconds, n: 0 });
    };
  }
  end(error = false, n?: number) {
    if (!this.open) return;
    const seconds = now() - this.open.t;
    const count = n ?? this.open.n;
    this.rows.push({ stage: this.open.stage, seconds });
    this.emit({
      stage: this.open.stage,
      event: error ? "error" : "end",
      t: now(),
      seconds,
      ...(count === undefined ? {} : { n: count }),
      ...(this.open.note === undefined ? {} : { note: this.open.note }),
    });
    this.open = null;
  }
}

export const checkCancel = (p: Progress) => {
  if (p.cancelled()) throw new RerunCancelled();
};

/** The streaming pass of the run worker in its inputs mode on the stitched units: `C.npy`,
 * `YrA.npy`, `b.npy`, `Amerged_*.npy`, `summary.json`. */
async function streamPass(
  ctx: Ctx,
  run: RunInfo,
  session: Uint8Array,
  stitch: Map<string, Uint8Array>,
  T: number,
  progress: Progress,
  stages: Stages,
): Promise<Map<string, Uint8Array | Blob>> {
  const core = await wasm();
  // 404 "raw videos not found", also when the raw folder itself is gone
  const raw = await ctx.registry.rawDir(run).catch(() => {
    throw new HttpError(404, "raw videos not found");
  });
  const avis = (await raw.entries()).filter((e) => e.kind === "file" && /\.avi$/i.test(e.name));
  const files: File[] = [];
  for (const e of avis) files.push(await raw.file(e.name));
  const z = await readNpz(await fileAt(run.dir, "motion/arrays.npz"));
  const shifts = z.get("shifts");
  if (!shifts) throw new HttpError(404, "the motion is not available for this run: motion/arrays.npz has no shifts");
  const inputs = new Map<string, Uint8Array>([
    ["params.json", core.stream_params(session, T)],
    ["shifts.npy", shifts],
  ]);
  for (const name of ["A_data.npy", "A_indices.npy", "A_indptr.npy", "ymean.npy", "sn.npy"]) {
    const b = stitch.get(name);
    if (!b) throw new Error(`the stitch gave no ${name}`);
    inputs.set(name, b);
  }
  return new Promise((ok, no) => {
    const w = ctx.makeStreamWorker();
    const stop = (e: unknown) => {
      w.terminate();
      no(e instanceof Error ? e : new Error(String(e)));
    };
    w.onerror = (e: ErrorEvent) => stop(new Error(e.message || "the worker script failed"));
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (progress.cancelled()) return stop(new RerunCancelled());
      if (m.type === "stage") {
        const { type: _, ...ev } = m;
        stages.lines.push(JSON.stringify(ev));
        progress.stage(ev);
        if (ev.event === "end") stages.rows.push({ stage: ev.stage, seconds: ev.seconds ?? 0 });
      } else if (m.type === "progress")
        progress.live({ stage: m.stage, done: m.done, frames: m.frames, seconds: m.seconds, unit: m.unit });
      else if (m.type === "log") progress.log(m.message);
      else if (m.type === "error") stop(new Error(m.message));
      else if (m.type === "done") {
        w.terminate();
        ok(m.files);
      }
    };
    const msg: ToWorker = {
      avis: files,
      inputs: [...inputs].map(([n, b]) => new File([b as BlobPart], n)),
      params: {},
      keepStore: false,
      rawDir: "",
      fps: null,
      scratch: ctx.scratch,
      browserStorage: false,
    };
    w.postMessage(msg);
  });
}

/** The checks the trial needs only to drive the engines: a rerun has no checks and discards the score. */
const PLACEHOLDER_CHECKS = {
  diameter: 1,
  pnr_min: 12,
  piece_frac: 0.9,
  size_lo: 0.3,
  size_hi: 3,
  dup_corr: 0.6,
  keep_dist: 0.5,
};

/** Window 2 of `run` with `params` (the changed keys over the run's) into `reruns/<name>`. With
 * `workers` (auto-tune's, already loaded, `reruns/<name>` already reset by the caller) those are
 * used and left open; otherwise the rerun loads and closes its own. */
export async function rerun(
  ctx: Ctx,
  run: RunInfo,
  params: Params,
  name: string,
  progress: Progress,
  workers?: TuneWorkers,
  shared?: Stages,
): Promise<Dir> {
  if (!(workers && name === AUTO)) checkRerunName(name); // auto-tune's own rerun may use the reserved name
  checkWindow2(params);
  const t0 = now();
  const base = flatParams(run);
  const merged = { ...base, ...params };
  const { H, W, T, tile } = run.meta;
  const session = sessionFor(H, W, merged);
  const sessionBytes = enc(session);
  const runSession = enc(sessionFor(H, W, base));
  const own = !workers;
  const out = own ? await resetRerunDir(run, name) : await walk(run.dir, `reruns/${name}`, { create: true });
  const stages = shared ?? new Stages(progress, out);
  const rows0 = stages.rows.length; // the report holds this rerun's own rows only
  await out.write(
    "params.json",
    JSON.stringify({
      ...merged,
      ...session,
      source: "browser",
      raw_dir: run.params.raw_dir,
      fps: run.params.fps ?? null,
      source_run: run.vpath,
    }),
  );
  const tw = workers ?? new TuneWorkers(ctx.makeTuneWorker, runSession, ctx.maxWorkers);
  try {
    const windows = windowsOf(run);
    stages.start("window2", `${windows.length} windows`);
    if (own) {
      const loads = [];
      for (let i = 0; i < windows.length; i++) loads.push(await loadWindowFiles(run, i));
      await tw.load(loads, await windowBytes(run));
    }
    checkCancel(progress);
    if (workers) workers.clearCaps(); // the final parameters run uncapped
    const trial = await tw.trial(sessionBytes, PLACEHOLDER_CHECKS, false);
    if (trial.error) throw new HttpError(400, trial.error.replace(/^(ValueError|RuntimeError): /, ""));
    stages.end();
    checkCancel(progress);
    stages.start("stitch");
    const core = await wasm();
    const st = ctx.stitch
      ? ctx.stitch(sessionBytes)
      : (new core.StitchEngine(new Map([["session.json", sessionBytes]])) as unknown as Stitch);
    const units = await tw.units();
    const windowRows: Record<string, unknown>[] = [];
    for (const u of units) {
      st.add_window(u.bytes, u.t0, u.t1);
      const wf = await loadWindowFiles(run, u.i);
      st.add_mean_noise(
        wf.files.get("mean.npy") as Uint8Array,
        wf.files.get("sigma.npy") as Uint8Array,
        tile,
        u.t0,
        u.t1,
      );
      const pw = trial.perWindow.find((x) => x.i === u.i);
      windowRows.push({
        t0: u.t0,
        t1: u.t1,
        stages: { seeds: pw?.out.seconds[0], init: pw?.out.seconds[1], cnmf: pw?.out.seconds[2] },
        counts: pw?.out.counts ?? {},
      });
    }
    st.run();
    const stitch = st.finish(JSON.stringify({ where: "browser" }));
    st.free();
    stages.end();
    checkCancel(progress);
    const stream = await streamPass(ctx, run, sessionBytes, stitch, T, progress, stages);
    checkCancel(progress);
    stages.start("write_outputs");
    const traces = stream.get("C.npy");
    const yra = stream.get("YrA.npy");
    if (!(traces instanceof Blob) || !(yra instanceof Blob))
      throw new Error("the streaming pass gave no C.npy / YrA.npy");
    const report = rerunReport(
      stages.rows.slice(rows0),
      windowRows,
      dec(stitch.get("summary_stitch.json") ?? enc({})),
      dec((stream.get("summary.json") as Uint8Array | undefined) ?? enc({})),
      T,
      H,
      W,
      now() - t0,
      run.vpath,
      tw.heapPeakMb,
    );
    const parts: RerunParts = {
      session,
      params: merged,
      sourceRun: run.vpath,
      rawDir: String(run.params.raw_dir ?? ""),
      fps: (run.params.fps as number | null) ?? null,
      H,
      W,
      T,
      tile,
      windows,
      stitch,
      stream,
      traces,
      yra,
      report,
      progress: stages.lines,
    };
    stages.end();
    await writeRerun(out, run, parts);
    return out;
  } catch (e) {
    stages.end(true);
    const text = e instanceof RerunCancelled ? "cancelled" : e instanceof Error ? e.message : String(e);
    await out.write("error.txt", text).catch(() => undefined);
    throw e;
  } finally {
    if (own) tw.close();
  }
}

/** `autotune.run.StepLog`: one stage row per search step (`tune_seeds` ...), `n` settings each, and
 * one `tune_rerun` row for the final rerun's stages. */
export class StepLog {
  private step: string | null = null;
  private n = 0;
  private endRerun: ((error?: boolean) => void) | null = null;
  constructor(private stages: Stages) {}
  private switchTo(step: string) {
    if (step === this.step) return;
    this.close();
    this.step = step;
    this.n = 0;
    this.stages.start(`tune_${step}`);
  }
  trial(t: Trial) {
    this.switchTo(t.step);
    this.n += 1;
    this.stages.note(`${this.n} settings`, this.n);
  }
  /** The final rerun's row: spans the stage rows the rerun emits until `close`. */
  stage(name: string) {
    this.close();
    this.endRerun = this.stages.span(`tune_${name}`);
  }
  close(error = false) {
    if (this.endRerun) {
      this.endRerun(error);
      this.endRerun = null;
    }
    if (this.step === null) return;
    this.stages.end(error, this.n);
    this.step = null;
  }
}

/** `seeds.contrast_knee` (the Rust one lives in the worker; the summed histogram is judged here). */
export function contrastKnee(hist: number[], ratio: number): number | null {
  for (let c = 1; c < hist.length - 2; c++)
    if (hist[c + 1] > 0 && hist[c + 2] > 0 && hist[c] / hist[c + 1] < ratio && hist[c + 1] / hist[c + 2] < ratio)
      return c;
  return null;
}

/** `TrialRunner` over the tune workers: the explored windows, then all of them for `decide`. */
export class WorkersRunner implements TrialRunner {
  private current: "segment" | "all" | null = null;
  constructor(
    private tw: TuneWorkers,
    private run: RunInfo,
    private checks: Checks,
    private segment: { from: number; to: number } | null,
    private log: (m: string) => void,
  ) {}
  async use(set: "segment" | "all") {
    if (set === this.current) return;
    const n = windowsOf(this.run).length;
    const [from, to] = set === "segment" && this.segment ? [this.segment.from, this.segment.to] : [0, n];
    const loads = [];
    for (let i = from; i < to; i++) loads.push(await loadWindowFiles(this.run, i));
    await this.tw.load(loads, await windowBytes(this.run));
    this.log(`${to - from} windows on the tune workers (${set})`);
    this.current = set;
  }
  async trial(params: Params, reference: boolean) {
    const { H, W } = this.run.meta;
    const r = await this.tw.trial(enc(sessionFor(H, W, params)), this.checks, reference);
    return { counts: r.counts, score: r.score, error: r.error, size10Median: r.size10Median, seconds: r.seconds };
  }
  async startDiffThres(params: Params) {
    const hist = await this.tw.contrastHist(
      Number(params.seeds_wnd),
      Number(params.seeds_stp),
      Number(params.seeds_max_wnd),
      KNEE_WINDOWS,
    );
    return contrastKnee(hist, KNEE_RATIO);
  }
}

const counts = (p: Pairs): PairCounts => ({
  n_near: p.n_near,
  n_near_above: p.n_near_above,
  n_closer: p.n_closer,
  far_p99: p.far_p99,
});

/** `autotune.run.autotune`: the search, the final rerun into `reruns/auto`, `sweep.csv`,
 * `unit_flags.csv` and `autotune.json`. */
export async function autotune(ctx: Ctx, run: RunInfo, args: TuneArgs, progress: Progress): Promise<Dir> {
  const t0 = now();
  const { H, W, T } = run.meta;
  const source = args.diameter !== undefined ? "given" : "footprints";
  const diameter = args.diameter ?? (await estimateDiameter(run));
  checkDiameter(diameter, H, W);
  const checks: Checks = {
    diameter,
    pnr_min: 12,
    piece_frac: 0.9,
    size_lo: 0.3,
    size_hi: 3,
    dup_corr: args.dup_corr ?? 0.6,
    keep_dist: 0.5,
  };
  const minDistFrac = args.min_dist_frac ?? MIN_DIST_FRAC;
  const tuneFrames = args.tune_frames ?? TUNE_FRAMES;
  const runParams: Params = { ...DEFAULTS, ...flatParams(run) }; // a run's params.json may leave defaults out
  const base = derivedParams(runParams, diameter, minDistFrac);
  const runValues = Object.fromEntries(SEARCHED.map((k) => [k, runParams[k] as number | string])) as Values;
  const mid = middleWindows(windowsOf(run), tuneFrames);
  const out = await resetRerunDir(run, AUTO);
  const stages = new Stages(progress, out);
  const log = new StepLog(stages);
  const tw = new TuneWorkers(ctx.makeTuneWorker, enc(sessionFor(H, W, runParams)), ctx.maxWorkers);
  try {
    const runner = new WorkersRunner(tw, run, checks, mid, progress.log);
    const search = new LocalSearch(runner, {
      base,
      runParams,
      runValues,
      integerDiff: true,
      segment: mid !== null,
      onTrial: (t) => {
        checkCancel(progress);
        log.trial(t);
      },
    });
    const proposed = await search.run();
    log.stage("rerun");
    const params = { ...base, ...proposed };
    const changed = Object.fromEntries(Object.entries(params).filter(([k, v]) => norm(v) !== norm(runParams[k])));
    await rerun(ctx, run, changed, AUTO, progress, tw, stages); // the workers hold every window by now
    log.close();
    const auto = await ctx.registry.open(`${run.vpath}/reruns/${AUTO}`);
    const runScored = await scoreRun(run, checks, minDistFrac, []);
    const proposal = await scoreRun(auto, checks, minDistFrac, runScored.good);
    await out.write("sweep.csv", sweepCsv(search.trials));
    await out.write("unit_flags.csv", proposal.csv);
    const kept = derivedKeys(runParams, diameter);
    const doc: AutotuneDoc = {
      strategy: "local",
      passes: search.passes,
      start: search.start,
      diameter,
      diameter_source: source,
      min_dist_frac: minDistFrac,
      tune_frames: tuneFrames,
      segment: mid ? mid.segment : null,
      frames: T,
      pairs: { run: counts(runScored.pairs), proposal: counts(proposal.pairs) },
      checks,
      grids: search.grids,
      derived: Object.fromEntries(DERIVED.map((k) => [k, params[k]])),
      kept: DERIVED.filter((k) => !kept.includes(k)),
      proposed,
      run_values: Object.fromEntries([...SEARCHED, ...DERIVED].map((k) => [k, runParams[k]])),
      at_edge: SEARCHED.filter((k) => atEdge(proposed[k], search.grids[k] ?? [])),
      score: proposal.summary,
      run_score: search.runTrial?.score ?? null,
      better_than_run: search.betterThanRun,
      search_error: search.searchError,
      n_trials: search.trials.length,
      wall_s: Math.round((now() - t0) * 10) / 10,
      wasm_heap_peak_mb: Math.round(tw.heapPeakMb),
    };
    await out.write("autotune.json", JSON.stringify(doc, null, 1));
    return out;
  } catch (e) {
    log.close(true);
    const text = e instanceof RerunCancelled ? "cancelled" : e instanceof Error ? e.message : String(e);
    await out.write("error.txt", text).catch(() => undefined);
    throw e;
  } finally {
    tw.close();
  }
}
