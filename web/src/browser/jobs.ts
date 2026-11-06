import type { Device } from "../gpu/plan";
import { isFinal, type ProgressEvent, type RunPlan, type StageProgress } from "../launch/params";
import { workerCount } from "../stream/pool";
import type { FromWorker, LiveToggle, ToWorker } from "../stream/protocol";
import { DEFAULTS, unusedParams, WINDOW_FRAMES } from "../stream/session";
import type { Job } from "../types";
import { type Indexer, listSegments, openRawVideo, RawVideo } from "./avi";
import { type Dir, exists, HandleDir, readJSON, writeJSON } from "./dir";
import { type Folders, OPFS_NAME } from "./folders";
import type { KV } from "./kv";
import { pyFloat, pyRepr } from "./num";
import { HttpError } from "./router";
import type { RunInfo } from "./runs";
import { state } from "./state";
import { ASSUMED_UNITS, estimateRunBytes, pickDirectory, quotaMessage, RUN_LOCK, support } from "./support";
import { autotune, type Ctx, type Progress, RerunCancelled, rerun } from "./tune/driver";
import { checkWindow2, type Params, type TuneArgs, tuneArgs } from "./tune/params";
import { joinVpath, parentVpath, splitVpath } from "./vpath";

export type JobBody = {
  raw_dir: string;
  name: string;
  params: Record<string, unknown>;
  keep_store: boolean;
  overwrite: boolean;
  tune: boolean;
  tune_args: Record<string, number>;
  fps: number | null;
  /** Where the per-frame stages run (the browser build's Speed panel): auto, cpu, gpu or both. */
  device?: Device;
};

/** A job as main's `asdict(Job)` has it: the page's `Job` plus the fields it does not read. */
export type JobRecord = Job & { overwrite: boolean; pid: number | null; fps: number | null };

/** `jobs.STOPPED`: what a job that was going when the page went away says. */
const STOPPED = "the server stopped during the run";
type Emit = (name: string, data: unknown) => void;

/** `pipeline.check_run_name`: a run name is one plain directory name. */
function checkRunName(name: string): void {
  if (name === "" || name === "." || name === ".." || /[/\\]/.test(name))
    throw new HttpError(400, `invalid run name ${pyRepr(name)}: use a single, plain directory name`);
}

/** `io.check_fps`. */
function checkFps(fps: number): void {
  if (!(fps > 0 && fps <= 1000)) throw new HttpError(400, `frame rate must be in (0, 1000] Hz, got ${pyFloat(fps)}`);
}

/** `pipeline._check_out_dir`: the run is never written into, or around, the raw folder. */
function checkOutDir(raw: string, out: string): void {
  if (raw === out || raw.startsWith(`${out}/`))
    throw new HttpError(400, `output dir ${out} would contain the raw dir ${raw}; choose another`);
  if (out.startsWith(`${raw}/`))
    throw new HttpError(400, `output dir ${out} lies inside the raw dir ${raw}; choose another`);
}

/** `checkOutDir` for a folder picked in the dialog. It is a root of its own, so the vpaths say
 * nothing about it being the raw folder or lying inside it; the handles do (`resolve` gives `[]`
 * for the same folder, the names down to it for one inside, else null). Done before the folder is
 * remembered as a root, so a refused one is not. */
async function checkPickedOutsideRaw(raw: Dir, picked: Dir | null, rawDir: string, name: string): Promise<void> {
  if (!(raw instanceof HandleDir && picked instanceof HandleDir)) return;
  if ((await raw.handle.resolve(picked.handle).catch(() => null)) === null) return;
  throw new HttpError(
    400,
    `output dir ${joinVpath(picked.name, name)} lies inside the raw dir ${rawDir}; choose another`,
  );
}

/** The motion steps halve the frame (`motion_down` 2) and the band-pass registration image
 * (`motion_before_tophat`) needs an even size too; the preview refuses an odd one in the same
 * words (`preview.rs`), so New run says so before a run starts and fails at its first stage. */
function checkFrameSize(h: number, w: number, params: Record<string, unknown>): void {
  if (h % 2 === 0 && w % 2 === 0) return;
  const p = { ...DEFAULTS, ...params };
  if (p.motion_before_tophat)
    throw new HttpError(400, `${h} x ${w} frames: the band-pass registration image needs an even size`);
  if (p.motion_down === 2) throw new HttpError(400, `${h} x ${w} frames: 2x downsampling needs an even size`);
}

/** The path of `handle` inside OPFS (Browser storage), or null for a folder on disk (or no OPFS):
 * a run folder in OPFS goes to the worker by its path, as Safari cannot send it the handle. */
async function opfsPath(handle: FileSystemDirectoryHandle | undefined): Promise<string[] | null> {
  const storage = globalThis.navigator?.storage;
  if (!handle || !storage?.getDirectory) return null;
  return (await storage.getDirectory()).resolve(handle).catch(() => null);
}

/** `runs.is_run`: a `params.json` and a `report.json`. */
async function isRun(d: Dir): Promise<boolean> {
  return (await exists(d, "params.json")) && (await exists(d, "report.json"));
}

/** What `pipeline._is_minime_run` takes as a MiniMe `params.json`: a version key (any release of
 * the package) or, for a run this app wrote, `source: "browser"`. */
const VERSION_KEYS = ["minime_version", "caanfast_version", "cafast_version", "fastminian_version"];

/** `pipeline._is_minime_run`: `d` holds a `params.json` or `report.json` that a MiniMe run wrote. */
async function isMinimeRun(d: Dir): Promise<boolean> {
  const doc = async (name: string): Promise<Record<string, unknown> | null> => {
    const v = await readJSON<unknown>(d, name).catch(() => null);
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  };
  const params = await doc("params.json");
  if (params && (VERSION_KEYS.some((k) => k in params) || params.source === "browser")) return true;
  const report = await doc("report.json");
  return report !== null && "stages" in report;
}

/** `pipeline._check_wipe` and `_reset_dir`, with `runs.is_run` and the refusal of `submit`: the
 * fresh, empty `processed/<name>`. An existing folder is cleared only when it holds a MiniMe
 * run (the run being replaced needs `overwrite`; the leftovers of an interrupted one do not);
 * anything else in it is somebody's files and stays. */
async function resetOutDir(processed: Dir, name: string, outDir: string, overwrite: boolean): Promise<Dir> {
  const entry = (await processed.entries()).find((e) => e.name === name);
  if (entry?.kind === "file") throw new HttpError(400, `output path ${outDir} exists and is not a directory`);
  if (entry) {
    const old = await processed.dir(name);
    if ((await isRun(old)) && !overwrite)
      throw new HttpError(400, `${outDir} already holds a run; choose another name or overwrite`);
    if ((await old.entries()).length && !(await isMinimeRun(old)))
      throw new HttpError(
        400,
        `output dir ${outDir} is not empty and holds no minime params.json / report.json; refusing to delete it (choose another name)`,
      );
    await processed.remove(name, { recursive: true });
  }
  const out = await processed.dir(name, { create: true });
  // a handle that could not remove an entry says nothing: what is left would end up in the new run
  if ((await out.entries()).length) throw new HttpError(500, `could not clear ${outDir}`);
  return out;
}

/** What a rerun or auto-tune job runs (`tune/driver.ts`; tests replace it). */
export type TuneRunner = {
  rerun(run: RunInfo, params: Params, name: string, progress: Progress): Promise<unknown>;
  autotune(run: RunInfo, args: TuneArgs, progress: Progress): Promise<unknown>;
};

/** The part of the browser's Web Locks (`navigator.locks`) the job manager uses. */
export type Locks = Pick<LockManager, "request" | "query">;

/** One run at a time (main: `jobs.JobManager`, which queues), in this tab and, with the run lock,
 * across the site's tabs: they share browser storage and its scratch folder. Job records live in
 * `kv`; a run is a worker that writes its run folder into `<raw parent>/processed/<name>`. */
export class JobManager {
  private jobs = new Map<string, JobRecord>();
  /** The job this tab runs (the last one), the run lock's release while it goes, and a start
   * that is between its checks and its worker (a second one in this tab waits for neither). */
  private own: JobRecord | null = null;
  private unlock: (() => void) | null = null;
  private starting = false;
  /** This tab holds the run lock (from a start's lock to its run's end). */
  private holding = false;
  /** Per running tune job: what stops its workers at once, and its promise (the lock goes after it settles). */
  private tunes = new Map<string, { abort: (() => void) | null; settled: Promise<void> }>();
  private history = new Map<string, [string, unknown][]>();
  private listeners = new Map<string, Set<Emit>>();
  /** Per job, each stage's latest counters (the page's rates for the run's time left). */
  private latest = new Map<string, Map<string, StageProgress>>();
  /** Jobs whose live view is on (this tab's run only; not saved). */
  private liveOn = new Set<string>();
  /** The latest preview of each panel (each stitch window kept) of a running job, for a page opened later. */
  private latestLive = new Map<string, Map<string, unknown>>();
  private worker: Worker | null = null;
  private wake: WakeLockSentinel | null = null;
  private asking = false;
  private onVisible: (() => void) | null = null;
  private persisted = false;
  /** `pickOutput` is the folder dialog (the system's where the browser has one) and `index` how
   * the raw videos are indexed; both are replaced in tests. `locks` are the browser's Web Locks
   * (none: only this tab's check) and `scratch` the OPFS folder that holds each run's working files
   * in `<job id>/` (the pass-1 tile store, the traces on their way to the run folder). */
  constructor(
    private o: {
      kv: KV;
      folders: Folders;
      makeWorker: () => Worker;
      now?: () => number;
      pickOutput?: () => Promise<Dir>;
      index?: Indexer;
      locks?: Locks | null;
      scratch?: () => Promise<Dir | null>;
      tune?: TuneRunner;
    },
  ) {}

  private now(): number {
    return (this.o.now ?? (() => Date.now() / 1e3))();
  }

  /** Read the saved jobs. One that was going when its page went away is `interrupted`, and the
   * scratch folders runs left are removed, but only when no tab holds the run lock: a job another
   * tab still runs stays as it is. */
  async load(): Promise<void> {
    if (!(await this.refresh())) await this.sweepScratch();
  }

  /** The saved jobs as they are now (other tabs save theirs), with a job that is not final and that
   * no tab runs any more marked `interrupted`; returns whether another tab runs a job. */
  async refresh(): Promise<boolean> {
    const elsewhere = await this.runsElsewhere();
    for (const k of await this.o.kv.keys()) {
      if (k === this.own?.id) continue;
      const saved = await this.o.kv.get<JobRecord>(k);
      if (!saved) continue;
      const j = { ...saved }; // as IndexedDB gives it: a copy
      if (!isFinal(j.status) && !elsewhere) {
        j.status = "interrupted";
        j.error = STOPPED;
        j.ended = this.now();
        await this.o.kv.set(k, j);
      }
      this.jobs.set(j.id, j);
    }
    return elsewhere;
  }

  /** Whether another tab holds the run lock (this tab holds it while its own run goes). Without
   * Web Locks, never: a job found going was going when its page went away. */
  private async runsElsewhere(): Promise<boolean> {
    if (this.holding) return false;
    const state = await this.o.locks?.query().catch(() => null);
    return !!state?.held?.some((l) => l.name === RUN_LOCK);
  }

  /** The run lock (one run at a time across the site's tabs), held until the returned function is
   * called; null when another tab holds it. Without Web Locks there is only this tab's check. */
  private lock(): Promise<(() => void) | null> {
    const locks = this.o.locks;
    if (!locks) return Promise.resolve(() => {});
    return new Promise((ok, no) => {
      locks
        .request(RUN_LOCK, { ifAvailable: true }, (lock) => {
          if (!lock) return ok(null);
          return new Promise<void>((release) => ok(() => release()));
        })
        .catch(no);
    });
  }

  /** Remove run `id`'s scratch folder (`scratch/<id>/`). The stopped worker's files may stay open a
   * moment longer (a removal fails meanwhile), so it is tried again a few times. */
  private dropScratch(id: string): void {
    void (async () => {
      const dir = await this.o.scratch?.();
      for (let i = 0; dir && i < 5; i++) {
        await dir.remove(id, { recursive: true }).catch(() => undefined);
        if (!(await dir.entries()).some((e) => e.name === id)) return;
        await new Promise((r) => setTimeout(r, 200 * (i + 1)));
      }
    })().catch(() => undefined);
  }

  /** Remove every scratch folder: only while no run goes in any tab (what is there was left by a
   * run whose tab went away, or a removal that failed). */
  private async sweepScratch(): Promise<void> {
    const dir = await this.o.scratch?.().catch(() => null);
    if (!dir) return;
    for (const e of await dir.entries().catch(() => []))
      await dir.remove(e.name, { recursive: true }).catch(() => undefined);
  }

  list(): JobRecord[] {
    return [...this.jobs.values()].sort((a, b) => b.created - a.created);
  }

  get(id: string): JobRecord {
    const j = this.jobs.get(id);
    if (!j) throw new HttpError(404, `unknown job ${id}`);
    return j;
  }

  /** The output-folder dialog: the injected one, else the system's where the browser has it. */
  private outputDialog(): (() => Promise<Dir>) | null {
    if (this.o.pickOutput) return this.o.pickOutput;
    return support().fsAccess ? async () => new HandleDir(await pickDirectory("minime-out")) : null;
  }

  /** The folder that will hold the run folder `<name>` (spec section 3) and how to open it for
   * writing: `<raw parent>/processed` when the raw folder has a parent. A raw folder picked on
   * its own has none, and the run is never written inside it: it goes into the folder `picked` in
   * the dialog, else (a browser without the dialog) into Browser storage, else 400. */
  private async holder(
    rawDir: string,
    picked: Dir | null,
    hasDialog: boolean,
  ): Promise<{ vpath: string; open: () => Promise<Dir> }> {
    const { folders } = this.o;
    const parent = parentVpath(rawDir);
    if (parent !== null)
      return {
        vpath: joinVpath(parent, "processed"),
        open: async () => (await folders.resolve(parent, "readwrite")).dir("processed", { create: true }),
      };
    if (picked) {
      const at = await folders.pick(picked); // a root, or a folder inside one
      return { vpath: at, open: () => folders.resolve(at, "readwrite") };
    }
    if (!hasDialog && (await folders.list()).some((r) => r.name === OPFS_NAME))
      return { vpath: OPFS_NAME, open: () => folders.resolve(OPFS_NAME, "readwrite") };
    throw new HttpError(
      400,
      `${rawDir} has no parent folder the app can write to: start the run again and, in the dialog, pick the folder the run should be written to (it goes into a new folder there; the raw folder is never written to)`,
    );
  }

  /** A run into Browser storage has to fit in what the browser lets this site keep; the browser is
   * also asked, once, not to clear that storage by itself. Nothing is checked where the browser
   * cannot say how much room there is. */
  private async checkStorage(v: RawVideo, keepStore: boolean): Promise<void> {
    const storage = globalThis.navigator?.storage;
    if (!storage) return;
    const { quota, usage } = (await storage.estimate?.().catch(() => undefined)) ?? {};
    if (quota !== undefined && usage !== undefined) {
      const need = estimateRunBytes(
        ASSUMED_UNITS,
        v.frames,
        v.height * v.width,
        Math.ceil(v.frames / WINDOW_FRAMES),
        keepStore,
      );
      if (need > quota - usage) throw new HttpError(400, quotaMessage(need, quota - usage));
    }
    if (!this.persisted) {
      this.persisted = true;
      void storage.persist?.().catch(() => undefined);
    }
  }

  /** The tune runner: the injected one, else `driver.ts` with the app's folders and workers. */
  private tuneRunner(): TuneRunner {
    if (this.o.tune) return this.o.tune;
    const ctx: Ctx = {
      registry: state.registry,
      folders: state.folders,
      makeTuneWorker: () => new Worker(new URL("../stream/tune.worker.ts", import.meta.url), { type: "module" }),
      makeStreamWorker: this.o.makeWorker,
      maxWorkers: workerCount({}),
      scratch: "tune",
    };
    return {
      rerun: (run, params, name, p) => rerun({ ...ctx, onWorkers: p.onWorkers }, run, params, name, p),
      autotune: (run, args, p) => autotune({ ...ctx, onWorkers: p.onWorkers }, run, args, p),
    };
  }

  /** The run a rerun or auto-tune job starts from (a 404 when the folder is none). */
  private sourceRun(runDir: string): Promise<RunInfo> {
    return state.registry.open(runDir);
  }

  /** One job at a time, in this tab and across tabs: `begin` runs with the run lock held and gets
   * the function that lets it go when the job ends; a refusal lets it go at once. */
  private async withLock(begin: (unlock: () => void) => Promise<JobRecord>): Promise<JobRecord> {
    if (this.starting || (this.own && !isFinal(this.own.status)))
      throw new HttpError(400, "a run is already going in this tab; wait for it or cancel it");
    this.starting = true;
    try {
      const release = await this.lock();
      if (!release) throw new HttpError(400, "a run is already going in another tab; wait for it or cancel it there");
      this.holding = true;
      const unlock = () => {
        this.holding = false;
        release();
      };
      try {
        await this.sweepScratch(); // the lock is ours: no run goes anywhere
        return await begin(unlock);
      } catch (e) {
        unlock();
        throw e;
      }
    } finally {
      this.starting = false;
    }
  }

  /** `reruns/<name>` of a run as a job record of `kind`. */
  private tuneJob(kind: "rerun" | "autotune", runDir: string, name: string, params: Params, args: TuneArgs): JobRecord {
    return {
      id: crypto.randomUUID().replaceAll("-", "").slice(0, 10),
      raw_dir: "",
      name,
      params,
      keep_store: false,
      status: "running",
      out_dir: joinVpath(runDir, "reruns", name),
      created: this.now(),
      started: this.now(),
      ended: null,
      error: "",
      overwrite: true,
      pid: null,
      kind,
      run_dir: runDir,
      tune: false,
      tune_args: args as Record<string, number>,
      fps: null,
    };
  }

  /** Window 2 of the run at `runDir` with `params` into `reruns/<name>`. */
  async submitRerun(runDir: string, name: string, params: Params): Promise<JobRecord> {
    checkRunName(name);
    if (name === "auto") throw new HttpError(400, "run name 'auto' is reserved for auto-tune; pick another name");
    checkWindow2(params);
    const run = await this.sourceRun(runDir);
    return this.withLock(async (unlock) => {
      const job = this.tuneJob("rerun", runDir, name, params, {});
      return this.beginTune(job, unlock, (p) => this.tuneRunner().rerun(run, params, name, p));
    });
  }

  /** Auto-tune the run at `runDir`, into `reruns/auto`. */
  async submitAutotune(runDir: string, raw: Record<string, unknown>): Promise<JobRecord> {
    const args = tuneArgs(raw);
    const run = await this.sourceRun(runDir);
    return this.withLock(async (unlock) => {
      const job = this.tuneJob("autotune", runDir, "auto", {}, args);
      return this.beginTune(job, unlock, (p) => this.tuneRunner().autotune(run, args, p));
    });
  }

  private async beginTune(
    job: JobRecord,
    unlock: () => void,
    go: (p: Progress) => Promise<unknown>,
  ): Promise<JobRecord> {
    await this.o.kv.set(job.id, job);
    this.jobs.set(job.id, job);
    this.own = job;
    this.unlock = unlock;
    this.record(job.id, "status", { ...job });
    this.startTune(job, go, true);
    return job;
  }

  /** Run `go` for the job on its own progress; its end is the job's end, unless a cancel got there
   * first. `awake` is false for a tune chained after a run (the run took the wake lock). */
  private startTune(job: JobRecord, go: (p: Progress) => Promise<unknown>, awake: boolean): void {
    const p: Progress = {
      stage: (e) => this.record(job.id, "stage", e),
      live: (s) => this.live(job.id, s),
      log: (m) => this.record(job.id, "log", m),
      cancelled: () => isFinal(job.status),
    };
    const tune: { abort: (() => void) | null; settled: Promise<void> } = { abort: null, settled: Promise.resolve() };
    p.onWorkers = (stop) => {
      tune.abort = stop;
    };
    if (awake) this.keepAwake(job);
    const fail = (e: unknown) =>
      this.end(job, e instanceof RerunCancelled ? "cancelled" : "failed", e instanceof Error ? e.message : String(e));
    let cancelledFirst = false;
    const settled = (end: () => Promise<void>) => {
      cancelledFirst = isFinal(job.status); // a cancel got there first: `end` will do nothing
      return end();
    };
    tune.settled = Promise.resolve()
      .then(() => go(p))
      .then(
        () => settled(() => this.end(job, "done", "")),
        (e) => settled(() => fail(e)),
      )
      .then(async () => {
        this.tunes.delete(job.id);
        // the cancel left the lock to the settled tune
        if (cancelledFirst && this.own === job) await this.finish(job);
      })
      .catch((e) => console.error(`job ${job.id}: could not finish the tune job`, e)); // the status stays as `end` set it
    this.tunes.set(job.id, tune);
  }

  private async end(job: JobRecord, status: "done" | "failed" | "cancelled", error: string): Promise<void> {
    if (isFinal(job.status)) return; // a cancel got there first
    job.status = status;
    job.error = error;
    job.ended = this.now();
    await this.finish(job);
  }

  /** Start a run (main's `JobManager.submit` checks, in its order, then the browser's own). */
  async submit(b: JobBody): Promise<JobRecord> {
    const rawDir = b.raw_dir.split("/").filter(Boolean).join("/"); // "a/raw/" is the folder "a/raw"
    // A raw folder picked on its own asks for the folder to write to. The dialog needs the user
    // activation of the Start click, which lasts a few seconds and not past an await: it comes
    // before the first one, and the folder is only remembered once the request is accepted.
    const dialog = this.outputDialog();
    const picked = dialog && parentVpath(rawDir) === null ? await dialog().catch(() => null) : null;
    const raw = await this.o.folders.resolve(rawDir);
    const listed = await listSegments(raw);
    const { files } = listed;
    if (!files.length) throw new HttpError(400, `no videos in ${rawDir}: ${RawVideo.NAMING_RULE}`);
    if (b.fps !== null) checkFps(b.fps);
    checkRunName(b.name);
    const bad = unusedParams(b.params);
    if (bad.length) throw new HttpError(400, `${bad.join(", ")}: not used by the browser app`);
    if (b.tune) b = { ...b, tune_args: tuneArgs(b.tune_args) as Record<string, number> }; // refused now, not hours in
    return this.withLock((unlock) => this.begin(b, rawDir, raw, listed, picked, dialog !== null, unlock));
  }

  /** The rest of `submit`, with the run lock held (`unlock` lets it go when the run ends). */
  private async begin(
    b: JobBody,
    rawDir: string,
    raw: Dir,
    listed: Awaited<ReturnType<typeof listSegments>>,
    picked: Dir | null,
    hasDialog: boolean,
    unlock: () => void,
  ): Promise<JobRecord> {
    const { files } = listed;
    const video = await openRawVideo(rawDir, raw, { index: this.o.index, listed }); // as New run indexed it
    checkFrameSize(video.height, video.width, b.params);
    await checkPickedOutsideRaw(raw, picked, rawDir, b.name);
    const holder = await this.holder(rawDir, picked, hasDialog);
    const outDir = joinVpath(holder.vpath, b.name);
    checkOutDir(rawDir, outDir);
    if (splitVpath(outDir).root === OPFS_NAME) await this.checkStorage(video, b.keep_store);
    const out = await resetOutDir(await holder.open(), b.name, outDir, b.overwrite);
    const handle = (out as { handle?: FileSystemDirectoryHandle }).handle;
    const outPath = await opfsPath(handle);
    // first, as main's `_write_params` does: it marks the folder as a MiniMe run, so an interrupted
    // run (no `report.json` yet, so not listed) can be started again under its name
    await writeJSON(out, "params.json", { source: "browser", raw_dir: rawDir, fps: b.fps });
    const job: JobRecord = {
      id: crypto.randomUUID().replaceAll("-", "").slice(0, 10),
      raw_dir: rawDir,
      name: b.name,
      params: b.params,
      keep_store: b.keep_store,
      status: "running",
      out_dir: outDir,
      created: this.now(),
      started: this.now(),
      ended: null,
      error: "",
      overwrite: b.overwrite,
      pid: null,
      kind: "run",
      run_dir: "",
      tune: b.tune,
      tune_args: b.tune_args,
      fps: b.fps,
    };
    await this.o.kv.set(job.id, job);
    await this.o.folders.touch(splitVpath(holder.vpath).root, rawDir);
    // from here on the run ends through `finish`, which lets the lock go
    this.jobs.set(job.id, job);
    this.own = job;
    this.unlock = unlock;
    this.record(job.id, "status", { ...job });
    this.start(job, files, outPath ? { outPath } : { out: handle }, { ...b, raw_dir: rawDir });
    return job;
  }

  /** Hand the run to a worker; whatever stops it from starting fails the job. */
  private start(job: JobRecord, avis: File[], dest: Pick<ToWorker, "out" | "outPath">, b: JobBody): void {
    const msg: ToWorker = {
      avis,
      inputs: [],
      ...dest,
      params: b.params,
      keepStore: b.keep_store,
      rawDir: b.raw_dir,
      fps: b.fps,
      scratch: job.id,
      browserStorage: splitVpath(job.out_dir).root === OPFS_NAME,
      device: b.device,
    };
    try {
      const w = this.o.makeWorker();
      this.worker = w;
      w.onmessage = (e: MessageEvent<FromWorker>) => void this.onWorker(job, e.data);
      w.onerror = (e: ErrorEvent) => void this.fail(job, e.message || "the worker script failed");
      w.postMessage(msg);
    } catch (e) {
      void this.fail(job, e);
      return;
    }
    this.keepAwake(job);
  }

  /** Keep the screen (and with it the page's worker) awake while the run goes, where the browser
   * has a Wake Lock. A browser lets the lock go when the tab is hidden, so it is taken again
   * whenever the tab is shown, until the run ends (`finish`). */
  private keepAwake(job: JobRecord): void {
    const wakeLock = globalThis.navigator?.wakeLock;
    if (!wakeLock) return;
    const take = async () => {
      if (this.asking || isFinal(job.status) || (this.wake && !this.wake.released)) return;
      this.asking = true;
      const lock = await wakeLock.request("screen").catch(() => null);
      this.asking = false;
      if (!lock) return;
      if (isFinal(job.status))
        await lock.release().catch(() => undefined); // the run ended meanwhile
      else this.wake = lock;
    };
    this.onVisible = () => {
      if (globalThis.document.visibilityState === "visible") void take();
    };
    globalThis.document?.addEventListener("visibilitychange", this.onVisible);
    void take();
  }

  private fail(job: JobRecord, why: unknown): Promise<void> {
    const message = why instanceof Error ? why.message : String(why);
    return this.onWorker(job, { type: "error", message: `could not start the run: ${message}` });
  }

  private async onWorker(job: JobRecord, m: FromWorker): Promise<void> {
    if (isFinal(job.status)) return; // a message that was already on its way when the job was cancelled
    if (m.type === "stage") {
      const { type: _, ...event } = m; // `progress.jsonl`'s dict, as main's events carry it
      this.record(job.id, "stage", event satisfies ProgressEvent);
    } else if (m.type === "progress") {
      const { stage, done, frames, seconds, unit } = m;
      this.live(job.id, { stage, done, frames, seconds, unit } satisfies StageProgress);
    } else if (m.type === "live") {
      const { type: _, ...p } = m;
      const key = p.panel === "stitch" ? `stitch ${p.window}` : p.panel;
      const panels = this.latestLive.get(job.id) ?? new Map<string, unknown>();
      this.latestLive.set(job.id, panels);
      panels.set(key, p);
      for (const l of this.listeners.get(job.id) ?? []) l("live", p);
    } else if (m.type === "plan") {
      const { type: _, ...plan } = m;
      this.record(job.id, "plan", plan satisfies RunPlan);
    } else if (m.type === "log") this.record(job.id, "log", m.message);
    else if (m.type === "done" && job.kind === "run" && job.tune) await this.chainTune(job);
    else if (m.type === "done" || m.type === "error") {
      job.status = m.type === "done" ? "done" : "failed";
      job.error = m.type === "error" ? m.message : "";
      job.ended = this.now();
      await this.finish(job);
    }
  }

  /** The run is done and asked for a tune: the same job goes on with the auto-tune of its folder. */
  private async chainTune(job: JobRecord): Promise<void> {
    this.worker?.terminate();
    this.worker = null;
    this.liveOn.delete(job.id);
    this.latestLive.delete(job.id);
    try {
      const run = await state.registry.open(job.out_dir);
      this.startTune(job, (p) => this.tuneRunner().autotune(run, job.tune_args, p), false);
    } catch (e) {
      await this.end(job, "failed", e instanceof Error ? e.message : String(e));
    }
  }

  private async finish(job: JobRecord): Promise<void> {
    this.worker?.terminate();
    this.worker = null;
    if (this.onVisible) globalThis.document?.removeEventListener("visibilitychange", this.onVisible);
    this.onVisible = null;
    await this.wake?.release().catch(() => undefined);
    this.wake = null;
    await this.o.kv.set(job.id, job);
    // saved first: a tab that sees the lock free finds the job final, not one to interrupt
    this.unlock?.();
    this.unlock = null;
    this.dropScratch(job.id);
    this.latest.delete(job.id);
    this.liveOn.delete(job.id);
    this.latestLive.delete(job.id);
    this.record(job.id, "status", { ...job });
  }

  /** The job page's Live view switch: tells this tab's run worker to send previews (or stop). */
  setLive(id: string, on: boolean): { on: boolean } {
    const job = this.get(id);
    if (job !== this.own || !this.worker || isFinal(job.status) || job.kind !== "run")
      throw new HttpError(400, "this run is not going in this tab");
    if (on) this.liveOn.add(id);
    else this.liveOn.delete(id);
    this.worker.postMessage({ type: "live", on } satisfies LiveToggle);
    for (const l of this.listeners.get(id) ?? []) l("liveon", { on });
    return { on };
  }

  async cancel(id: string): Promise<JobRecord> {
    let job = this.get(id);
    if (job !== this.own && !isFinal(job.status)) {
      await this.refresh(); // its tab may have finished it, or gone away
      job = this.get(id);
      if (!isFinal(job.status)) throw new HttpError(400, "this run is going in another tab; cancel it there");
    }
    if (!isFinal(job.status)) {
      job.status = "cancelled";
      job.ended = this.now();
      const tune = this.tunes.get(job.id);
      if (tune) {
        tune.abort?.(); // the workers stop now; the lock goes when the tune has settled
      } else await this.finish(job);
    }
    return job;
  }

  private record(id: string, name: string, data: unknown): void {
    const h = this.history.get(id) ?? [];
    this.history.set(id, h);
    h.push([name, data]);
    for (const l of this.listeners.get(id) ?? []) l(name, data);
  }

  /** The running stage's counters: sent to the open job pages and kept only as each stage's latest
   * (a page opened later gets them after the replay), not in the history or `progress.jsonl`. */
  private live(id: string, p: StageProgress): void {
    const stages = this.latest.get(id) ?? new Map<string, StageProgress>();
    this.latest.set(id, stages);
    stages.delete(p.stage); // the running stage last: it is the page's live one
    stages.set(p.stage, p);
    for (const l of this.listeners.get(id) ?? []) l("progress", p);
  }

  /** Replay the job's events so far, then stream new ones (main's `/jobs/<id>/events`). A job this
   * tab runs replays what it recorded: a status first, then the stages and log lines, then the
   * status again when it changes. Any other job (from before a reload, or another tab's) replays
   * as main does: the stage events of its run folder's `progress.jsonl`, then its status. */
  events(id: string, emit: Emit): () => void {
    const job = this.get(id);
    let open = true;
    const send: Emit = (n, d) => {
      if (open) emit(n, d);
    };
    const past = this.history.get(id);
    if (past) {
      for (const [n, d] of past) send(n, d);
      if (!isFinal(job.status)) for (const p of this.latest.get(id)?.values() ?? []) send("progress", p);
      if (!isFinal(job.status) && job === this.own && this.worker) {
        send("liveon", { on: this.liveOn.has(id) });
        for (const p of this.latestLive.get(id)?.values() ?? []) send("live", p);
      }
    } else
      void this.progress(job).then((stages) => {
        for (const e of stages) send("stage", e);
        send("status", { ...this.get(id) });
      });
    const set = this.listeners.get(id) ?? new Set<Emit>();
    this.listeners.set(id, set);
    set.add(send);
    return () => {
      open = false;
      set.delete(send);
    };
  }

  /** `JobManager.progress` of main: the stage events in `<out_dir>/progress.jsonl` from the job's
   * start on (older lines are an earlier run's), by time. A line that does not parse is skipped (it
   * is being written); a folder that cannot be read gives none. */
  private async progress(job: JobRecord): Promise<ProgressEvent[]> {
    const started = job.started;
    if (started === null) return [];
    const text = await this.o.folders
      .resolve(job.out_dir)
      .then((d) => d.file("progress.jsonl"))
      .then((f) => f.text())
      .catch(() => "");
    const out: ProgressEvent[] = [];
    for (const line of text.split(/\r?\n/)) {
      if (!line.trim()) continue;
      try {
        const e = JSON.parse(line) as ProgressEvent;
        if (e && typeof e === "object" && (e.t ?? 0) >= started - 1) out.push(e);
      } catch {
        // a line being written
      }
    }
    return out.sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
  }
}
