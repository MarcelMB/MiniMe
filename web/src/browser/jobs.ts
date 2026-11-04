import { isFinal, type ProgressEvent } from "../launch/params";
import type { FromWorker, ToWorker } from "../stream/protocol";
import { unusedParams } from "../stream/session";
import type { Job } from "../types";
import { listSegments, RawVideo } from "./avi";
import { type Dir, exists } from "./dir";
import type { Folders } from "./folders";
import type { KV } from "./kv";
import { pyFloat, pyRepr } from "./num";
import { HttpError } from "./router";
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

/** One run at a time in this tab (main: `jobs.JobManager`, which queues). Job records live in
 * `kv`; a run is a worker that writes its run folder into `<raw parent>/processed/<name>`. */
export class JobManager {
  private jobs = new Map<string, JobRecord>();
  private history = new Map<string, [string, unknown][]>();
  private listeners = new Map<string, Set<Emit>>();
  private worker: Worker | null = null;
  private wake: { release(): Promise<void> } | null = null;
  constructor(private o: { kv: KV; folders: Folders; makeWorker: () => Worker; now?: () => number }) {}

  private now(): number {
    return (this.o.now ?? (() => Date.now() / 1e3))();
  }

  /** Read the saved jobs; one that was going when the page went away is `interrupted`. */
  async load(): Promise<void> {
    for (const k of await this.o.kv.keys()) {
      const j = await this.o.kv.get<JobRecord>(k);
      if (!j) continue;
      if (!isFinal(j.status)) {
        j.status = "interrupted";
        j.error = STOPPED;
        j.ended = this.now();
        await this.o.kv.set(k, j);
      }
      this.jobs.set(j.id, j);
    }
  }

  list(): JobRecord[] {
    return [...this.jobs.values()].sort((a, b) => b.created - a.created);
  }

  get(id: string): JobRecord {
    const j = this.jobs.get(id);
    if (!j) throw new HttpError(404, `unknown job ${id}`);
    return j;
  }

  /** Start a run (main's `JobManager.submit` checks, in its order, then the browser's own). */
  async submit(b: JobBody): Promise<JobRecord> {
    if (b.tune) throw new HttpError(400, "auto-tune is not part of the browser app");
    const raw = await this.o.folders.resolve(b.raw_dir);
    const { files } = await listSegments(raw);
    if (!files.length) throw new HttpError(400, `no videos in ${b.raw_dir}: ${RawVideo.NAMING_RULE}`);
    if (b.fps !== null) checkFps(b.fps);
    checkRunName(b.name);
    const bad = unusedParams(b.params);
    if (bad.length) throw new HttpError(400, `${bad.join(", ")}: not used by the browser app`);
    if ([...this.jobs.values()].some((j) => !isFinal(j.status)))
      throw new HttpError(400, "a run is already going in this tab; wait for it or cancel it");
    const parent = parentVpath(b.raw_dir);
    if (parent === null)
      throw new HttpError(
        400,
        `${b.raw_dir} has no parent folder the app can write to: under Runs → Change folder pick the folder that holds ${splitVpath(b.raw_dir).root}, then choose the videos inside it`,
      );
    const outDir = joinVpath(parent, "processed", b.name);
    const processed = await (await this.o.folders.resolve(parent, "readwrite")).dir("processed", { create: true });
    // a folder with either marker counts as a run, so the leftovers of an interrupted run are
    // only replaced when asked for
    const holdsRun = await processed.dir(b.name).then(
      async (d) => (await exists(d, "report.json")) || (await exists(d, "params.json")),
      () => false,
    );
    if (holdsRun && !b.overwrite)
      throw new HttpError(400, `${outDir} already holds a run; choose another name or overwrite`);
    if (holdsRun) await processed.remove(b.name, { recursive: true });
    const out = await processed.dir(b.name, { create: true });
    const job: JobRecord = {
      id: crypto.randomUUID().replaceAll("-", "").slice(0, 10),
      raw_dir: b.raw_dir,
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
      tune: false,
      tune_args: {},
      fps: b.fps,
    };
    this.jobs.set(job.id, job);
    await this.o.kv.set(job.id, job);
    await this.o.folders.touch(splitVpath(parent).root, b.raw_dir);
    this.record(job.id, "status", { ...job });
    this.start(job, files, out, b);
    return job;
  }

  /** Hand the run to a worker; whatever stops it from starting fails the job. */
  private start(job: JobRecord, avis: File[], out: Dir, b: JobBody): void {
    const msg: ToWorker = {
      avis,
      inputs: [],
      out: (out as { handle?: FileSystemDirectoryHandle }).handle,
      params: b.params,
      keepStore: b.keep_store,
      rawDir: b.raw_dir,
      fps: b.fps,
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
    void this.keepAwake(job);
  }

  /** Keep the screen (and with it the page's worker) awake while the run goes, where the
   * browser has a Wake Lock. */
  private async keepAwake(job: JobRecord): Promise<void> {
    const lock = await globalThis.navigator?.wakeLock?.request("screen").catch(() => null);
    if (!lock) return;
    if (isFinal(job.status)) await lock.release().catch(() => undefined);
    else this.wake = lock;
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
    } else if (m.type === "log") this.record(job.id, "log", m.message);
    else if (m.type === "done" || m.type === "error") {
      job.status = m.type === "done" ? "done" : "failed";
      job.error = m.type === "error" ? m.message : "";
      job.ended = this.now();
      await this.finish(job);
    }
  }

  private async finish(job: JobRecord): Promise<void> {
    this.worker?.terminate();
    this.worker = null;
    await this.wake?.release().catch(() => undefined);
    this.wake = null;
    await this.o.kv.set(job.id, job);
    this.record(job.id, "status", { ...job });
  }

  async cancel(id: string): Promise<JobRecord> {
    const job = this.get(id);
    if (!isFinal(job.status)) {
      job.status = "cancelled";
      job.ended = this.now();
      await this.finish(job);
    }
    return job;
  }

  private record(id: string, name: string, data: unknown): void {
    const h = this.history.get(id) ?? [];
    this.history.set(id, h);
    h.push([name, data]);
    for (const l of this.listeners.get(id) ?? []) l(name, data);
  }

  /** Replay the job's events so far, then stream new ones (main's `/jobs/<id>/events`: a
   * status first, then the stages and log lines, then the status again when it changes). A job
   * from before a reload has no history: its state is the one status event. */
  events(id: string, emit: Emit): () => void {
    const job = this.get(id);
    const past = this.history.get(id);
    if (past) for (const [n, d] of past) emit(n, d);
    else emit("status", { ...job });
    const set = this.listeners.get(id) ?? new Set<Emit>();
    this.listeners.set(id, set);
    set.add(emit);
    return () => void set.delete(emit);
  }
}
