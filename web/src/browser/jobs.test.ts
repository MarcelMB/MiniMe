import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fmtBytes } from "../format";
import type { FromWorker, ToWorker } from "../stream/protocol";
import type { Indexer } from "./avi";
import { type Dir, FileListDir, HandleDir } from "./dir";
import { Folders } from "./folders";
import { type JobBody, JobManager } from "./jobs";
import { memoryKV } from "./kv";
import { ASSUMED_UNITS, estimateRunBytes } from "./support";
import { diskHandle } from "./testing/diskHandle";
import { NodeDir } from "./testing/nodeDir";

class FakeWorker {
  onmessage: ((e: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  sent: unknown[] = [];
  terminated = false;
  postMessage(m: unknown) {
    this.sent.push(m);
  }
  terminate() {
    this.terminated = true;
  }
  say(m: FromWorker) {
    this.onmessage?.({ data: m } as MessageEvent<FromWorker>);
  }
}

const tick = () => new Promise((r) => setTimeout(r, 0));

async function setup(kv = memoryKV()) {
  const top = mkdtempSync(join(tmpdir(), "jobs-"));
  cpSync(resolve(__dirname, "../../e2e/.data/raw"), join(top, "raw"), { recursive: true });
  const folders = new Folders(memoryKV(), async () => null);
  const name = await folders.add(new NodeDir(top), "picked");
  const workers: FakeWorker[] = [];
  const makeWorker = () => {
    const w = new FakeWorker();
    workers.push(w);
    return w as unknown as Worker;
  };
  const m = new JobManager({ kv, folders, makeWorker });
  await m.load();
  const body: JobBody = {
    raw_dir: `${name}/raw`,
    name: "t1",
    params: {},
    keep_store: false,
    overwrite: false,
    tune: false,
    tune_args: {},
    fps: null,
  };
  return { m, workers, body, name, kv, folders, top, makeWorker };
}

describe("JobManager", () => {
  it("runs a job into <raw parent>/processed/<name> and reports its events", async () => {
    const { m, workers, body, name } = await setup();
    const job = await m.submit(body);
    expect(job).toMatchObject({ status: "running", kind: "run", tune: false, out_dir: `${name}/processed/t1` });
    const seen: [string, unknown][] = [];
    m.events(job.id, (n, d) => seen.push([n, d]));
    workers[0].say({ type: "stage", stage: "pass1", event: "start", t: 1 });
    workers[0].say({ type: "log", message: "hi" });
    workers[0].say({ type: "done", files: new Map(), seconds: 3 });
    await tick(); // the final status is recorded after the job is saved
    expect(seen.map(([n]) => n)).toEqual(["status", "stage", "log", "status"]);
    expect(seen[1][1]).toEqual({ stage: "pass1", event: "start", t: 1 }); // progress.jsonl's dict, no `type`
    expect(seen[2][1]).toBe("hi");
    expect(m.get(job.id).status).toBe("done");
  });

  it("refuses an existing run unless overwrite, with main's wording", async () => {
    const { m, body, name, folders } = await setup();
    const out = await (await folders.resolve(name)).dir("processed", { create: true });
    await (await out.dir("t1", { create: true })).write("report.json", JSON.stringify({ stages: [] }));
    await (await out.dir("t1")).write("params.json", JSON.stringify({ minime_version: "0.0.2" }));
    await expect(m.submit(body)).rejects.toThrow("already holds a run; choose another name or overwrite");
    await expect(m.submit({ ...body, overwrite: true })).resolves.toMatchObject({ status: "running" });
  });

  it("refuses parameters the browser pipeline does not read", async () => {
    const { m, body } = await setup();
    await expect(m.submit({ ...body, params: { gmm_refine: true, made_up: 1 } })).rejects.toThrow(
      /made_up: not used by the browser app/,
    );
  });

  it("turns a running job into interrupted after a reload", async () => {
    const kv = memoryKV();
    const a = await setup(kv);
    const job = await a.m.submit(a.body);
    const again = new JobManager({ kv, folders: a.folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await again.load();
    expect(again.get(job.id)).toMatchObject({ status: "interrupted" });
    expect(again.get(job.id).error).toBe("the server stopped during the run");
    // the interrupted record is saved: a third load finds it as it was
    const third = new JobManager({ kv, folders: a.folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await third.load();
    expect(third.get(job.id)).toMatchObject({ status: "interrupted", ended: again.get(job.id).ended });
  });

  it("cancel terminates the worker", async () => {
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    expect((await m.cancel(job.id)).status).toBe("cancelled");
    expect(workers[0].terminated).toBe(true);
  });
});

describe("JobManager submit checks (main's JobManager.submit)", () => {
  it("refuses a run name that is not one plain directory name", async () => {
    const { m, body } = await setup();
    for (const name of ["", ".", "..", "a/b", "../x", "a\\b"])
      await expect(m.submit({ ...body, name })).rejects.toMatchObject({
        status: 400,
        message: expect.stringMatching(/^invalid run name .*: use a single, plain directory name$/),
      });
    await expect(m.submit({ ...body, name: "../x" })).rejects.toThrow(
      "invalid run name '../x': use a single, plain directory name",
    );
  });

  it("refuses a frame rate outside (0, 1000] Hz", async () => {
    const { m, body } = await setup();
    await expect(m.submit({ ...body, fps: 0 })).rejects.toThrow("frame rate must be in (0, 1000] Hz, got 0.0");
    await expect(m.submit({ ...body, fps: 1001 })).rejects.toThrow("frame rate must be in (0, 1000] Hz, got 1001.0");
    await expect(m.submit({ ...body, fps: 29.97 })).resolves.toMatchObject({ status: "running" });
  });

  it("refuses a folder without numbered videos, with the naming rule", async () => {
    const { m, body, top } = await setup();
    const empty = join(top, "empty");
    mkdirSync(empty);
    cpSync(join(top, "raw", "0.avi"), join(empty, "clip.avi")); // an .avi without a number
    writeFileSync(join(empty, "notes.txt"), "x");
    await expect(m.submit({ ...body, raw_dir: `${body.raw_dir.split("/")[0]}/empty` })).rejects.toMatchObject({
      status: 400,
      message: expect.stringMatching(/^no videos in .*empty: each video's file name must start with a number/),
    });
  });

  it("refuses a second run while one is going, and takes one again once it is over", async () => {
    const { m, workers, body } = await setup();
    const first = await m.submit(body);
    await expect(m.submit({ ...body, name: "t2" })).rejects.toThrow("a run is already going in this tab");
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    expect(m.get(first.id).status).toBe("done");
    await expect(m.submit({ ...body, name: "t2" })).resolves.toMatchObject({ status: "running" });
  });

  it("refuses auto-tune, which the browser app does not have", async () => {
    const { m, body } = await setup();
    await expect(m.submit({ ...body, tune: true })).rejects.toMatchObject({ status: 400 });
  });

  it("refuses a raw folder that is itself a root: there is nowhere to write", async () => {
    const { m, body, folders, top } = await setup();
    const root = await folders.add(new NodeDir(join(top, "raw")), "picked");
    await expect(m.submit({ ...body, raw_dir: root })).rejects.toMatchObject({
      status: 400,
      message: expect.stringContaining("has no parent folder the app can write to"),
    });
    expect(existsSync(join(top, "processed"))).toBe(false);
  });

  it("clears an overwritten run and leaves the raw folder alone", async () => {
    const { m, body, top } = await setup();
    const old = join(top, "processed", "t1");
    cpSync(join(top, "raw"), join(old, "stale"), { recursive: true });
    writeFileSync(join(old, "report.json"), JSON.stringify({ stages: [] }));
    writeFileSync(join(old, "params.json"), JSON.stringify({ minime_version: "0.0.2" }));
    await m.submit({ ...body, overwrite: true });
    expect(existsSync(join(old, "stale"))).toBe(false);
    expect(existsSync(join(old, "report.json"))).toBe(false);
    expect(existsSync(join(top, "raw", "0.avi"))).toBe(true);
  });

  it("marks the run folder as a MiniMe run before the worker starts, like main's first write", async () => {
    const { m, body, top, name } = await setup();
    await m.submit({ ...body, fps: 25 });
    const doc = JSON.parse(readFileSync(join(top, "processed", "t1", "params.json"), "utf8"));
    expect(doc).toEqual({ source: "browser", raw_dir: `${name}/raw`, fps: 25 });
    expect(existsSync(join(top, "processed", "t1", "report.json"))).toBe(false); // so it is not listed as a run
  });

  it("starts again over an interrupted run of the same name, clearing every file it left", async () => {
    const { m, workers, body, top } = await setup();
    const first = await m.submit(body);
    const out = join(top, "processed", "t1");
    // what the worker had written when the run was cancelled: window factors, no report.json
    mkdirSync(join(out, "factors", "w00"), { recursive: true });
    mkdirSync(join(out, "factors", "w07"), { recursive: true });
    writeFileSync(join(out, "factors", "w00", "A.npy"), "x");
    writeFileSync(join(out, "factors", "w07", "A.npy"), "x");
    await m.cancel(first.id);
    expect(workers[0].terminated).toBe(true);
    await expect(m.submit(body)).resolves.toMatchObject({ status: "running" }); // no overwrite asked
    expect(existsSync(join(out, "factors"))).toBe(false);
    expect(existsSync(join(out, "params.json"))).toBe(true);
  });

  it("refuses a non-empty folder that holds no MiniMe run, with or without overwrite, and keeps it", async () => {
    const { m, body, top, name } = await setup();
    const old = join(top, "processed", "t1");
    mkdirSync(join(old, "factors", "w00"), { recursive: true });
    writeFileSync(join(old, "factors", "w00", "A.npy"), "x");
    writeFileSync(join(old, "notes.txt"), "mine");
    const why = `output dir ${name}/processed/t1 is not empty and holds no minime params.json / report.json; refusing to delete it (choose another name)`;
    await expect(m.submit(body)).rejects.toMatchObject({ status: 400, message: why });
    await expect(m.submit({ ...body, overwrite: true })).rejects.toMatchObject({ status: 400, message: why });
    expect(existsSync(join(old, "notes.txt"))).toBe(true);
    expect(existsSync(join(old, "factors", "w00", "A.npy"))).toBe(true);
  });

  it("decides what a run is by the markers' contents, not by file names", async () => {
    const { m, body, top } = await setup();
    const old = join(top, "processed", "t1");
    mkdirSync(old, { recursive: true });
    for (const name of ["params.json", "report.json"]) writeFileSync(join(old, name), "{}"); // not MiniMe's
    writeFileSync(join(old, "mine.txt"), "x");
    await expect(m.submit(body)).rejects.toThrow("already holds a run"); // main's is_run: both files
    await expect(m.submit({ ...body, overwrite: true })).rejects.toThrow("holds no minime params.json / report.json");
    expect(existsSync(join(old, "mine.txt"))).toBe(true);
    // one marker with MiniMe's contents is enough, for every producer main knows
    for (const [file, doc] of [
      ["params.json", { fastminian_version: "0.1" }],
      ["params.json", { source: "browser" }],
      ["report.json", { stages: [] }],
    ] as const) {
      const top2 = await setup();
      const o = join(top2.top, "processed", "t1");
      mkdirSync(o, { recursive: true });
      writeFileSync(join(o, file), JSON.stringify(doc));
      writeFileSync(join(o, "stale.txt"), "x");
      await top2.m.submit(top2.body);
      expect(existsSync(join(o, "stale.txt")), `${file} ${JSON.stringify(doc)}`).toBe(false);
    }
  });

  it("refuses a name that is a file in processed/", async () => {
    const { m, body, top, name } = await setup();
    mkdirSync(join(top, "processed"));
    writeFileSync(join(top, "processed", "t1"), "x");
    await expect(m.submit(body)).rejects.toMatchObject({
      status: 400,
      message: `output path ${name}/processed/t1 exists and is not a directory`,
    });
  });

  it("does not start when the old folder could not be cleared", async () => {
    const { body, top } = await setup();
    class Stuck extends NodeDir {
      async dir(n: string, o?: { create?: boolean }): Promise<Dir> {
        return new Stuck(((await super.dir(n, o)) as NodeDir).path);
      }
      async remove(): Promise<void> {} // as a handle whose removeEntry failed
    }
    const folders = new Folders(memoryKV(), async () => null);
    const root = await folders.add(new Stuck(top), "picked");
    const m = new JobManager({ kv: memoryKV(), folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    const old = join(top, "processed", "t1");
    mkdirSync(old, { recursive: true });
    writeFileSync(join(old, "report.json"), JSON.stringify({ stages: [] }));
    writeFileSync(join(old, "params.json"), JSON.stringify({ minime_version: "0.0.2" }));
    await expect(m.submit({ ...body, raw_dir: `${root}/raw`, overwrite: true })).rejects.toMatchObject({
      status: 500,
      message: `could not clear ${root}/processed/t1`,
    });
  });

  it("never writes inside the raw folder (main's _check_out_dir)", async () => {
    const { m, body, top, name } = await setup();
    // the raw folder is called processed: <parent>/processed/<name> would lie inside it
    cpSync(join(top, "raw"), join(top, "processed"), { recursive: true });
    await expect(m.submit({ ...body, raw_dir: `${name}/processed` })).rejects.toMatchObject({
      status: 400,
      message: `output dir ${name}/processed/t1 lies inside the raw dir ${name}/processed; choose another`,
    });
    expect(existsSync(join(top, "processed", "t1"))).toBe(false);
  });

  it("reads a raw folder given with a trailing slash as that folder", async () => {
    const { m, body, top, name } = await setup();
    const job = await m.submit({ ...body, raw_dir: `${name}/raw/` });
    expect(job).toMatchObject({ raw_dir: `${name}/raw`, out_dir: `${name}/processed/t1` });
    expect(existsSync(join(top, "raw", "processed"))).toBe(false);
    expect(existsSync(join(top, "processed", "t1", "params.json"))).toBe(true);
  });
});

describe("JobManager and its worker", () => {
  it("hands the worker the AVIs in number order, the run folder, parameters, raw folder and frame rate", async () => {
    const { m, workers, body, name, top } = await setup();
    // names that sort differently as text: 10 comes after 2 in number order
    cpSync(join(top, "raw", "2.avi"), join(top, "raw", "10.avi"));
    writeFileSync(join(top, "raw", "notes.txt"), "x");
    writeFileSync(join(top, "raw", "._0.avi"), "x");
    await m.submit({ ...body, params: { tile: 32 }, keep_store: true, fps: 30 });
    for (let i = 0; i < 100 && !workers[0].sent.length; i++) await tick();
    const msg = workers[0].sent[0] as ToWorker;
    expect(msg.avis.map((f) => f.name)).toEqual(["0.avi", "1.avi", "2.avi", "10.avi"]);
    expect(msg).toMatchObject({ inputs: [], params: { tile: 32 }, keepStore: true, rawDir: `${name}/raw`, fps: 30 });
  });

  it("fails the job with the worker's message and frees the slot", async () => {
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    workers[0].say({ type: "error", message: "out of memory" });
    await tick();
    expect(m.get(job.id)).toMatchObject({ status: "failed", error: "out of memory" });
    expect(m.get(job.id).ended).not.toBeNull();
    expect(workers[0].terminated).toBe(true);
    await expect(m.submit({ ...body, name: "t2" })).resolves.toMatchObject({ status: "running" });
  });

  it("fails the job when the worker script does not load", async () => {
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    workers[0].onerror?.({ message: "Failed to fetch module" } as ErrorEvent);
    await tick();
    expect(m.get(job.id)).toMatchObject({ status: "failed", error: expect.stringContaining("Failed to fetch module") });
  });

  it("fails the job, not the request, when no worker can be made", async () => {
    const { folders, body } = await setup();
    const m = new JobManager({
      kv: memoryKV(),
      folders,
      makeWorker: () => {
        throw new Error("no workers here");
      },
    });
    const job = await m.submit(body);
    await tick();
    expect(m.get(job.id)).toMatchObject({ status: "failed", error: "could not start the run: no workers here" });
  });

  it("fails the job, not the request, when the worker cannot be sent the run", async () => {
    const { folders, body } = await setup();
    const terminated: boolean[] = [];
    const broken = {
      postMessage() {
        throw new Error("could not be cloned");
      },
      terminate: () => terminated.push(true),
    };
    const m = new JobManager({ kv: memoryKV(), folders, makeWorker: () => broken as unknown as Worker });
    const job = await m.submit(body);
    await tick();
    expect(m.get(job.id)).toMatchObject({ status: "failed", error: "could not start the run: could not be cloned" });
    expect(terminated).toEqual([true]);
  });

  it("keeps a cancelled job cancelled when the worker's last message was already on its way", async () => {
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    await m.cancel(job.id);
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    expect(m.get(job.id).status).toBe("cancelled");
  });

  it("saves each job and lists the newest first, also after a reload", async () => {
    const kv = memoryKV();
    const { workers, body, folders, makeWorker } = await setup(kv);
    let t = 100;
    const timed = new JobManager({ kv, folders, makeWorker, now: () => t++ });
    await timed.load();
    const a = await timed.submit(body);
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    const b = await timed.submit({ ...body, name: "t2" });
    await timed.cancel(b.id);
    expect(timed.list().map((j) => j.id)).toEqual([b.id, a.id]);
    const again = new JobManager({ kv, folders, makeWorker });
    await again.load();
    expect(again.list().map((j) => [j.id, j.status])).toEqual([
      [b.id, "cancelled"],
      [a.id, "done"],
    ]);
  });

  it("marks the raw folder as the last one used in its root", async () => {
    const { m, body, folders, name } = await setup();
    await m.submit(body);
    expect((await folders.list()).find((r) => r.name === name)?.raw).toBe(`${name}/raw`);
  });

  it("records the job as main does, with the options of the request", async () => {
    const { m, body } = await setup();
    const job = await m.submit({ ...body, keep_store: true, fps: 25, params: { tile: 32 } });
    expect(job).toMatchObject({
      raw_dir: body.raw_dir,
      name: "t1",
      params: { tile: 32 },
      keep_store: true,
      overwrite: false,
      fps: 25,
      run_dir: "",
      error: "",
      ended: null,
      pid: null,
    });
    expect(typeof job.created).toBe("number");
    expect(job.started).toBe(job.created);
  });

  it("replays a finished job's events to a late subscriber, final status last", async () => {
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    workers[0].say({ type: "log", message: "one" });
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    const seen: [string, unknown][] = [];
    m.events(job.id, (n, d) => seen.push([n, d]));
    expect(seen.map(([n]) => n)).toEqual(["status", "log", "status"]);
    expect((seen[seen.length - 1][1] as { status: string }).status).toBe("done");
    // a job from before a reload whose run folder has no progress.jsonl: one status event
    const kv = memoryKV();
    const a = await setup(kv);
    const old = await a.m.submit(a.body);
    const again = new JobManager({ kv, folders: a.folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await again.load();
    const late: [string, unknown][] = [];
    again.events(old.id, (n, d) => late.push([n, d]));
    await tick();
    expect(late.map(([n]) => n)).toEqual(["status"]);
    expect((late[0][1] as { status: string }).status).toBe("interrupted");
  });

  it("replays a job from before a reload from its run folder's progress.jsonl, as main does, status last", async () => {
    const kv = memoryKV();
    let t = 1000;
    const a = await setup(kv);
    const timed = new JobManager({ kv, folders: a.folders, makeWorker: a.makeWorker, now: () => t });
    const job = await timed.submit(a.body); // started at 1000
    a.workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    const line = (stage: string, event: string, at: number, seconds?: number) =>
      JSON.stringify({ stage, event, t: at, ...(seconds !== undefined && { seconds }) });
    // what the worker wrote as it went (Windows line ends are read too), with a line of an older
    // run of the same folder (main skips what is older than the job) and a line cut off
    writeFileSync(
      join(a.top, "processed", "t1", "progress.jsonl"),
      [
        line("motion", "start", 900),
        line("pass1", "start", 1003),
        line("motion", "start", 1001),
        line("motion", "end", 1002, 1),
        line("pass1", "end", 1005, 2),
        '{"stage": "str',
      ].join("\r\n"),
    );
    t = 2000;
    const again = new JobManager({ kv, folders: a.folders, makeWorker: a.makeWorker });
    await again.load();
    const seen: [string, unknown][] = [];
    again.events(job.id, (n, d) => seen.push([n, d]));
    await tick();
    expect(
      seen.map(([n, d]) => (n === "stage" ? `${(d as { stage: string }).stage} ${(d as { event: string }).event}` : n)),
    ).toEqual(["motion start", "motion end", "pass1 start", "pass1 end", "status"]);
    expect(seen[1][1]).toEqual({ stage: "motion", event: "end", t: 1002, seconds: 1 });
    expect((seen[4][1] as { status: string }).status).toBe("done");
    // a subscriber that left before the file was read gets nothing
    const gone: string[] = [];
    again.events(job.id, (n) => gone.push(n))();
    await tick();
    expect(gone).toEqual([]);
  });

  it("answers an unknown job with 404", async () => {
    const { m } = await setup();
    expect(() => m.get("nope")).toThrow("unknown job nope");
    expect(() => m.events("nope", () => {})).toThrow(expect.objectContaining({ status: 404 }));
    await expect(m.cancel("nope")).rejects.toMatchObject({ status: 404 });
  });

  it("does not deliver to a subscriber that closed", async () => {
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    const seen: string[] = [];
    const close = m.events(job.id, (n) => seen.push(n));
    close();
    workers[0].say({ type: "log", message: "x" });
    expect(seen).toEqual(["status"]);
  });
});

/** A browser's screen wake lock: it lets go by itself when the tab is hidden. */
function fakeWakeLock() {
  const locks: { released: boolean; release(): Promise<void> }[] = [];
  return {
    locks,
    request: async (type: string) => {
      expect(type).toBe("screen");
      const lock = {
        released: false,
        release: async () => {
          lock.released = true;
        },
      };
      locks.push(lock);
      return lock;
    },
  };
}

describe("JobManager and the screen wake lock", () => {
  afterEach(() => vi.unstubAllGlobals());

  function stubBrowser() {
    const wakeLock = fakeWakeLock();
    const doc = Object.assign(new EventTarget(), { visibilityState: "visible" });
    vi.stubGlobal("navigator", { wakeLock });
    vi.stubGlobal("document", doc);
    const show = (state: "visible" | "hidden") => {
      doc.visibilityState = state;
      if (state === "hidden") for (const l of wakeLock.locks) l.released = true; // the browser lets go
      doc.dispatchEvent(new Event("visibilitychange"));
    };
    return { wakeLock, show };
  }

  it("holds the lock while the run goes and takes it again when the tab is shown after being hidden", async () => {
    const { wakeLock, show } = stubBrowser();
    const { m, body } = await setup();
    await m.submit(body);
    await tick();
    expect(wakeLock.locks).toHaveLength(1);
    show("hidden");
    await tick();
    expect(wakeLock.locks).toHaveLength(1); // nothing to take while hidden
    show("visible");
    await tick();
    expect(wakeLock.locks).toHaveLength(2);
    expect(wakeLock.locks[1].released).toBe(false);
    show("visible"); // already held: no second lock
    await tick();
    expect(wakeLock.locks).toHaveLength(2);
  });

  it("lets the lock go and stops listening when the run ends", async () => {
    const { wakeLock, show } = stubBrowser();
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    await tick();
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    expect(m.get(job.id).status).toBe("done");
    expect(wakeLock.locks.map((l) => l.released)).toEqual([true]);
    show("hidden");
    show("visible");
    await tick();
    expect(wakeLock.locks).toHaveLength(1); // no lock for a job that is over
  });

  it("does not leave a lock behind when the run ended before the browser answered", async () => {
    const { wakeLock } = stubBrowser();
    let answer: (lock: { released: boolean; release(): Promise<void> }) => void = () => {};
    const slow = new Promise<{ released: boolean; release(): Promise<void> }>((ok) => {
      answer = ok;
    });
    vi.stubGlobal("navigator", { wakeLock: { request: () => slow } });
    const { m, workers, body } = await setup();
    const job = await m.submit(body);
    await m.cancel(job.id);
    const late = {
      released: false,
      release: async () => {
        late.released = true;
      },
    };
    answer(late);
    await tick();
    expect(late.released).toBe(true);
    expect(workers[0].terminated).toBe(true);
    expect(wakeLock.locks).toHaveLength(0);
  });

  it("runs without a wake lock where the browser has none", async () => {
    vi.stubGlobal("navigator", {});
    const { m, body } = await setup();
    await expect(m.submit(body)).resolves.toMatchObject({ status: "running" });
  });
});

/** Raw folders picked on their own (nothing beside them), with or without Browser storage. */
async function rawRoot(o: { opfs?: boolean; pickOutput?: () => Promise<Dir>; index?: Indexer } = {}) {
  const top = mkdtempSync(join(tmpdir(), "jobs-"));
  cpSync(resolve(__dirname, "../../e2e/.data/raw"), join(top, "raw"), { recursive: true });
  const opfs = mkdtempSync(join(tmpdir(), "opfs-"));
  const folders = new Folders(memoryKV(), async () => (o.opfs ? new NodeDir(opfs) : null));
  const raw = await folders.add(new NodeDir(join(top, "raw")), "picked");
  const workers: FakeWorker[] = [];
  const makeWorker = () => {
    const w = new FakeWorker();
    workers.push(w);
    return w as unknown as Worker;
  };
  const m = new JobManager({ kv: memoryKV(), folders, makeWorker, pickOutput: o.pickOutput, index: o.index });
  const body: JobBody = {
    raw_dir: raw,
    name: "t1",
    params: {},
    keep_store: false,
    overwrite: false,
    tune: false,
    tune_args: {},
    fps: null,
  };
  return { m, workers, folders, top, opfs, raw, body, rawEntries: () => readdirSync(join(top, "raw")).sort() };
}

const abort = () => Object.assign(new Error("The user aborted a request."), { name: "AbortError" });

describe("JobManager and the output folder of a raw folder picked on its own (Review Focus 3)", () => {
  it("asks for the output folder before it awaits anything, and writes the run there", async () => {
    const out = mkdtempSync(join(tmpdir(), "out-"));
    const pickOutput = vi.fn(async () => new NodeDir(out) as Dir);
    const { m, body, raw, folders, rawEntries } = await rawRoot({ pickOutput });
    const before = rawEntries();
    const submitted = m.submit(body);
    expect(pickOutput).toHaveBeenCalledTimes(1); // the Start click's user activation is still there
    const job = await submitted;
    expect(job).toMatchObject({ status: "running", raw_dir: raw, out_dir: `${basename(out)}/t1` });
    expect(JSON.parse(readFileSync(join(out, "t1", "params.json"), "utf8"))).toMatchObject({
      source: "browser",
      raw_dir: raw,
    });
    expect(rawEntries()).toEqual(before); // nothing in the raw folder
    // the picked folder is a root now, with this raw folder as its last one
    expect((await folders.list()).find((r) => r.name === basename(out))?.raw).toBe(raw);
  });

  it("does not ask when the raw folder has a folder beside it", async () => {
    const pickOutput = vi.fn(async () => new NodeDir(mkdtempSync(join(tmpdir(), "out-"))) as Dir);
    const { m, workers, body, name } = await setupWith(pickOutput);
    await expect(m.submit(body)).resolves.toMatchObject({ out_dir: `${name}/processed/t1` });
    expect(pickOutput).not.toHaveBeenCalled();
    expect(workers).toHaveLength(1);
  });

  it("answers 400 naming what to pick when the dialog is cancelled, and writes nothing in the raw folder", async () => {
    const { m, body, top, opfs, rawEntries } = await rawRoot({
      opfs: true,
      pickOutput: async () => {
        throw abort();
      },
    });
    const before = rawEntries();
    await expect(m.submit(body)).rejects.toMatchObject({
      status: 400,
      message: expect.stringMatching(
        /has no parent folder the app can write to: .*pick the folder the run should be written to/,
      ),
    });
    expect(rawEntries()).toEqual(before);
    expect(existsSync(join(top, "processed"))).toBe(false);
    expect(readdirSync(opfs)).toEqual([]); // a browser that has the dialog does not fall back to Browser storage
  });

  it("answers the same 400 where there is neither a dialog nor Browser storage", async () => {
    const { m, body, top, rawEntries } = await rawRoot();
    const before = rawEntries();
    await expect(m.submit(body)).rejects.toMatchObject({
      status: 400,
      message: expect.stringMatching(
        /^.* has no parent folder the app can write to: .*pick the folder the run should be written to/,
      ),
    });
    expect(rawEntries()).toEqual(before);
    expect(existsSync(join(top, "processed"))).toBe(false);
  });
});

/** The usual `setup()` with a dialog the manager must not use. */
async function setupWith(pickOutput: () => Promise<Dir>) {
  const s = await setup();
  const workers: FakeWorker[] = [];
  const m = new JobManager({
    kv: memoryKV(),
    folders: s.folders,
    makeWorker: () => {
      const w = new FakeWorker();
      workers.push(w);
      return w as unknown as Worker;
    },
    pickOutput,
  });
  return { m, workers, body: s.body, name: s.name };
}

describe("JobManager and an output folder picked in the dialog that is, or lies inside, the raw folder", () => {
  /** The e2e raw folder as a root of handles (`top/raw`), and a dialog that returns the folder at
   * `pick` (a path under `top`, made if it is not there). */
  async function withDialog(pick: string[]) {
    const top = mkdtempSync(join(tmpdir(), "jobs-"));
    cpSync(resolve(__dirname, "../../e2e/.data/raw"), join(top, "raw"), { recursive: true });
    mkdirSync(join(top, ...pick), { recursive: true });
    const folders = new Folders(memoryKV(), async () => null);
    await folders.add(new HandleDir(diskHandle(["top", "raw"], join(top, "raw"))), "picked");
    const picked = new HandleDir(diskHandle(["top", ...pick], join(top, ...pick)));
    const workers: FakeWorker[] = [];
    const m = new JobManager({
      kv: memoryKV(),
      folders,
      pickOutput: async () => picked,
      makeWorker: () => {
        const w = new FakeWorker();
        workers.push(w);
        return w as unknown as Worker;
      },
    });
    const body: JobBody = {
      raw_dir: "raw",
      name: "t1",
      params: {},
      keep_store: false,
      overwrite: false,
      tune: false,
      tune_args: {},
      fps: null,
    };
    return { m, folders, workers, body, top };
  }
  const roots = async (f: Folders) => (await f.list()).map((r) => r.name);

  it("refuses the raw folder itself, in main's words, and does not remember it as a second root", async () => {
    const { m, folders, workers, body, top } = await withDialog(["raw"]);
    const before = readdirSync(join(top, "raw")).sort();
    await expect(m.submit(body)).rejects.toMatchObject({
      status: 400,
      message: "output dir raw/t1 lies inside the raw dir raw; choose another",
    });
    expect(readdirSync(join(top, "raw")).sort()).toEqual(before);
    expect(await roots(folders)).toEqual(["raw"]);
    expect(workers).toHaveLength(0);
  });

  it("refuses a folder made inside the raw folder, and writes nothing in it", async () => {
    const { m, folders, workers, body, top } = await withDialog(["raw", "new folder"]);
    await expect(m.submit(body)).rejects.toMatchObject({
      status: 400,
      message: "output dir new folder/t1 lies inside the raw dir raw; choose another",
    });
    expect(readdirSync(join(top, "raw", "new folder"))).toEqual([]);
    expect(await roots(folders)).toEqual(["raw"]);
    expect(workers).toHaveLength(0);
  });

  it("takes a folder beside the raw folder, also one whose name starts like it", async () => {
    for (const pick of [["out"], ["raw2"]]) {
      const { m, folders, workers, body, top } = await withDialog(pick);
      await expect(m.submit(body)).resolves.toMatchObject({ status: "running", out_dir: `${pick[0]}/t1` });
      expect(existsSync(join(top, pick[0], "t1", "params.json"))).toBe(true);
      expect(readdirSync(join(top, "raw")).sort()).toEqual(["0.avi", "1.avi", "2.avi"]);
      expect((await roots(folders)).sort()).toEqual(["raw", pick[0]].sort());
      expect(workers).toHaveLength(1);
    }
  });
});

describe("JobManager and folders picked in the dialog inside a Runs folder (spec section 3)", () => {
  /** `top/sess` (the e2e raw videos in `sess/raw`) remembered as a Runs folder of handles. */
  async function runsFolder(pickOutput: () => Promise<Dir>) {
    const top = mkdtempSync(join(tmpdir(), "jobs-"));
    cpSync(resolve(__dirname, "../../e2e/.data/raw"), join(top, "sess", "raw"), { recursive: true });
    mkdirSync(join(top, "sess", "out"));
    const h = (...path: string[]) => new HandleDir(diskHandle(path, join(top, ...path)));
    const folders = new Folders(memoryKV(), async () => null);
    expect(await folders.pick(h("sess"))).toBe("sess");
    const workers: FakeWorker[] = [];
    const m = new JobManager({
      kv: memoryKV(),
      folders,
      pickOutput,
      makeWorker: () => {
        const w = new FakeWorker();
        workers.push(w);
        return w as unknown as Worker;
      },
    });
    const body: JobBody = {
      raw_dir: "",
      name: "t1",
      params: {},
      keep_store: false,
      overwrite: false,
      tune: false,
      tune_args: {},
      fps: null,
    };
    return { m, folders, workers, body, top, h };
  }

  it("runs a raw folder picked inside it into <raw parent>/processed/<name>, without asking for an output folder", async () => {
    const pickOutput = vi.fn(async (): Promise<Dir> => {
      throw new Error("not asked");
    });
    const { m, folders, body, top, h } = await runsFolder(pickOutput);
    const raw = await folders.pick(h("sess", "raw")); // New run → Choose folder
    expect(raw).toBe("sess/raw");
    await expect(m.submit({ ...body, raw_dir: raw })).resolves.toMatchObject({ out_dir: "sess/processed/t1" });
    expect(pickOutput).not.toHaveBeenCalled();
    expect(existsSync(join(top, "sess", "processed", "t1", "params.json"))).toBe(true);
    expect((await folders.list()).map((r) => r.name)).toEqual(["sess"]);
  });

  it("writes into an output folder picked inside a Runs folder at its path there", async () => {
    const { m, folders, body, top, h } = await runsFolder(async () => h("sess", "out"));
    expect(await folders.add(h("sess", "raw"), "picked")).toBe("raw"); // a raw folder on its own
    await expect(m.submit({ ...body, raw_dir: "raw" })).resolves.toMatchObject({ out_dir: "sess/out/t1" });
    expect(existsSync(join(top, "sess", "out", "t1", "params.json"))).toBe(true);
    expect((await folders.list()).map((r) => r.name).sort()).toEqual(["raw", "sess"]);
  });
});

describe("JobManager and Browser storage", () => {
  afterEach(() => vi.unstubAllGlobals());
  const storage = (estimate: { quota: number; usage: number } | null) => {
    const persist = vi.fn(async () => true);
    vi.stubGlobal("navigator", { storage: { ...(estimate && { estimate: async () => estimate }), persist } });
    return persist;
  };
  /** What a run of the e2e raw folder (64 x 64 px, 300 frames, one window) is expected to need. */
  const need = estimateRunBytes(ASSUMED_UNITS, 300, 64 * 64, 1);

  it("writes the run into Browser storage in a browser without the folder dialog", async () => {
    storage({ quota: 1e12, usage: 0 });
    const { m, body, opfs, rawEntries, raw } = await rawRoot({ opfs: true });
    const before = rawEntries();
    const job = await m.submit(body);
    expect(job.out_dir).toBe("Browser storage/t1");
    expect(existsSync(join(opfs, "t1", "params.json"))).toBe(true);
    expect(rawEntries()).toEqual(before);
    expect(job.raw_dir).toBe(raw);
  });

  it("refuses a run that does not fit, with the numbers, before it touches the old run or starts", async () => {
    const persist = storage({ quota: need + 100_000, usage: 200_000 });
    const { m, workers, body, opfs } = await rawRoot({ opfs: true });
    mkdirSync(join(opfs, "t1"));
    writeFileSync(join(opfs, "t1", "params.json"), JSON.stringify({ source: "browser" }));
    writeFileSync(join(opfs, "t1", "report.json"), JSON.stringify({ stages: [] }));
    await expect(m.submit({ ...body, overwrite: true })).rejects.toMatchObject({
      status: 400,
      message: `this run needs about ${fmtBytes(need)} of browser storage, ${fmtBytes(need - 100_000)} are free; pick a folder on disk under Runs → Change folder`,
    });
    expect(existsSync(join(opfs, "t1", "report.json"))).toBe(true); // the run it would replace is still there
    expect(workers).toHaveLength(0);
    expect(persist).not.toHaveBeenCalled();
  });

  it("counts the tile store when it is kept", async () => {
    storage({ quota: need + 100_000, usage: 0 });
    const { m, body } = await rawRoot({ opfs: true });
    await expect(m.submit({ ...body, keep_store: true })).rejects.toThrow(
      `this run needs about ${fmtBytes(estimateRunBytes(ASSUMED_UNITS, 300, 64 * 64, 1, true))}`,
    );
  });

  it("asks the browser to keep its storage once, and starts a run that fits", async () => {
    const persist = storage({ quota: need * 2, usage: 0 });
    const { m, workers, body } = await rawRoot({ opfs: true });
    const first = await m.submit(body);
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    expect(m.get(first.id).status).toBe("done");
    await m.submit({ ...body, name: "t2" });
    expect(workers).toHaveLength(2);
    expect(persist).toHaveBeenCalledTimes(1);
  });

  it("runs where the browser cannot say how much room there is", async () => {
    const persist = storage(null);
    const { m, body } = await rawRoot({ opfs: true });
    await expect(m.submit(body)).resolves.toMatchObject({ out_dir: "Browser storage/t1" });
    expect(persist).toHaveBeenCalledTimes(1);
  });

  it("does not check a folder on disk", async () => {
    const persist = storage({ quota: 1, usage: 0 });
    const out = mkdtempSync(join(tmpdir(), "out-"));
    const { m, body } = await rawRoot({ opfs: true, pickOutput: async () => new NodeDir(out) });
    await expect(m.submit(body)).resolves.toMatchObject({ out_dir: `${basename(out)}/t1` });
    expect(persist).not.toHaveBeenCalled();
  });
});

describe("JobManager and the frame size", () => {
  /** Every video is 100 frames of `h` x `w` px. */
  const sized =
    (h: number, w: number): Indexer =>
    async (files) =>
      files.map(() => ({ n: 100, height: h, width: w, offsets: [], sizes: [] }));
  const odd = (index: Indexer, params: Record<string, unknown>) =>
    rawRoot({ index }).then(({ m, body, top }) => ({ run: () => m.submit({ ...body, params }), top }));

  it("refuses an odd height or width for the band-pass image, in the preview's words, before anything is written", async () => {
    const { run, top } = await odd(sized(63, 64), {});
    await expect(run()).rejects.toMatchObject({
      status: 400,
      message: "63 x 64 frames: the band-pass registration image needs an even size",
    });
    const wide = await odd(sized(64, 65), {});
    await expect(wide.run()).rejects.toThrow("64 x 65 frames: the band-pass registration image needs an even size");
    expect(readdirSync(top)).toEqual(["raw"]);
  });

  it("refuses an odd size for the 2x downsampling of the motion estimate when the band-pass is off", async () => {
    const { run } = await odd(sized(63, 64), { motion_before_tophat: false });
    await expect(run()).rejects.toThrow("63 x 64 frames: 2x downsampling needs an even size");
  });

  it("takes an odd size when neither step needs an even one", async () => {
    const { run } = await odd(sized(63, 65), { motion_before_tophat: false, motion_down: 1 });
    await expect(run()).rejects.toThrow("has no parent folder"); // past the size check: on to the output folder
  });
});

describe("JobManager and folder permissions", () => {
  it("answers 403 Reconnect, without asking, when the folder lost its write permission", async () => {
    const top = mkdtempSync(join(tmpdir(), "jobs-"));
    cpSync(resolve(__dirname, "../../e2e/.data/raw"), join(top, "raw"), { recursive: true });
    const d: Dir = new NodeDir(top);
    const asked: [string, boolean][] = [];
    d.permission = async (mode, ask) => {
      asked.push([mode, ask]);
      return mode === "read";
    };
    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(d, "picked");
    const m = new JobManager({ kv: memoryKV(), folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await expect(
      m.submit({
        raw_dir: `${name}/raw`,
        name: "t1",
        params: {},
        keep_store: false,
        overwrite: false,
        tune: false,
        tune_args: {},
        fps: null,
      }),
    ).rejects.toMatchObject({ status: 403, message: expect.stringMatching(/^Reconnect the folder /) });
    expect(asked.some(([, ask]) => ask)).toBe(false); // a prompt here would have no user activation
    expect(asked).toContainEqual(["readwrite", false]);
    expect(existsSync(join(top, "processed"))).toBe(false);
  });

  it("passes on a 403 from the raw folder as it is, rather than saying nothing was found", async () => {
    const top = mkdtempSync(join(tmpdir(), "jobs-"));
    cpSync(resolve(__dirname, "../../e2e/.data/raw"), join(top, "raw"), { recursive: true });
    const d: Dir = new NodeDir(top);
    d.permission = async () => false;
    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(d, "picked");
    const m = new JobManager({ kv: memoryKV(), folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await expect(
      m.submit({
        raw_dir: `${name}/raw`,
        name: "t1",
        params: {},
        keep_store: false,
        overwrite: false,
        tune: false,
        tune_args: {},
        fps: null,
      }),
    ).rejects.toMatchObject({ status: 403, message: expect.stringMatching(/^Reconnect the folder /) });
    expect(existsSync(join(top, "processed"))).toBe(false);
  });

  it("says a read-only upload cannot hold the run, not that it needs reconnecting", async () => {
    const files = ["0.avi", "1.avi", "2.avi"].map((n) => {
      const f = new File([readFileSync(resolve(__dirname, "../../e2e/.data/raw", n))], n);
      Object.defineProperty(f, "webkitRelativePath", { value: `sess/raw/${n}` });
      return f;
    });
    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(FileListDir.fromFiles(files), "files");
    const m = new JobManager({ kv: memoryKV(), folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await expect(
      m.submit({
        raw_dir: `${name}/raw`,
        name: "t1",
        params: {},
        keep_store: false,
        overwrite: false,
        tune: false,
        tune_args: {},
        fps: null,
      }),
    ).rejects.toMatchObject({ status: 403, message: expect.stringMatching(/was opened read-only/) });
  });
});
