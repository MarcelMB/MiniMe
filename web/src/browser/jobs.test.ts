import { cpSync, existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { FromWorker, ToWorker } from "../stream/protocol";
import { Folders } from "./folders";
import { type JobBody, JobManager } from "./jobs";
import { memoryKV } from "./kv";
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
    await (await out.dir("t1", { create: true })).write("report.json", "{}");
    await (await out.dir("t1")).write("params.json", "{}");
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

  it("treats the leftovers of an interrupted run (params.json only) as a run to overwrite", async () => {
    const { m, body, top } = await setup();
    const old = join(top, "processed", "t1");
    mkdirSync(old, { recursive: true });
    writeFileSync(join(old, "params.json"), "{}");
    await expect(m.submit(body)).rejects.toThrow("already holds a run");
    await expect(m.submit({ ...body, overwrite: true })).resolves.toMatchObject({ status: "running" });
    expect(existsSync(join(old, "params.json"))).toBe(false);
  });

  it("clears an overwritten run and leaves the raw folder alone", async () => {
    const { m, body, top } = await setup();
    const old = join(top, "processed", "t1");
    cpSync(join(top, "raw"), join(old, "stale"), { recursive: true });
    writeFileSync(join(old, "report.json"), "{}");
    await m.submit({ ...body, overwrite: true });
    expect(existsSync(join(old, "stale"))).toBe(false);
    expect(existsSync(join(top, "raw", "0.avi"))).toBe(true);
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
    // a job from before a reload has no history: one status event carries its state
    const kv = memoryKV();
    const a = await setup(kv);
    const old = await a.m.submit(a.body);
    const again = new JobManager({ kv, folders: a.folders, makeWorker: () => new FakeWorker() as unknown as Worker });
    await again.load();
    const late: [string, unknown][] = [];
    again.events(old.id, (n, d) => late.push([n, d]));
    expect(late.map(([n]) => n)).toEqual(["status"]);
    expect((late[0][1] as { status: string }).status).toBe("interrupted");
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
