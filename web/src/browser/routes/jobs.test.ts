import { cpSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getJSON, postJSON, subscribe } from "../../api";
import { isFinal } from "../../launch/params";
import type { FromWorker } from "../../stream/protocol";
import type { Job } from "../../types";
import { Folders } from "../folders";
import { type JobBody, JobManager } from "../jobs";
import { memoryKV } from "../kv";
import { configure } from "../state";
import { NodeDir } from "../testing/nodeDir";
import { jobs, setJobs } from "./jobs";

class FakeWorker {
  onmessage: ((e: MessageEvent<FromWorker>) => void) | null = null;
  terminated = false;
  postMessage() {}
  terminate() {
    this.terminated = true;
  }
  say(m: FromWorker) {
    this.onmessage?.({ data: m } as MessageEvent<FromWorker>);
  }
}

const tick = () => new Promise((r) => setTimeout(r, 0));
let workers: FakeWorker[] = [];
let body: JobBody;

beforeEach(async () => {
  const top = mkdtempSync(join(tmpdir(), "jobs-routes-"));
  cpSync(resolve(__dirname, "../../../e2e/.data/raw"), join(top, "raw"), { recursive: true });
  const folders = new Folders(memoryKV(), async () => null);
  const name = await folders.add(new NodeDir(top), "picked");
  configure({ folders });
  workers = [];
  const m = new JobManager({
    kv: memoryKV(),
    folders,
    makeWorker: () => {
      const w = new FakeWorker();
      workers.push(w);
      return w as unknown as Worker;
    },
  });
  await m.load();
  setJobs(m);
  body = {
    raw_dir: `${name}/raw`,
    name: "t1",
    params: {},
    keep_store: false,
    overwrite: false,
    tune: false,
    tune_args: {},
    fps: null,
  };
});

describe("job routes", () => {
  it("submit, list, get and cancel answer main's job records", async () => {
    const job = await postJSON<Job>("/jobs", body);
    expect(job).toMatchObject({ status: "running", kind: "run", name: "t1" });
    expect((await getJSON<{ jobs: Job[] }>("/jobs")).jobs.map((j) => j.id)).toEqual([job.id]);
    expect(await getJSON<Job>(`/jobs/${job.id}`)).toMatchObject({ id: job.id, status: "running" });
    expect(await postJSON<Job>(`/jobs/${job.id}/cancel`, {})).toMatchObject({ id: job.id, status: "cancelled" });
    expect(workers[0].terminated).toBe(true);
  });

  it("answers 404 for an unknown job and 400 for a refused request", async () => {
    await expect(getJSON("/jobs/nope")).rejects.toMatchObject({ status: 404, message: "unknown job nope" });
    await expect(postJSON("/jobs/nope/cancel", {})).rejects.toMatchObject({ status: 404 });
    await expect(postJSON("/jobs", { ...body, params: { made_up: 1 } })).rejects.toMatchObject({
      status: 400,
      message: "made_up: not used by the browser app",
    });
  });

  it("POST /jobs/rerun and /jobs/autotune take main's bodies", async () => {
    const m = await jobs();
    const rerun = vi.spyOn(m, "submitRerun").mockResolvedValue({} as Job as never);
    const auto = vi.spyOn(m, "submitAutotune").mockResolvedValue({} as Job as never);
    await postJSON("/jobs/rerun", { run_dir: "r/processed/t", name: "r1", params: { sparse_snr: 0.5 } });
    expect(rerun).toHaveBeenCalledWith("r/processed/t", "r1", { sparse_snr: 0.5 });
    await postJSON("/jobs/autotune", { run_dir: "r/processed/t", tune_args: { diameter: "7" } });
    expect(auto).toHaveBeenCalledWith("r/processed/t", { diameter: "7" });
    await expect(postJSON("/jobs/rerun", { name: "r1" })).rejects.toMatchObject({ status: 422 });
  });
});

describe("job events", () => {
  it("opens the stream before any handler runs, so the page can keep its close", async () => {
    const job = await postJSON<Job>("/jobs", body);
    workers[0].say({ type: "log", message: "line" });
    workers[0].say({ type: "done", files: new Map(), seconds: 1 });
    await tick();
    expect((await getJSON<Job>(`/jobs/${job.id}`)).status).toBe("done");

    // JobPage sets `close` from subscribe()'s return value and its status handler calls it on the
    // final status: a status replayed synchronously would find `close` still unset
    const seen: [string, unknown][] = [];
    let close: () => void = () => {
      throw new Error("close called before subscribe returned");
    };
    close = subscribe(
      `/jobs/${job.id}/events`,
      {
        log: (d) => seen.push(["log", d]),
        status: (d) => {
          seen.push(["status", (d as Job).status]);
          if (isFinal((d as Job).status)) close();
        },
      },
      () => seen.push(["error", null]),
    );
    expect(seen).toEqual([]); // nothing delivered before subscribe returned
    await tick();
    expect(seen).toEqual([
      ["status", "running"],
      ["log", "line"],
      ["status", "done"],
    ]);
  });

  it("streams a running job's events to the subscriber until it closes", async () => {
    const job = await postJSON<Job>("/jobs", body);
    const seen: string[] = [];
    const close = subscribe(
      `/jobs/${job.id}/events`,
      { log: (d) => seen.push(String(d)), stage: (d) => seen.push((d as { stage: string }).stage) },
      () => {},
    );
    await tick();
    workers[0].say({ type: "stage", stage: "pass1", event: "start", t: 1 });
    workers[0].say({ type: "log", message: "a" });
    close();
    workers[0].say({ type: "log", message: "b" });
    expect(seen).toEqual(["pass1", "a"]);
  });

  it("lets go of a subscriber that closes while the past events are replayed", async () => {
    const job = await postJSON<Job>("/jobs", body); // running: more events will come
    workers[0].say({ type: "log", message: "before" });
    const seen: string[] = [];
    let close: () => void = () => {};
    close = subscribe(
      `/jobs/${job.id}/events`,
      {
        status: () => {
          seen.push("status");
          close(); // on the first event, in the middle of the replay
        },
        log: (d) => seen.push(String(d)),
      },
      () => {},
    );
    await tick();
    workers[0].say({ type: "log", message: "after" });
    expect(seen).toEqual(["status"]); // not the rest of the replay, not what came after
  });

  it("does not subscribe at all when the page closes before the job manager is ready", async () => {
    const job = await postJSON<Job>("/jobs", body);
    const seen: string[] = [];
    subscribe(`/jobs/${job.id}/events`, { log: (d) => seen.push(String(d)) }, () => {})(); // closed at once
    await tick();
    workers[0].say({ type: "log", message: "late" });
    expect(seen).toEqual([]);
  });

  it("ignores the stream of a job it does not know (the page's GET shows the 404)", async () => {
    const seen: unknown[] = [];
    subscribe("/jobs/nope/events", { status: (d) => seen.push(d) }, () => {});
    await tick();
    expect(seen).toEqual([]);
  });
});
