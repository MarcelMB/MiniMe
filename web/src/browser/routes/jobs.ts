import { DEVICES, type Device } from "../../gpu/plan";
import { addEventProvider } from "../events";
import { type JobBody, JobManager } from "../jobs";
import { idbKV } from "../kv";
import { HttpError, router } from "../router";
import { opfsDir, state } from "../state";

let manager: Promise<JobManager> | null = null;

/** The tab's job manager, with the saved jobs loaded (a request that comes while they load waits). */
export function jobs(): Promise<JobManager> {
  manager ??= (async () => {
    const m = new JobManager({
      kv: idbKV("jobs"),
      folders: state.folders,
      makeWorker: () => new Worker(new URL("../../stream/stream.worker.ts", import.meta.url), { type: "module" }),
      locks: globalThis.navigator?.locks ?? null,
      scratch: () => opfsDir("scratch"),
    });
    await m.load();
    return m;
  })();
  return manager;
}

/** Tests replace the manager (with fake workers and a memory `kv`). */
export function setJobs(m: JobManager): void {
  manager = Promise.resolve(m);
}

/** `routes_jobs.JobBody`'s defaults over the request. */
function jobBody(raw: unknown): JobBody {
  const b = (raw ?? {}) as Partial<JobBody>;
  if (typeof b.raw_dir !== "string" || typeof b.name !== "string")
    throw new HttpError(422, "raw_dir and name must be text");
  if (b.device !== undefined && !DEVICES.includes(b.device as Device))
    throw new HttpError(422, `device must be one of ${DEVICES.join(", ")}`);
  return {
    params: {},
    keep_store: false,
    overwrite: false,
    tune: false,
    tune_args: {},
    fps: null,
    ...b,
    raw_dir: b.raw_dir,
    name: b.name,
  };
}

/** The lists read the saved jobs again: another tab saves the jobs it runs. */
router.add("GET", "/jobs", async () => {
  const m = await jobs();
  await m.refresh();
  return { json: { jobs: m.list() } };
});
router.add("POST", "/jobs", async (q) => ({ json: await (await jobs()).submit(jobBody(q.body)) }));
router.add("GET", "/jobs/:jid", async (_q, p) => {
  const m = await jobs();
  await m.refresh();
  return { json: m.get(p.jid) };
});
router.add("POST", "/jobs/:jid/cancel", async (_q, p) => ({ json: await (await jobs()).cancel(p.jid) }));

/** `/jobs/<id>/events`. The events reach the page after `openEvents` has returned (the manager
 * loads first), so `JobPage` has its `close` by the time the final status arrives. */
addEventProvider((path, emit) => {
  const m = /^\/jobs\/([^/]+)\/events$/.exec(path);
  if (!m) return null;
  let close = () => {};
  let closed = false;
  jobs()
    .then((j) => {
      if (closed) return;
      // a handler may close the stream while the past events are replayed, before `events`
      // has returned the way to unsubscribe: nothing is delivered after that, and it unsubscribes
      close = j.events(decodeURIComponent(m[1]), (name, data) => {
        if (!closed) emit(name, data);
      });
      if (closed) close();
    })
    .catch(() => {
      // an unknown job: the page's own GET /jobs/<id> shows the 404
    });
  return () => {
    closed = true;
    close();
  };
});
