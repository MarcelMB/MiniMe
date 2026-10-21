import { useEffect, useState } from "react";
import { getJSON, postJSON, tokenQuery } from "../api";
import { ErrorNote } from "../controls";
import { fmtDuration, stageLabel } from "../format";
import { go } from "../router";
import type { Job, RunListing } from "../types";
import { stageRows, type ProgressEvent } from "./params";

export function JobPage({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [err, setErr] = useState("");
  const [now, setNow] = useState(Date.now() / 1000);

  useEffect(() => {
    getJSON<Job>(`/jobs/${id}`).then(setJob, (e: Error) => setErr(e.message));
    const es = new EventSource(`/api/jobs/${id}/events?${tokenQuery()}`);
    es.addEventListener("stage", (e) =>
      setEvents((x) => [...x, JSON.parse((e as MessageEvent).data)]),
    );
    es.addEventListener("log", (e) =>
      setLog((x) => [...x.slice(-299), JSON.parse((e as MessageEvent).data)]),
    );
    es.addEventListener("status", (e) =>
      setJob(JSON.parse((e as MessageEvent).data)),
    );
    es.onerror = () => es.close();
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => {
      es.close();
      clearInterval(tick);
    };
  }, [id]);

  const openRun = () =>
    getJSON<{ runs: RunListing[] }>(
      `/runs?root=${encodeURIComponent(job!.out_dir)}`,
    ).then(
      (r) => go({ name: "run", id: r.runs[0].id, page: "overview" }),
      (e: Error) => setErr(e.message),
    );

  if (err)
    return (
      <div className="page">
        <ErrorNote message={err} />
      </div>
    );
  if (!job)
    return (
      <div className="page">
        <p className="muted">Loading the job…</p>
      </div>
    );
  const end = job.ended ?? now;
  return (
    <div className="page">
      <div className="row">
        <div>
          <span className="label">Run · {job.status}</span>
          <h1>{job.name}</h1>
        </div>
        <span className="spacer" />
        {job.started && (
          <span className="big" style={{ fontSize: 28 }}>
            {fmtDuration(end - job.started)}
          </span>
        )}
        {(job.status === "queued" || job.status === "running") && (
          <button
            className="btn"
            type="button"
            onClick={() => postJSON<Job>(`/jobs/${id}/cancel`, {}).then(setJob)}
          >
            Cancel
          </button>
        )}
        {job.status === "done" && (
          <button className="btn primary" type="button" onClick={openRun}>
            Open run
          </button>
        )}
      </div>
      <div className="bento">
        <section className="card span2" aria-label="Stages">
          <span className="label">Stages</span>
          {stageRows(events).map((r) => (
            <div className="barrow" key={r.stage}>
              <span>
                {r.running ? (
                  "…"
                ) : r.failed ? (
                  <span role="img" aria-label="failed">
                    ✗
                  </span>
                ) : (
                  <span role="img" aria-label="done">
                    ✓
                  </span>
                )}{" "}
                {stageLabel(r.stage)}
              </span>
              <span className="spacer" />
              <span className="mono">
                {r.seconds === null ? "" : fmtDuration(r.seconds)}
              </span>
            </div>
          ))}
          {events.length === 0 && (
            <span className="muted">
              {job.status === "queued"
                ? "Waiting for the run before it."
                : "Starting…"}
            </span>
          )}
        </section>
        <section className="card span2" aria-label="Log">
          <span className="label">Log</span>
          <pre className="log">{log.slice(-40).join("\n")}</pre>
        </section>
      </div>
      {job.status === "failed" && (
        <ErrorNote message={job.error || "the run failed"} />
      )}
      {job.status === "interrupted" && <ErrorNote message={job.error} />}
      <span className="muted mono">{job.out_dir}</span>
    </div>
  );
}
