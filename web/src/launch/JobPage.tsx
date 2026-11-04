import { useEffect, useRef, useState } from "react";
import { getJSON, postJSON, subscribe } from "../api";
import { ErrorNote } from "../controls";
import { fmtDuration } from "../format";
import { go } from "../router";
import type { Job, RunListing } from "../types";
import { isFinal, type ProgressEvent, rate, type StageProgress, type StageRow, stageRows } from "./params";
import { StageRows } from "./StageRows";

/** The running stage's row with the browser run's counters: where it sits in the run ("window
 * 2/3") and its seconds so far (this window's on top of the earlier windows'). */
function liveRow(r: StageRow, events: ProgressEvent[], now: number): StageRow {
  const last = (event: ProgressEvent["event"]) =>
    [...events].reverse().find((e) => e.stage === r.stage && e.event === event);
  const start = last("start");
  const earlier = last("end")?.seconds ?? 0;
  const note = start?.note && start.note !== r.stage ? start.note : r.note; // "motion" says nothing more
  return { ...r, note, seconds: earlier + Math.max(0, now - (start?.t ?? now)) };
}

/** The running stage's bar: units done of all, rate and time left; a stage of one step (seeds,
 * init, CNMF, stitch) gets a moving bar without numbers. */
function LiveBar({ p }: { p: StageProgress }) {
  if (p.frames <= 1)
    return (
      <div className="barrow computing">
        <span />
        <div className="bar">
          <span />
        </div>
        <span />
      </div>
    );
  const { perSecond, eta } = rate(p.done, p.frames, p.seconds);
  const tiles = p.stage === "pass2";
  return (
    <div className="barrow">
      <span className="muted">
        {p.done} / {p.frames} {tiles ? "tiles" : "frames"} ·{" "}
        {tiles ? `${perSecond.toFixed(1)} tiles/s` : `${perSecond.toFixed(0)} fps`}
      </span>
      <div className="bar" role="progressbar" aria-valuenow={p.done} aria-valuemax={p.frames}>
        <span style={{ width: `${(100 * p.done) / p.frames}%` }} />
      </div>
      <span className="muted">{Number.isFinite(eta) ? `${fmtDuration(eta)} left` : "…"}</span>
    </div>
  );
}

export function JobPage({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [live, setLive] = useState<StageProgress | null>(null);
  const [err, setErr] = useState("");
  const [now, setNow] = useState(Date.now() / 1000);

  const gotStatus = useRef(false);
  const final = job !== null && isFinal(job.status);
  const [lost, setLost] = useState(false);

  useEffect(() => {
    gotStatus.current = false;
    setLost(false);
    getJSON<Job>(`/jobs/${id}`).then(
      (j) => {
        if (!gotStatus.current) setJob(j);
      },
      (e: Error) => setErr(e.message),
    );
    let close = () => {};
    close = subscribe(
      `/jobs/${id}/events`,
      {
        stage: (d) => {
          setEvents((x) => [...x, d as ProgressEvent]);
          setLive(null); // a stage (or window) began or ended: its counters start over
        },
        progress: (d) => setLive(d as StageProgress),
        log: (d) => setLog((x) => [...x.slice(-299), d as string]),
        status: (d) => {
          const j = d as Job;
          gotStatus.current = true;
          setJob(j);
          if (isFinal(j.status)) close();
        },
      },
      () =>
        getJSON<Job>(`/jobs/${id}`).then(
          (j) => {
            gotStatus.current = true;
            setJob(j);
            if (!isFinal(j.status)) setLost(true);
          },
          () => setLost(true),
        ),
    );
    return () => close();
  }, [id]);

  useEffect(() => {
    if (final) return;
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(tick);
  }, [final]);

  // a rerun or auto-tune job opens its source run on the Tune page
  const openRun = () =>
    getJSON<{ runs: RunListing[] }>(`/runs?root=${encodeURIComponent(job!.run_dir || job!.out_dir)}`).then(
      (r) => go({ name: "run", id: r.runs[0].id, page: job!.kind !== "run" || job!.tune ? "tune" : "overview" }),
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
  const rows = stageRows(events).map((r) => ({
    ...r,
    note: r.n !== undefined && r.n > 0 ? `${r.n} ${r.n === 1 ? "setting" : "settings"}` : undefined,
  }));
  const running = final ? undefined : rows.find((r) => r.running);
  const shown = running && live?.stage === running.stage ? live : null;
  return (
    <div className="page">
      <div className="row">
        <div>
          <span className="label">
            {job.kind === "rerun"
              ? "Window-2 rerun"
              : job.kind === "autotune"
                ? "Auto-tune"
                : job.tune
                  ? "Run + auto-tune"
                  : "Run"}{" "}
            · {job.status}
          </span>
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
            onClick={() => postJSON<Job>(`/jobs/${id}/cancel`, {}).then(setJob, (e: Error) => setErr(e.message))}
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
          <StageRows rows={rows.map((r) => (r === running ? liveRow(r, events, now) : r))} final={final} />
          {shown && <LiveBar p={shown} />}
          {events.length === 0 && (
            <span className="muted">{job.status === "queued" ? "Waiting for the run before it." : "Starting…"}</span>
          )}
        </section>
        <section className="card span2" aria-label="Log">
          <span className="label">Log</span>
          <pre className="log">{log.slice(-40).join("\n")}</pre>
        </section>
      </div>
      {lost && !final && <ErrorNote message="Live updates lost — reload to reconnect." />}
      {job.status === "failed" && <ErrorNote message={job.error || "the run failed"} />}
      {job.status === "interrupted" && <ErrorNote message={job.error} />}
      <span className="muted mono">{job.out_dir}</span>
    </div>
  );
}
