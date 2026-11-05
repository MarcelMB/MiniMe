import { useEffect, useRef, useState } from "react";
import { getJSON, postJSON, subscribe } from "../api";
import { ErrorNote } from "../controls";
import { fmtDuration } from "../format";
import { go } from "../router";
import type { Job, RunListing } from "../types";
import { fmtLeft } from "./eta";
import { LiveRun } from "./live";
import { isFinal, type ProgressEvent, type RunPlan, type StageProgress, type StageRow, stageRows } from "./params";
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

const UNIT_NAMES = { frames: "frames", tiles: "tiles", steps: "windows" } as const;

/** The running stage's bar over the whole run (a stage that runs per window counts every
 * window): units done of all, smoothed rate, and time left or a stall; its stripes drift
 * backwards (Harrison et al. 2010). A stage of one step gets a moving bar without numbers. */
function LiveBar({
  p,
  perSecond,
  left,
  stalled,
}: {
  p: StageProgress;
  perSecond: number | null;
  left: number | null;
  stalled: number | null;
}) {
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
  const unit = UNIT_NAMES[p.unit];
  const speed =
    perSecond === null || p.unit === "steps"
      ? ""
      : p.unit === "tiles"
        ? ` · ${perSecond.toFixed(1)} tiles/s`
        : ` · ${perSecond.toFixed(0)} fps`;
  // never full while the stage runs: it ends with its own row
  const share = Math.min(p.done / p.frames, 0.995);
  return (
    <div className="barrow">
      <span className="muted">
        {p.done} / {p.frames} {unit}
        {speed}
      </span>
      <div className="bar live" role="progressbar" aria-valuenow={p.done} aria-valuemax={p.frames}>
        <span style={{ width: `${100 * share}%` }} />
      </div>
      <span className="muted" style={{ width: 150 }}>
        {stalled !== null
          ? `no new ${unit} for ${fmtDuration(stalled)}`
          : left !== null
            ? `${fmtLeft(left)} left`
            : "estimating…"}
      </span>
    </div>
  );
}

/** The whole run's time left under the clock, worded roughly. */
function runLeftText(left: number | null): string {
  if (left === null) return "estimating time left…";
  return left < 10 ? "almost done" : `about ${fmtLeft(left)} left`;
}

export function JobPage({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const live = useRef(new LiveRun());
  const [, setVersion] = useState(0);
  const [err, setErr] = useState("");
  const [now, setNow] = useState(Date.now() / 1000);

  const gotStatus = useRef(false);
  const final = job !== null && isFinal(job.status);
  const [lost, setLost] = useState(false);

  useEffect(() => {
    gotStatus.current = false;
    setLost(false);
    live.current = new LiveRun();
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
          const e = d as ProgressEvent;
          if (e.event === "start") live.current.started(e.stage, e.t);
          setEvents((x) => [...x, e]);
        },
        // the browser run's counters (`minime serve` sends none)
        progress: (d) => {
          live.current.progress(d as StageProgress, Date.now() / 1000);
          setVersion((v) => v + 1);
        },
        plan: (d) => live.current.setPlan(d as RunPlan),
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

  // a finished job marks the tab, so a run left in a background tab shows when it is done
  const status = job?.status;
  const name = job?.name;
  useEffect(() => {
    if (!status || !name || !isFinal(status)) return;
    const before = document.title;
    const mark = status === "done" ? "✓" : status === "failed" ? "✗" : "–";
    document.title = `${mark} ${name} · ${before}`;
    return () => {
      document.title = before;
    };
  }, [status, name]);

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
  const wall = Date.now() / 1000;
  const shown = running ? live.current.get(running.stage) : null;
  const runText = job.status === "running" && live.current.hasPlan() ? runLeftText(live.current.runLeft(wall)) : null;
  // the browser run's device (its "device: ..." log line), main's runs have none
  const device = [...log].reverse().find((l) => l.startsWith("device: "));
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
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
            <span className="big" style={{ fontSize: 28 }}>
              {fmtDuration(end - job.started)}
            </span>
            {runText && <span className="muted">{runText}</span>}
          </div>
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
          {device && <span className="muted">Ran on {device.slice(8).replace(/ at [\d.]+ s$/, "")}</span>}
          <StageRows rows={rows.map((r) => (r === running ? liveRow(r, events, now) : r))} final={final} />
          {shown && (
            <LiveBar
              key={shown.stage}
              p={shown}
              perSecond={live.current.perSecond(shown.stage)}
              left={live.current.stageLeft(shown.stage, wall)}
              stalled={live.current.stalled(shown.stage, wall)}
            />
          )}
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
