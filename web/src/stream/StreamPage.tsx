import { useEffect, useRef, useState } from "react";
import { fmtDuration } from "../format";
import { type ProgressEvent, type StageRow, stageRows } from "../launch/params";
import { StageRows } from "../launch/StageRows";
import { orderAvis, rate } from "./order";
import type { FromWorker, Stage } from "./protocol";

/** The running stage's counters, from the worker's last progress message. */
type Live = {
  stage: Stage;
  label: string;
  done: number;
  frames: number;
  seconds: number;
  stateMb: number;
  heapMb: number;
};
type Done = {
  files: [string, string, number][];
  summary: Record<string, unknown>;
  motion: Record<string, unknown> | null;
  pass2: Record<string, unknown> | null;
  seeds: Record<string, unknown> | null;
  init: Record<string, unknown> | null;
  cnmf: Record<string, unknown> | null;
  stitch: Record<string, unknown> | null;
  report: Record<string, unknown> | null;
  seconds: number;
};
type Run = {
  status: "running" | "done" | "failed";
  began: number; // seconds since the epoch
  ended: number | null;
  events: ProgressEvent[];
  live: Live | null;
  log: string[];
  done: Done | null;
  error: string;
};

/** The outputs worth a first look, in this order; the rest sit behind "all files". */
const MAIN_OUTPUTS: [string, string][] = [
  ["A_data.npy", "stitched footprints (CSR data; with A_indices and A_indptr, K x H*W)"],
  ["A_indices.npy", "footprint pixel indices"],
  ["A_indptr.npy", "footprint row pointers"],
  ["C.npy", "traces, (K, T) float32"],
  ["units.csv", "one row per unit: centroid, size, windows, brightness, kept"],
  ["shifts.npy", "motion, (T, 2) dy, dx"],
  ["Amerged_data.npy", "footprints after the streaming update and merge (with Amerged_indices, Amerged_indptr)"],
  ["report.json", "stages, seconds, counts (MiniMe's report.json fields)"],
  ["session.json", "the parameters of this run"],
];

/** MiniMe in the tab: pick the raw AVIs, get footprints and traces. */
export function StreamPage() {
  const [avis, setAvis] = useState<File[]>([]);
  const [inputs, setInputs] = useState<File[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [now, setNow] = useState(Date.now() / 1e3);
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);
  const running = run?.status === "running";
  useEffect(() => {
    if (!running) return;
    const tick = setInterval(() => setNow(Date.now() / 1e3), 1000);
    return () => clearInterval(tick);
  }, [running]);

  const start = () => {
    worker.current?.terminate();
    const w = new Worker(new URL("./stream.worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    setRun({
      status: "running",
      began: Date.now() / 1e3,
      ended: null,
      events: [],
      live: null,
      log: [],
      done: null,
      error: "",
    });
    const update = (f: (r: Run) => Run) => setRun((r) => (r ? f(r) : r));
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "log") {
        console.log(`[worker] ${m.message}`);
        update((r) => ({ ...r, log: [...r.log.slice(-299), m.message] }));
      } else if (m.type === "start") {
        update((r) => ({
          ...r,
          live: {
            stage: m.stage,
            label: m.label,
            done: 0,
            frames: m.frames,
            seconds: 0,
            stateMb: 0,
            heapMb: r.live?.heapMb ?? 0,
          },
        }));
      } else if (m.type === "stage") {
        const { type: _, ...ev } = m;
        update((r) => ({ ...r, events: [...r.events, ev] }));
      } else if (m.type === "progress") {
        update((r) => ({
          ...r,
          live: r.live && { ...r.live, done: m.done, seconds: m.seconds, stateMb: m.stateMb, heapMb: m.heapMb },
        }));
      } else if (m.type === "done") {
        const files: [string, string, number][] = [...m.files].map(([name, bytes]) => {
          const blob = bytes instanceof Blob ? bytes : new Blob([bytes]);
          return [name, URL.createObjectURL(blob), blob.size];
        });
        const json = (name: string) => {
          const b = m.files.get(name);
          return b instanceof Uint8Array ? (JSON.parse(new TextDecoder().decode(b)) as Record<string, unknown>) : null;
        };
        update((r) => ({
          ...r,
          status: "done",
          ended: Date.now() / 1e3,
          live: null,
          done: {
            files,
            summary: json("summary.json") ?? {},
            motion: json("summary_motion.json"),
            pass2: json("summary_pass2.json"),
            seeds: json("summary_seeds.json"),
            init: json("summary_init.json"),
            cnmf: json("summary_cnmf.json"),
            stitch: json("summary_stitch.json"),
            report: json("report.json"),
            seconds: m.seconds,
          },
        }));
        w.terminate();
      } else update((r) => ({ ...r, status: "failed", ended: Date.now() / 1e3, error: m.message }));
    };
    w.onerror = (e) =>
      update((r) => ({
        ...r,
        status: "failed",
        ended: Date.now() / 1e3,
        error: `worker failed: ${e.message || "could not start"}`,
      }));
    w.postMessage({ avis, inputs });
  };

  const ordered = orderAvis(avis);
  return (
    <main className="page">
      <h1>MiniMe in the browser</h1>
      <p className="muted">
        Pick the raw AVIs of a recording and press Run: motion correction, then MiniMe's full pipeline on every
        2.5-minute window (max projection, compression, seeds, init, CNMF loop), the windows' units stitched into one
        set of footprints, and one streaming pass over the whole movie for the traces and a footprint update. Files are
        read from disk in place; nothing is uploaded, and nothing is installed.
      </p>
      <section className="card" aria-label="Inputs">
        <div className="row" style={{ alignItems: "flex-start" }}>
          <label className="field" style={{ flex: "1 1 260px" }}>
            Raw AVIs (uncompressed GREY)
            <input
              type="file"
              accept=".avi"
              multiple
              aria-label="Raw AVIs"
              onChange={(e) => setAvis([...(e.target.files ?? [])])}
            />
            <span className="muted">
              {ordered.length} files, {(ordered.reduce((s, f) => s + f.size, 0) / 2 ** 30).toFixed(1)} GB
            </span>
          </label>
          <label className="field" style={{ flex: "1 1 260px" }}>
            Reference inputs, optional (a session, stage or footprint folder from the ref_*.py scripts)
            <input
              type="file"
              multiple
              aria-label="Inputs folder"
              // @ts-expect-error non-standard but supported by every browser
              webkitdirectory=""
              onChange={(e) => setInputs([...(e.target.files ?? [])])}
            />
            <span className="muted">{inputs.length ? `${inputs.length} files` : "none: default parameters"}</span>
          </label>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <button type="button" className="btn" disabled={running || !ordered.length} onClick={start}>
            Run
          </button>
        </div>
      </section>
      {run && <RunView run={run} now={now} />}
    </main>
  );
}

/** The run as `minime serve`'s job page shows one: status and elapsed time, one row per
 * stage (live counters on the running one), the log, then the results. */
function RunView({ run, now }: { run: Run; now: number }) {
  const final = run.status !== "running";
  const end = run.ended ?? now;
  const rows = stageRows(run.events).map((r) => (r.running && run.live ? liveRow(r, run.live, now, run.events) : r));
  const overall: StageRow = {
    stage: "overall",
    seconds: run.done?.report ? Number(run.done.report.wall_total) : final ? end - run.began : null,
    running: !final,
    failed: run.status === "failed",
    note: final ? undefined : `${fmtDuration(end - run.began)} so far`,
  };
  return (
    <>
      <div className="row">
        <span className="label">Run · {run.status}</span>
        <span className="spacer" />
        <span className="big" style={{ fontSize: 28 }}>
          {fmtDuration(end - run.began)}
        </span>
      </div>
      <div className="bento">
        <section className="card span2" aria-label="Stages">
          <span className="label">Stages</span>
          <StageRows rows={[...rows, overall]} final={final} />
          {run.live && !final && <LiveBar live={run.live} />}
        </section>
        <section className="card span2" aria-label="Log">
          <span className="label">Log</span>
          <pre className="log">{run.log.slice(-40).join("\n")}</pre>
        </section>
      </div>
      {run.status === "failed" && (
        <p role="alert" className="error">
          {run.error}
        </p>
      )}
      {run.done && <Results d={run.done} />}
    </>
  );
}

/** The running stage's row: elapsed seconds (this window's on top of the earlier windows'),
 * frames or tiles done, rate and wasm heap. */
function liveRow(r: StageRow, live: Live, now: number, events: ProgressEvent[]): StageRow {
  const began = [...events].reverse().find((e) => e.stage === r.stage && e.event === "start")?.t ?? now;
  const earlier = [...events].reverse().find((e) => e.stage === r.stage && e.event === "end")?.seconds ?? 0;
  const { fps } = rate(live.done, live.frames, live.seconds);
  const what = live.stage === "pass2" ? "tiles" : "frames";
  const counter =
    live.frames > 1
      ? ` · ${live.done} / ${live.frames} ${what} · ${live.stage === "pass2" ? `${fps.toFixed(1)} tiles/s` : `${fps.toFixed(0)} fps`}`
      : "";
  const state = live.stateMb ? ` · state ${live.stateMb.toFixed(0)} MB` : "";
  return {
    ...r,
    seconds: earlier + Math.max(0, now - began),
    note: `${live.label ? `${live.label}` : "running"}${counter}${state} · heap ${live.heapMb.toFixed(0)} MB`,
  };
}

function LiveBar({ live }: { live: Live }) {
  const { eta } = rate(live.done, live.frames, live.seconds);
  const short = live.frames === 1; // seeds, init, CNMF and the stitch: one step, no rate
  return (
    <div className={`barrow${short ? " computing" : ""}`} style={{ marginTop: 8 }}>
      <span className="muted">{short ? "" : "this stage"}</span>
      <div className="bar" role="progressbar" aria-valuenow={live.done} aria-valuemax={live.frames}>
        <span style={{ width: short ? undefined : `${(100 * live.done) / Math.max(live.frames, 1)}%` }} />
      </div>
      <span className="muted">{short ? "" : Number.isFinite(eta) ? `${fmtDuration(eta)} left` : "…"}</span>
    </div>
  );
}

function Results({ d }: { d: Done }) {
  const st = d;
  return (
    <section className="card" aria-label="Results">
      {st.stitch ? (
        <p>
          Done in {fmtDuration(st.seconds)}
          {st.motion &&
            `: motion for ${String(st.motion.frames)} frames (largest shift ${Number(st.motion.max_abs_shift).toFixed(1)} px)`}
          {st.motion ? ";" : ":"} {String(st.stitch.windows)} windows, {String(st.stitch.units_in)} window units
          stitched into {String(st.stitch.units_out)} units ({String(st.stitch.edges)} links)
          {st.summary.K !== undefined &&
            `; streaming pass over ${String(st.summary.T)} frames: ${String(st.summary.kept)} units kept by the footprint update, ${String(st.summary.merged)} after the merge`}
          .
        </p>
      ) : st.summary.stage === "pass1" ? (
        <p>
          Done in {fmtDuration(st.seconds)}: pass 1 on {String(st.summary.frames)} frames, {String(st.summary.blocks)}{" "}
          seed blocks, {String(st.summary.tiles)} tiles ({Number(st.summary.store_mb).toFixed(0)} MB in browser storage)
          {st.pass2 && `; pass 2 kept ${String(st.pass2.rank_total)} components over ${String(st.pass2.tiles)} tiles`}
          {st.seeds && `; ${String(st.seeds.seeds_merged)} merged seeds`}
          {st.init &&
            `; init: ${String(st.init.units_initA)} footprints, ${String(st.init.units_init_merge)} units after the merge`}
          {st.cnmf &&
            `; CNMF loop: ${String(st.cnmf.units_in)} → ${String(st.cnmf.units_final)} units in ${String(st.cnmf.iterations)} iterations`}
          .
        </p>
      ) : st.summary.stage === "motion" ? (
        <p>
          Done in {fmtDuration(st.seconds)}: motion for {String(st.summary.frames)} frames, largest shift{" "}
          {Number(st.summary.max_abs_shift).toFixed(2)} px.
        </p>
      ) : (
        <p>
          Done in {fmtDuration(st.seconds)}: {String(st.summary.K)} units in, {String(st.summary.kept)} kept,{" "}
          {String(st.summary.merged)} after merge; state {Number(st.summary.state_mb).toFixed(0)} MB.
        </p>
      )}
      <Outputs files={st.files} />
    </section>
  );
}

function Outputs({ files }: { files: [string, string, number][] }) {
  const byName = new Map(files.map((f) => [f[0], f]));
  const main = MAIN_OUTPUTS.filter(([name]) => byName.has(name));
  const rest = files.filter(([name]) => !main.some(([m]) => m === name));
  const link = ([name, url, n]: [string, string, number], what?: string) => (
    <li key={name}>
      <a href={url} download={name}>
        {name}
      </a>{" "}
      <span className="muted">
        {(n / 2 ** 20).toFixed(1)} MB{what && ` · ${what}`}
      </span>
    </li>
  );
  return (
    <>
      {main.length > 0 && (
        <ul aria-label="Output files">{main.map(([name, what]) => link(byName.get(name)!, what))}</ul>
      )}
      {rest.length > 0 && (
        <details>
          <summary className="muted">{main.length ? `all ${files.length} files` : `${files.length} files`}</summary>
          <ul aria-label={main.length ? "Other output files" : "Output files"}>{rest.map((f) => link(f))}</ul>
        </details>
      )}
    </>
  );
}
