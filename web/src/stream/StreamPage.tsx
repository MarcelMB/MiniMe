import { useEffect, useRef, useState } from "react";
import { fmtDuration } from "../format";
import { orderAvis, rate } from "./order";
import type { FromWorker } from "./protocol";

type State =
  | { phase: "idle" }
  | { phase: "running"; stage: string; frames: number; units: number; done: number; seconds: number; stateMb: number }
  | {
      phase: "done";
      files: [string, string, number][];
      summary: Record<string, unknown>;
      pass2: Record<string, unknown> | null;
      seconds: number;
    }
  | { phase: "error"; message: string };

/** Streaming MiniMe in the tab: one pass over raw AVIs with fixed footprints. */
export function StreamPage() {
  const [avis, setAvis] = useState<File[]>([]);
  const [inputs, setInputs] = useState<File[]>([]);
  const [st, setSt] = useState<State>({ phase: "idle" });
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);

  const start = () => {
    worker.current?.terminate();
    const w = new Worker(new URL("./stream.worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "log") console.log(`[worker] ${m.message}`);
      else if (m.type === "start")
        setSt({ phase: "running", stage: m.stage, frames: m.frames, units: m.units, done: 0, seconds: 0, stateMb: 0 });
      else if (m.type === "progress")
        setSt((s) => (s.phase === "running" ? { ...s, done: m.done, seconds: m.seconds, stateMb: m.stateMb } : s));
      else if (m.type === "done") {
        const files: [string, string, number][] = [...m.files].map(([name, bytes]) => [
          name,
          URL.createObjectURL(new Blob([bytes])),
          bytes.byteLength,
        ]);
        const json = (name: string) => {
          const b = m.files.get(name);
          return b ? (JSON.parse(new TextDecoder().decode(b)) as Record<string, unknown>) : null;
        };
        setSt({ phase: "done", files, summary: json("summary.json") ?? {}, pass2: json("summary_pass2.json"), seconds: m.seconds });
        w.terminate();
      } else setSt({ phase: "error", message: m.message });
    };
    w.onerror = (e) => setSt({ phase: "error", message: `worker failed: ${e.message || "could not start"}` });
    w.postMessage({ avis, inputs });
  };

  const ordered = orderAvis(avis);
  const running = st.phase === "running";
  return (
    <main className="page">
      <h1>Stream in the browser</h1>
      <p className="muted">
        One pass over the raw movie with fixed footprints: per-frame traces, running sums and one footprint
        update. Files are read from disk in place; nothing is uploaded.
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
            Inputs folder (footprints, shifts, mean, noise; or motion parameters only)
            <input
              type="file"
              multiple
              aria-label="Inputs folder"
              // @ts-expect-error non-standard but supported by every browser
              webkitdirectory=""
              onChange={(e) => setInputs([...(e.target.files ?? [])])}
            />
            <span className="muted">{inputs.length} files</span>
          </label>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn" disabled={running || !ordered.length || !inputs.length} onClick={start}>
            Run
          </button>
        </div>
      </section>
      {st.phase === "running" && <Progress {...st} />}
      {st.phase === "error" && (
        <p role="alert" className="error">
          {st.message}
        </p>
      )}
      {st.phase === "done" && (
        <section className="card" aria-label="Results">
          {st.summary.stage === "pass1" ? (
            <p>
              Done in {fmtDuration(st.seconds)}: pass 1 on {String(st.summary.frames)} frames,{" "}
              {String(st.summary.blocks)} seed blocks, {String(st.summary.tiles)} tiles (
              {Number(st.summary.store_mb).toFixed(0)} MB in browser storage)
              {st.pass2 &&
                `; pass 2 kept ${String(st.pass2.rank_total)} components over ${String(st.pass2.tiles)} tiles`}
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
          <ul aria-label="Output files">
            {st.files.map(([name, url, n]) => (
              <li key={name}>
                <a href={url} download={name}>
                  {name}
                </a>{" "}
                <span className="muted">{(n / 2 ** 20).toFixed(1)} MB</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

function Progress(p: { stage: string; frames: number; units: number; done: number; seconds: number; stateMb: number }) {
  const { fps, eta } = rate(p.done, p.frames, p.seconds);
  const what = p.stage === "pass2" ? "tiles" : "frames";
  return (
    <section className="card" aria-label="Progress">
      <div className="barrow">
        <span>
          {p.done} / {p.frames} {what}
        </span>
        <div className="bar" role="progressbar" aria-valuenow={p.done} aria-valuemax={p.frames}>
          <span style={{ width: `${(100 * p.done) / Math.max(p.frames, 1)}%` }} />
        </div>
        <span>{p.stage === "pass2" ? `${fps.toFixed(1)} tiles/s` : `${fps.toFixed(0)} fps`}</span>
      </div>
      <p className="muted">
        {p.units} units · state {p.stateMb.toFixed(0)} MB · {Number.isFinite(eta) ? fmtDuration(eta) : "…"} left
      </p>
    </section>
  );
}
