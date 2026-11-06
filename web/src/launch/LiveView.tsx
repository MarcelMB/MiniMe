import { memo, useMemo, useState } from "react";
import type uPlot from "uplot";
import { postJSON } from "../api";
import { Chart } from "../Chart";
import { Toggle } from "../controls";
import type { LivePanel } from "../stream/protocol";
import { type Layer, LiveCanvas, linkSegments, windowColor } from "./LiveCanvas";
import { type LiveState, shiftSeries, shownPanel } from "./liveState";

const NAMES: Record<LivePanel, string> = {
  motion: "Motion",
  pass2: "Compression",
  seeds: "Seeds",
  units: "Units",
  stitch: "Stitch",
};
const series = (labels: string[]): uPlot.Series[] =>
  labels.map((label, i) => ({ label, stroke: windowColor(i), width: 1 }));
const xs = (n: number) => Float64Array.from({ length: n }, (_, i) => i);
const NO_LAYERS: Layer[] = [];
const SHIFT_SERIES = series(["shift y", "shift x"]);
/** A chart's data: x 0..n-1 under the traces. */
const traceData = (traces: Float32Array[]): uPlot.AlignedData => [xs(traces[0]?.length ?? 0), ...traces];

/** The live view card. Memoized: the job page's 1 s clock re-renders it only when its props change.
 * `follow` is the running stage's panel: shown (once it has a preview) unless the user picked one. */
export const LiveView = memo(function LiveView({
  jobId,
  on,
  state,
  follow,
}: {
  jobId: string;
  on: boolean;
  state: LiveState;
  follow: LivePanel | null;
}) {
  const [picked, setPicked] = useState<LivePanel | null>(null);
  const [err, setErr] = useState("");
  const shown = shownPanel(state.order, picked, follow);
  const toggle = (v: boolean) =>
    postJSON(`/jobs/${jobId}/live`, { on: v }).then(
      () => setErr(""),
      (e: Error) => setErr(e.message),
    );
  return (
    <section className="card span4" aria-label="Live view">
      <div className="row">
        <span className="label">Live view</span>
        <span className="spacer" />
        <Toggle label="Show each step on the data (slightly slower)" checked={on} onChange={toggle} />
      </div>
      {err && <span className="muted">{err}</span>}
      {on && state.order.length === 0 && <span className="muted">Waiting for the next step…</span>}
      {on && state.order.length > 0 && (
        <>
          <div className="row" role="tablist">
            {state.order.map((p) => (
              <button
                key={p}
                type="button"
                role="tab"
                aria-selected={p === shown}
                className={p === shown ? "btn primary" : "btn"}
                onClick={() => setPicked(p === picked ? null : p)}
              >
                {NAMES[p]}
              </button>
            ))}
          </div>
          {shown && <Panel panel={shown} s={state} />}
        </>
      )}
    </section>
  );
});

function MotionPanel({ s }: { s: LiveState }) {
  const m = s.motion;
  const data = useMemo<uPlot.AlignedData>(() => shiftSeries(s.shifts), [s.shifts]);
  if (!m) return null;
  return (
    <div className="liverow">
      <LiveCanvas image={m.frame} h={m.h} w={m.w} f={m.f} layers={NO_LAYERS} label="corrected frame" />
      <div className="livechart">
        <Chart data={data} series={SHIFT_SERIES} height={180} xLabel="frame" />
      </div>
    </div>
  );
}

function Panel({ panel, s }: { panel: LivePanel; s: LiveState }) {
  if (panel === "motion") return <MotionPanel s={s} />;
  if (panel === "pass2" && s.pass2) return <Pass2Panel p={s.pass2} />;
  if (panel === "seeds" && s.seeds) return <SeedsPanel p={s.seeds} />;
  if (panel === "units" && s.units) return <UnitsPanel p={s.units} />;
  if (panel === "stitch") return <StitchPanel s={s} />;
  return null;
}

function Pass2Panel({ p }: { p: NonNullable<LiveState["pass2"]> }) {
  const done = useMemo(() => p.ranks.reduce((n, r) => n + (r >= 0 ? 1 : 0), 0), [p.ranks]);
  const data = useMemo(() => traceData(p.traces), [p.traces]);
  const ser = useMemo(() => series(p.traces.map((_, j) => `comp ${j + 1}`)), [p.traces]);
  return (
    <div className="liverow">
      <RankGrid p={p} />
      <div>
        <span className="muted">
          {p.label} · tile {p.k + 1} · rank {p.ranks[p.k]} · {done}/{p.ranks.length} tiles seen
        </span>
        <div className="row">
          {p.comps.map((c, j) => (
            <LiveCanvas key={j} image={c} h={p.th} w={p.tw} f={1} layers={NO_LAYERS} label={`component ${j + 1}`} />
          ))}
        </div>
        {p.traces.length > 0 && <Chart data={data} series={ser} height={140} />}
      </div>
    </div>
  );
}

function SeedsPanel({ p }: { p: NonNullable<LiveState["seeds"]> }) {
  const layers = useMemo<Layer[]>(
    () => [
      { kind: "points", points: p.seeds, color: "#9ca3af", r: 1.5 },
      { kind: "points", points: p.merged, color: "#fb923c", r: 2.5 },
    ],
    [p.seeds, p.merged],
  );
  return (
    <div className="liverow">
      <LiveCanvas image={p.image} h={p.h} w={p.w} f={p.f} label="seeds on the max projection" layers={layers} />
      <span className="muted">
        {p.label} · {p.seeds.length / 2} seeds, {p.merged.length / 2} after the merge
      </span>
    </div>
  );
}

function UnitsPanel({ p }: { p: NonNullable<LiveState["units"]> }) {
  const layers = useMemo<Layer[]>(() => [{ kind: "outlines", polygons: p.polygons, color: "#4f9cf9" }], [p.polygons]);
  const data = useMemo(() => traceData(p.traces), [p.traces]);
  const ser = useMemo(() => series(p.ids.map((u) => `unit ${u}`)), [p.ids]);
  return (
    <div className="liverow">
      <LiveCanvas image={p.image} h={p.h} w={p.w} f={p.f} label="footprints on the mean image" layers={layers} />
      <div>
        <span className="muted">
          {p.label} · after {p.step === "init" ? "init" : "CNMF"} · {p.count} units
        </span>
        {p.traces.length > 0 && <Chart data={data} series={ser} height={180} />}
      </div>
    </div>
  );
}

function StitchPanel({ s }: { s: LiveState }) {
  const st = s.stitch;
  const ref = s.units ?? s.seeds;
  // rebuilt only when the stitch state changes (a new window or the links), not on every render
  const layers = useMemo<Layer[]>(() => {
    const out: Layer[] = [...st.windows].map(([i, w]) => ({
      kind: "outlines" as const,
      polygons: w.polygons,
      color: windowColor(i),
    }));
    if (st.edges) out.push({ kind: "links", segments: linkSegments(st.windows, st.edges), color: "#ffffff" });
    return out;
  }, [st]);
  if (!ref) return null;
  return (
    <div className="liverow">
      <LiveCanvas image={null} h={ref.h} w={ref.w} f={ref.f} layers={layers} label="units by window and their links" />
      <span className="muted">
        {st.windows.size} windows
        {st.edges && st.units !== undefined
          ? ` · ${st.edges.length / 2} links → ${st.units} units`
          : " · links when the stitch runs"}
      </span>
    </div>
  );
}

function RankGrid({ p }: { p: NonNullable<LiveState["pass2"]> }) {
  const { h, w, tile } = p.grid;
  const [rows, cols] = [Math.ceil(h / tile), Math.ceil(w / tile)];
  const max = Math.max(1, ...p.ranks);
  return (
    <div
      className="rankgrid"
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, aspectRatio: `${w} / ${h}` }}
      role="img"
      aria-label="tile ranks"
    >
      {Array.from({ length: rows * cols }, (_, k) => (
        <span
          key={k}
          title={p.ranks[k] >= 0 ? `tile ${k + 1}: rank ${p.ranks[k]}` : `tile ${k + 1}`}
          style={{
            opacity: p.ranks[k] >= 0 ? 0.25 + (0.75 * p.ranks[k]) / max : 0.08,
            outline: k === p.k ? "2px solid var(--accent)" : undefined,
          }}
        />
      ))}
    </div>
  );
}
