import { useEffect, useMemo, useState } from "react";
import { getArray, getJSON, postJSON, type NDArray } from "../../api";
import { Chart } from "../../Chart";
import { Loading, Segmented, Slider, Toggle } from "../../controls";
import { fmtClock } from "../../format";
import { useResource } from "../../poll";
import type { Zooms } from "../../types";
import { useRun } from "../RunContext";
import { clampRange, lowpassState, overviewColumns, traceMode } from "../traceMode";

type Kind = "C" | "YrA";
type Range = [number, number];
type Win = { start: number; arr: NDArray };

// semi-transparent mid blue: visible on both the light and the dark background
const BAND_FILL = "rgba(96, 140, 230, 0.35)";
const BANDS = [{ series: [2, 1] as [number, number], fill: BAND_FILL }];
const SERIES_BAND = [
  { stroke: "--accent", width: 1 },
  { stroke: "--accent", width: 1 },
];
const SERIES_LINE = [{ stroke: "--accent", width: 1.2 }];

function UnitTrace(p: {
  unit: number;
  kind: Kind;
  overview: NDArray | undefined;
  range: Range;
  full: boolean;
  lowpassHz: number | null;
  setRange: (r: Range | null) => void;
  zoomed: boolean;
}) {
  const { summary: s, t } = useRun();
  const [win, setWin] = useState<Win | null>(null);
  const [err, setErr] = useState("");
  const [a, b] = p.range;
  const { id, fps, T } = s;

  useEffect(() => {
    if (!p.full) return;
    let cancelled = false;
    const q = `start=${a}&stop=${b}&kind=${p.kind}${p.lowpassHz ? `&lowpass_hz=${p.lowpassHz}` : ""}`;
    getArray(`/runs/${id}/traces/${p.unit}?${q}`).then(
      (arr) => {
        if (cancelled) return;
        setWin({ start: a, arr });
        setErr("");
      },
      (e: Error) => {
        if (!cancelled) setErr(e.message);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [p.full, a, b, p.kind, p.lowpassHz, p.unit, id]);

  const scale = fps ? 1 / fps : 1;
  const data = useMemo(() => {
    if (p.full && win) {
      const n = win.arr.shape[0];
      const xs = Float64Array.from({ length: n }, (_, i) => (win.start + i) * scale);
      return [xs, Float64Array.from(win.arr.data as ArrayLike<number>)] as [Float64Array, Float64Array];
    }
    if (p.overview) {
      const { lo, hi } = overviewColumns(p.overview, p.unit);
      const bins = lo.length;
      const xs = Float64Array.from({ length: bins }, (_, i) => ((i + 0.5) * T * scale) / bins);
      return [xs, lo, hi] as [Float64Array, Float64Array, Float64Array];
    }
    return null;
  }, [p.full, win, p.overview, p.unit, T, scale]);

  if (err) {
    return (
      <p className="error" role="alert">
        Unit {p.unit}: {err}
      </p>
    );
  }
  if (!data) return <p className="muted">Loading unit {p.unit}…</p>;
  const band = data.length === 3;
  return (
    <div>
      <span className="label mono">Unit {p.unit}</span>
      <Chart
        data={data}
        series={band ? SERIES_BAND : SERIES_LINE}
        bands={band ? BANDS : undefined}
        height={120}
        xLabel={fps ? "seconds" : "frame"}
        cursorX={t * scale}
        xRange={p.zoomed ? [a * scale, b * scale] : null}
        onSelectX={(x0, x1) => p.setRange(clampRange(x0 / scale, x1 / scale, T))}
        onResetX={() => p.setRange(null)}
        sync="traces"
      />
    </div>
  );
}

function FpsField() {
  const { summary: s } = useRun();
  const [v, setV] = useState("20");
  const [err, setErr] = useState("");
  return (
    <form
      className="row"
      onSubmit={(e) => {
        e.preventDefault();
        postJSON(`/runs/${s.id}/fps`, { fps: Number(v) }).then(
          () => window.location.reload(),
          (x: Error) => setErr(x.message),
        );
      }}
    >
      <label className="field">
        Frame rate of this recording (Hz)
        <input type="number" min={1} max={1000} step="any" value={v} onChange={(e) => setV(e.target.value)} />
      </label>
      <button className="btn" type="submit">
        Save
      </button>
      {err && <span className="error">{err}</span>}
    </form>
  );
}

export function Traces() {
  const { summary: s, sel, setSel, setT } = useRun();
  const [kind, setKind] = useState<Kind>("C");
  const [lpOn, setLpOn] = useState(false);
  const [hz, setHz] = useState(5);
  const [range, setRange] = useState<Range | null>(null);
  const zooms = useResource(`zooms:${s.id}`, () => getJSON<Zooms>(`/runs/${s.id}/zooms`));
  const overview = useResource(`ov:${s.id}:${kind}`, () =>
    getArray(`/runs/${s.id}/traces/overview?kind=${kind}&bins=2000`),
  );
  const [extra, setExtra] = useState<number[]>([]);
  const lp = lowpassState(s.fps, lpOn, hz);
  const full = traceMode(range, s.T) === "window";
  const r: Range = range ?? [0, s.T];
  const shown = useMemo(() => {
    const base = sel !== null ? [sel] : (zooms.data?.units.slice(0, 3) ?? []);
    return [...new Set([...base, ...extra])].slice(0, 8);
  }, [sel, zooms.data, extra]);

  return (
    <div className="bento">
      <section className="card span4" aria-label="Trace controls">
        <div className="row">
          <Segmented
            label="Trace"
            options={[
              { value: "C", label: "C (denoised)" },
              { value: "YrA", label: "YrA (raw)" },
            ]}
            value={kind}
            onChange={setKind}
          />
          <Toggle
            label="Low-pass"
            checked={lpOn && lp.enabled}
            disabled={!lp.enabled}
            title={lp.reason}
            onChange={setLpOn}
          />
          <div style={{ minWidth: 240 }}>
            <Slider
              label="Cutoff"
              min={0.5}
              max={lp.max}
              step={0.5}
              value={Math.min(hz, lp.max)}
              disabled={!lp.enabled || !lpOn}
              onChange={setHz}
              format={(v) => `${v.toFixed(1)} Hz`}
            />
          </div>
          <span className="spacer" />
          <button className="btn" type="button" onClick={() => setRange(null)} disabled={!range}>
            Whole recording
          </button>
        </div>
        {!s.fps && <FpsField />}
        <span className="muted">
          {full
            ? `Full resolution${lp.hz ? `, low-pass ${lp.hz.toFixed(1)} Hz` : ""}.`
            : "Overview (min/max per bin); drag across a chart to zoom in to full resolution."}{" "}
          {fmtClock(r[0], s.fps)} – {fmtClock(r[1], s.fps)}
        </span>
        <Loading res={zooms} what="zoom suggestions">
          {(z) => (
            <div className="row">
              <span className="label">Jump to</span>
              {z.windows.map((w) => (
                <button
                  key={w.start}
                  className="chip"
                  type="button"
                  onClick={() => (setRange([w.start, w.stop]), setT(w.start))}
                >
                  {fmtClock(w.start, s.fps)}
                </button>
              ))}
              <span className="label">Add unit</span>
              {z.units.map((u) => (
                <button
                  key={u}
                  className="chip"
                  type="button"
                  aria-pressed={shown.includes(u)}
                  onClick={() =>
                    sel === null
                      ? setSel(u)
                      : setExtra((x) => (x.includes(u) ? x.filter((y) => y !== u) : [...x, u]))
                  }
                >
                  {u}
                </button>
              ))}
            </div>
          )}
        </Loading>
      </section>
      <section className="card span4" aria-label="Traces">
        {overview.error && (
          <p className="error" role="alert">
            {overview.error}
          </p>
        )}
        {overview.computing && !full && (
          <p className="muted">Computing the trace overview… {overview.elapsed.toFixed(0)} s</p>
        )}
        {shown.length === 0 && <p className="muted">Select a unit in the image above.</p>}
        {shown.map((u) => (
          <UnitTrace
            key={u}
            unit={u}
            kind={kind}
            overview={overview.data}
            range={r}
            full={full}
            lowpassHz={lp.hz}
            setRange={setRange}
            zoomed={range !== null}
          />
        ))}
      </section>
    </div>
  );
}
