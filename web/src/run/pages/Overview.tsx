import { fmtBytes, fmtDuration, fmtInt, stageLabel } from "../../format";
import type { Stage } from "../../types";
import { useRun } from "../RunContext";

export function funnel(c: Record<string, number>): { name: string; value: number }[] {
  return [
    { name: "Seeds", value: c.seeds_init ?? 0 },
    { name: "After PNR", value: c.seeds_pnr ?? 0 },
    { name: "Merged", value: c.seeds_merged ?? 0 },
    { name: "Units", value: c.units_final ?? 0 },
  ];
}

export function stageBars(stages: Stage[]): { name: string; seconds: number; pct: number }[] {
  const max = Math.max(1e-9, ...stages.map((s) => s.seconds));
  return stages.map((s) => ({ name: s.stage, seconds: s.seconds, pct: (s.seconds / max) * 100 }));
}

export function Overview() {
  const { summary: s } = useRun();
  const f = funnel(s.counts);
  const top = Math.max(1, f[0].value);
  const changed = Object.entries(s.params_changed);
  return (
    <div className="bento">
      <section className="card">
        <span className="label">Units</span>
        <span className="big" style={{ color: "var(--accent)" }}>{fmtInt(s.counts.units_final ?? 0)}</span>
        <span className="muted">after merging {fmtInt(s.counts.seeds_merged ?? 0)} candidates</span>
      </section>
      <section className="card">
        <span className="label">Run time</span>
        <span className="big">{s.wall_total ? fmtDuration(s.wall_total) : "–"}</span>
        <span className="muted">{fmtInt(s.T)} frames{s.workers ? ` · ${s.workers} workers` : ""}</span>
      </section>
      <section className="card">
        <span className="label">Peak memory</span>
        <span className="big">{s.peak_rss ? fmtBytes(s.peak_rss) : "–"}</span>
        <span className="muted">
          {s.peak_rss_children ? `${fmtBytes(s.peak_rss_children)} per worker` : ""}
          {s.bytes.raw ? ` · raw ${fmtBytes(s.bytes.raw)}` : ""}
        </span>
      </section>
      <section className="card">
        <span className="label">Motion</span>
        <span className="big">{s.motion.max_abs_shift_px?.toFixed(1) ?? "–"}<span className="muted" style={{ fontSize: 18 }}> px</span></span>
        <span className="muted">largest shift · median quality {s.motion.quality_median?.toFixed(2) ?? "–"}</span>
      </section>
      <section className="card span2" aria-label="Seed funnel">
        <span className="label">From seeds to units</span>
        {f.map((x) => (
          <div className="barrow" key={x.name}>
            <span>{x.name}</span>
            <div className="bar"><span style={{ width: `${Math.max(1.5, (x.value / top) * 100)}%` }} /></div>
            <span className="mono">{fmtInt(x.value)}</span>
          </div>
        ))}
      </section>
      <section className="card span2" aria-label="Stage timings">
        <span className="label">Where the time goes</span>
        {stageBars(s.stages).filter((b) => b.seconds >= 0.5).map((b) => (
          <div className="barrow" key={b.name}>
            <span>{stageLabel(b.name)}</span>
            <div className="bar"><span style={{ width: `${b.pct}%`, background: "var(--text)" }} /></div>
            <span className="mono">{fmtDuration(b.seconds)}</span>
          </div>
        ))}
      </section>
      <section className="card span4" aria-label="Changed parameters">
        <span className="label">Parameters changed from the defaults</span>
        {changed.length === 0 ? (
          <span className="muted">None: this run used the defaults.</span>
        ) : (
          <table className="data">
            <thead><tr><th>Parameter</th><th>This run</th><th>Default</th></tr></thead>
            <tbody>
              {changed.map(([k, v]) => (
                <tr key={k}><td className="mono">{k}</td><td className="mono">{JSON.stringify(v.value)}</td><td className="mono muted">{JSON.stringify(v.default)}</td></tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
