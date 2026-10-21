import { useMemo } from "react";
import { getArray } from "../../api";
import { Loading } from "../../controls";
import { imgFrom } from "../../draw";
import { fmtInt } from "../../format";
import { ImageView } from "../../ImageView";
import { useResource } from "../../poll";
import { useRun } from "../RunContext";

export function rankHistogram(ranks: number[], bins = 10): { lo: number; hi: number; n: number }[] {
  if (ranks.length === 0) return [];
  const lo = Math.min(...ranks);
  const hi = Math.max(...ranks);
  const w = Math.max(1e-9, (hi - lo) / bins);
  const out = Array.from({ length: bins }, (_, i) => ({ lo: lo + i * w, hi: lo + (i + 1) * w, n: 0 }));
  for (const r of ranks) out[Math.min(bins - 1, Math.floor((r - lo) / w))].n++;
  return out;
}

function HeatImage({ a, title }: { a: Parameters<typeof imgFrom>[0]; title: string }) {
  const img = useMemo(() => imgFrom(a, "heat", 1, 99), [a]);
  return <ImageView image={img} label={title} />;
}

function MapTile({ name, title, note }: { name: string; title: string; note: string }) {
  const { summary: s } = useRun();
  const res = useResource(`img:${s.id}:${name}`, () => getArray(`/runs/${s.id}/image/${name}`));
  return (
    <section className="card span2" aria-label={title}>
      <span className="label">{title}</span>
      <Loading res={res} what={title.toLowerCase()}>
        {(a) => <HeatImage a={a} title={title} />}
      </Loading>
      <span className="muted">{note}</span>
    </section>
  );
}

export function Compression() {
  const { summary: s } = useRun();
  const hist = useMemo(() => rankHistogram(s.ranks), [s.ranks]);
  const top = Math.max(1, ...hist.map((b) => b.n));
  const sum = s.ranks.reduce((a, b) => a + b, 0);
  return (
    <div className="bento">
      <MapTile name="rank" title="Rank per tile" note={`${s.ranks.length} tiles of ${s.tile} px`} />
      <MapTile name="noise" title="Noise per pixel" note="standard deviation of the residual noise" />
      <section className="card">
        <span className="label">Total rank</span>
        <span className="big">{fmtInt(sum)}</span>
        <span className="muted">largest tile {s.ranks.length ? Math.max(...s.ranks) : "–"}</span>
      </section>
      <section className="card span3" aria-label="Rank histogram">
        <span className="label">Tiles per rank</span>
        {hist.map((b) => (
          <div className="barrow" key={b.lo}>
            <span className="mono">
              {Math.round(b.lo)}–{Math.round(b.hi)}
            </span>
            <div className="bar">
              <span style={{ width: `${(b.n / top) * 100}%` }} />
            </div>
            <span className="mono">{b.n}</span>
          </div>
        ))}
      </section>
      <MapTile name="background" title="Background" note="spatial background b of the final CNMF" />
    </div>
  );
}
