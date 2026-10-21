import { useMemo } from "react";
import { getArray } from "../../api";
import { Chart, cssVar } from "../../Chart";
import { Loading } from "../../controls";
import { useResource } from "../../poll";
import { useRun } from "../RunContext";

function columns(data: ArrayLike<number>, n: number, k: number): Float64Array[] {
  const out = Array.from({ length: k }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[j][i] = data[i * k + j];
  return out;
}

export function Motion() {
  const { summary: s, t } = useRun();
  const motion = useResource(`motion:${s.id}`, () => getArray(`/runs/${s.id}/motion`));
  const resid = useResource(`resid:${s.id}`, () => getArray(`/runs/${s.id}/motion/residual`));
  const fps = s.fps;
  const xs = useMemo(() => Float64Array.from({ length: s.T }, (_, i) => (fps ? i / fps / 60 : i)), [s.T, fps]);
  const xLabel = fps ? "minutes" : "frame";
  const cursor = fps ? t / fps / 60 : t;
  return (
    <div className="bento">
      <section className="card span4" aria-label="Motion shifts">
        <span className="label">Shifts (px) · y in blue, x in orange</span>
        <Loading res={motion} what="motion">
          {(m) => {
            const [dy, dx, q] = columns(m.data, m.shape[0], 3);
            return (
              <>
                <Chart
                  data={[xs, dy, dx]}
                  series={[{ stroke: cssVar("--accent"), width: 1 }, { stroke: cssVar("--hot"), width: 1 }]}
                  cursorX={cursor}
                  xLabel={xLabel}
                  sync="motion"
                />
                <span className="label">Motion quality (correlation with the template)</span>
                <Chart data={[xs, q]} series={[{ stroke: cssVar("--ok"), width: 1 }]} height={140} cursorX={cursor} xLabel={xLabel} sync="motion" />
              </>
            );
          }}
        </Loading>
      </section>
      <section className="card span4" aria-label="Residual shift per block">
        <span className="label">Leftover shift per block after correction (should stay near 0 px)</span>
        <Loading res={resid} what="the leftover shift per block">
          {(r) => {
            const n = r.shape[0];
            const [ry, rx, rc] = columns(r.data, n, 3);
            const bx = Float64Array.from({ length: n }, (_, i) => i);
            return (
              <>
                <Chart data={[bx, ry, rx]} series={[{ stroke: cssVar("--accent"), width: 1 }, { stroke: cssVar("--hot"), width: 1 }]} xLabel="block" height={160} />
                <Chart data={[bx, rc]} series={[{ stroke: cssVar("--ok"), width: 1 }]} xLabel="block" height={120} />
              </>
            );
          }}
        </Loading>
      </section>
    </div>
  );
}
