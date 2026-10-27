import { useMemo } from "react";
import { getArray, type NDArray } from "../../api";
import { Chart } from "../../Chart";
import { Loading } from "../../controls";
import { useResource } from "../../poll";
import { useRun } from "../RunContext";

function columns(data: ArrayLike<number>, n: number, k: number): Float64Array[] {
  const out = Array.from({ length: k }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[j][i] = data[i * k + j];
  return out;
}

function MotionCharts({ m, xs, cursor, xLabel }: { m: NDArray; xs: Float64Array; cursor: number; xLabel: string }) {
  const [dy, dx, q] = useMemo(() => columns(m.data, m.shape[0], 3), [m]);
  const shifts = useMemo(() => [xs, dy, dx] as [Float64Array, Float64Array, Float64Array], [xs, dy, dx]);
  const quality = useMemo(() => [xs, q] as [Float64Array, Float64Array], [xs, q]);
  return (
    <>
      <Chart data={shifts} series={[{ stroke: "--accent", width: 1 }, { stroke: "--hot", width: 1 }]} cursorX={cursor} xLabel={xLabel} sync="motion" />
      <span className="label">Motion quality (correlation with the template)</span>
      <Chart data={quality} series={[{ stroke: "--ok", width: 1 }]} height={140} cursorX={cursor} xLabel={xLabel} sync="motion" />
    </>
  );
}

function ResidualCharts({ r }: { r: NDArray }) {
  const [pos, conf] = useMemo(() => {
    const n = r.shape[0];
    const [ry, rx, rc] = columns(r.data, n, 3);
    const bx = Float64Array.from({ length: n }, (_, i) => i);
    return [
      [bx, ry, rx] as [Float64Array, Float64Array, Float64Array],
      [bx, rc] as [Float64Array, Float64Array],
    ] as const;
  }, [r]);
  return (
    <>
      <Chart data={pos} series={[{ stroke: "--accent", width: 1 }, { stroke: "--hot", width: 1 }]} xLabel="block" height={160} />
      <Chart data={conf} series={[{ stroke: "--ok", width: 1 }]} xLabel="block" height={120} />
    </>
  );
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
          {(m) => <MotionCharts m={m} xs={xs} cursor={cursor} xLabel={xLabel} />}
        </Loading>
      </section>
      <section className="card span4" aria-label="Residual shift per block">
        <span className="label">Leftover shift per block after correction (should stay near 0 px)</span>
        <Loading res={resid} what="the leftover shift per block">
          {(r) => <ResidualCharts r={r} />}
        </Loading>
      </section>
    </div>
  );
}
