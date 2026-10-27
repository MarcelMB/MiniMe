import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { cssVar, fullExtent, resolveSeries } from "./chartUtil";

export { cssVar };

export type ChartProps = {
  data: uPlot.AlignedData;
  series: uPlot.Series[];
  height?: number;
  bands?: uPlot.Band[];
  cursorX?: number | null;
  xRange?: [number, number] | null;
  onSelectX?: (x0: number, x1: number) => void;
  onResetX?: () => void;
  sync?: string;
  xLabel?: string;
};

export function Chart(p: ChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const props = useRef(p);
  props.current = p;
  const [theme, setTheme] = useState(0);
  const pinned = useRef(false);
  const lo = p.xRange?.[0];
  const hi = p.xRange?.[1];

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setTheme((n) => n + 1);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  useEffect(() => {
    const el = host.current!;
    const axis = { stroke: cssVar("--muted"), grid: { stroke: cssVar("--track") }, ticks: { stroke: cssVar("--track") } };
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height: p.height ?? 180,
      scales: { x: { time: false } },
      series: [{ label: p.xLabel ?? "x" }, ...resolveSeries(p.series)],
      bands: p.bands,
      legend: { show: false },
      cursor: {
        sync: p.sync ? { key: p.sync } : undefined,
        drag: { x: true, y: false, setScale: !p.onSelectX },
      },
      axes: [{ ...axis, label: p.xLabel }, axis],
      hooks: {
        draw: [
          (u) => {
            const x = props.current.cursorX;
            if (x == null) return;
            const { min, max } = u.scales.x;
            if (min != null && max != null && (x < min || x > max)) return;
            const cx = u.valToPos(x, "x", true);
            const ctx = u.ctx;
            ctx.save();
            ctx.strokeStyle = cssVar("--hot");
            ctx.lineWidth = 2 * devicePixelRatio;
            ctx.beginPath();
            ctx.moveTo(cx, u.bbox.top);
            ctx.lineTo(cx, u.bbox.top + u.bbox.height);
            ctx.stroke();
            ctx.restore();
          },
        ],
        setSelect: [
          (u) => {
            const cb = props.current.onSelectX;
            if (!cb || u.select.width < 3) return;
            const x0 = u.posToVal(u.select.left, "x");
            const x1 = u.posToVal(u.select.left + u.select.width, "x");
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
            cb(x0, x1);
          },
        ],
      },
    };
    const u = new uPlot(opts, p.data, el);
    plot.current = u;
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height: props.current.height ?? 180 }));
    ro.observe(el);
    el.ondblclick = () => props.current.onResetX?.();
    return () => {
      ro.disconnect();
      u.destroy();
      plot.current = null;
    };
    // the chart is rebuilt only when its layout changes; data and range update in place
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.series.length, p.height, p.sync, p.bands?.length, theme]);

  useEffect(() => {
    plot.current?.setData(p.data);
  }, [p.data]);
  useEffect(() => {
    const u = plot.current;
    if (!u) return;
    if (lo != null && hi != null) {
      pinned.current = true;
      u.setScale("x", { min: lo, max: hi });
    } else if (pinned.current) {
      pinned.current = false;
      const e = fullExtent(props.current.data);
      if (e) u.setScale("x", { min: e[0], max: e[1] });
    }
  }, [lo, hi, p.data, theme]);
  useEffect(() => {
    plot.current?.redraw(false);
  }, [p.cursorX]);

  return <div ref={host} className="chart" />;
}
