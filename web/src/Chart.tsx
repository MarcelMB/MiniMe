import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#2563eb";
}

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

  useEffect(() => {
    const el = host.current!;
    const axis = { stroke: cssVar("--muted"), grid: { stroke: cssVar("--track") }, ticks: { stroke: cssVar("--track") } };
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height: p.height ?? 180,
      series: [{ label: p.xLabel ?? "x" }, ...p.series],
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
  }, [p.series.length, p.height, p.sync, p.bands?.length]);

  useEffect(() => {
    plot.current?.setData(p.data);
  }, [p.data]);
  useEffect(() => {
    if (plot.current && p.xRange) plot.current.setScale("x", { min: p.xRange[0], max: p.xRange[1] });
  }, [p.xRange, p.data]);
  useEffect(() => {
    plot.current?.redraw(false);
  }, [p.cursorX]);

  return <div ref={host} className="chart" />;
}
