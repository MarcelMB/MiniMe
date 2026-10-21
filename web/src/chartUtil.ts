import type uPlot from "uplot";

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#2563eb";
}

/** Resolve series strokes given as CSS variable names ("--accent") to colours. */
export function resolveSeries(series: uPlot.Series[], get: (name: string) => string = cssVar): uPlot.Series[] {
  return series.map((s) => (typeof s.stroke === "string" && s.stroke.startsWith("--") ? { ...s, stroke: get(s.stroke) } : s));
}

/** The full x extent of the data, or null when there is none. */
export function fullExtent(data: uPlot.AlignedData): [number, number] | null {
  const x = data[0];
  return x && x.length ? [x[0] as number, x[x.length - 1] as number] : null;
}
