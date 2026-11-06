import type { LivePanel, LivePayload } from "../stream/protocol";

export type StitchWindow = { polygons: number[][][][]; centroids: Float64Array };
export type LiveState = {
  /** Panels in the order they first arrived. */
  order: LivePanel[];
  motion?: Extract<LivePayload, { panel: "motion" }>;
  /** Shift chunks by their first absolute frame (`at`), x/y interleaved as the workers send them. */
  shifts: Map<number, Float32Array>;
  pass2?: Extract<LivePayload, { panel: "pass2" }>;
  seeds?: Extract<LivePayload, { panel: "seeds" }>;
  units?: Extract<LivePayload, { panel: "units" }>;
  stitch: { windows: Map<number, StitchWindow>; edges?: Float64Array; units?: number };
};

export const emptyLive = (): LiveState => ({ order: [], shifts: new Map(), stitch: { windows: new Map() } });

export function reduceLive(s: LiveState, p: LivePayload): LiveState {
  const order = s.order.includes(p.panel) ? s.order : [...s.order, p.panel];
  switch (p.panel) {
    case "motion":
      return { ...s, order, motion: p, shifts: new Map(s.shifts).set(p.at, p.shifts) };
    case "pass2":
      return { ...s, order, pass2: p };
    case "seeds":
      return { ...s, order, seeds: p };
    case "units":
      return { ...s, order, units: p };
    case "stitch":
      if ("edges" in p) return { ...s, order, stitch: { ...s.stitch, edges: p.edges, units: p.units } };
      return {
        ...s,
        order,
        stitch: {
          ...s.stitch,
          windows: new Map(s.stitch.windows).set(p.window, { polygons: p.polygons, centroids: p.centroids }),
        },
      };
  }
}

/** The shift chunks in frame order as (frame, shift 0, shift 1), at most `max` points (every k-th). */
export function shiftSeries(shifts: Map<number, Float32Array>, max = 2000): [Float64Array, Float32Array, Float32Array] {
  const chunks = [...shifts].sort((a, b) => a[0] - b[0]);
  const n = chunks.reduce((m, [, v]) => m + v.length / 2, 0);
  const step = Math.max(1, Math.ceil(n / max));
  const len = Math.ceil(n / step);
  const [x, a, b] = [new Float64Array(len), new Float32Array(len), new Float32Array(len)];
  let i = 0;
  let j = 0;
  for (const [at, v] of chunks)
    for (let f = 0; f < v.length / 2; f++, i++) {
      if (i % step) continue;
      [x[j], a[j], b[j]] = [at + f, v[2 * f], v[2 * f + 1]];
      j++;
    }
  return [x.subarray(0, j), a.subarray(0, j), b.subarray(0, j)];
}
