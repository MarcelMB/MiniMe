import type { NDArray } from "../api";

export const FULL_RES_MAX = 20000;

export function traceMode(range: [number, number] | null, T: number): "overview" | "window" {
  const [a, b] = range ?? [0, T];
  return b - a <= FULL_RES_MAX ? "window" : "overview";
}

export function clampRange(x0: number, x1: number, T: number): [number, number] {
  const a = Math.max(0, Math.min(T - 1, Math.floor(Math.min(x0, x1))));
  const b = Math.min(T, Math.max(a + 1, Math.ceil(Math.max(x0, x1))));
  return [a, b];
}

export function lowpassState(
  fps: number | null,
  on: boolean,
  hz: number,
): { enabled: boolean; hz: number | null; max: number; reason?: string } {
  if (!fps) return { enabled: false, hz: null, max: 10, reason: "set the frame rate of the recording first" };
  const max = fps / 2;
  return { enabled: true, hz: on ? Math.min(hz, max) : null, max };
}

export function overviewColumns(ov: NDArray, unit: number): { lo: Float64Array; hi: Float64Array } {
  const bins = ov.shape[2];
  const base = unit * 2 * bins;
  return {
    lo: Float64Array.from(ov.data.subarray(base, base + bins) as ArrayLike<number>),
    hi: Float64Array.from(ov.data.subarray(base + bins, base + 2 * bins) as ArrayLike<number>),
  };
}
