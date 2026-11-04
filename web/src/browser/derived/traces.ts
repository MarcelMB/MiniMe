import type { Zooms } from "../../types";
import { irfft, rfft } from "../fft";
import { pyFloat, pyRound } from "../num";
import { HttpError } from "../router";

const MIN_LEN = 8; // shorter traces are returned unchanged (as CaLab's MIN_SPECTRUM_LEN)

/** rfft gain of CaLab's low-pass: 1 below f_lp/2, cosine taper to 0 at 1.5 f_lp (`traces.lowpass_gain`). */
export function lowpassGain(n: number, fs: number, fLp: number): Float64Array {
  const half = (n >> 1) + 1;
  const g = new Float64Array(half).fill(1);
  const f0 = Math.min(fLp, fs / 2);
  const w = 0.5 * f0;
  for (let k = 0; k < half; k++) {
    const f = (k * fs) / n; // numpy rfftfreq(n, 1/fs)
    if (f >= f0 + w) g[k] = 0;
    else if (f >= f0 - w) g[k] = 0.5 * (1 + Math.cos((Math.PI * (f - (f0 - w))) / (2 * w)));
  }
  return g;
}

export function lowpass(x: Float32Array, fs: number, fLp: number): Float32Array {
  if (x.length < MIN_LEN) return x.slice();
  if (fLp <= 0 || fs <= 0)
    throw new HttpError(400, `cutoff and frame rate must be > 0 (got ${pyFloat(fLp)} Hz, ${pyFloat(fs)} Hz)`);
  const { re, im } = rfft(Float64Array.from(x));
  const g = lowpassGain(x.length, fs, fLp);
  for (let k = 0; k < g.length; k++) {
    re[k] *= g[k];
    im[k] *= g[k];
  }
  return Float32Array.from(irfft(re, im, x.length));
}

/** `trace[start:stop]`, low-passed with `marginS` of extra trace on each side. */
export function traceWindow(
  trace: Float32Array,
  start: number,
  stop: number,
  fs: number | null,
  lowpassHz: number | null,
  marginS = 5,
): Float32Array {
  const T = trace.length;
  if (!(0 <= start && start < stop && stop <= T)) throw new HttpError(400, `window ${start}:${stop} outside 0:${T}`);
  if (lowpassHz === null) return trace.slice(start, stop);
  if (fs === null) throw new HttpError(400, "the low-pass needs the frame rate of the recording");
  const m = pyRound(marginS * fs); // Python's round()
  const [a, b] = [Math.max(0, start - m), Math.min(T, stop + m)];
  return lowpass(trace.slice(a, b), fs, lowpassHz).slice(start - a, stop - a);
}

/** (n, 2, bins) min / max of each trace over `bins` equal spans (numpy's linspace starts). */
export async function overview(
  rows: (i: number) => Promise<Float32Array>,
  n: number,
  T: number,
  bins: number,
): Promise<Float32Array> {
  const b = Math.max(1, Math.min(Math.trunc(bins), T));
  const step = T / b;
  const starts = Array.from({ length: b + 1 }, (_, i) => (i === b ? T : Math.trunc(i * step)));
  const out = new Float32Array(n * 2 * b);
  for (let i = 0; i < n; i++) {
    const row = await rows(i);
    for (let j = 0; j < b; j++) {
      let [lo, hi] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
      const end = Math.max(starts[j + 1], starts[j] + 1); // reduceat: an empty span gives row[start]
      for (let t = starts[j]; t < end; t++) {
        lo = Math.min(lo, row[t]);
        hi = Math.max(hi, row[t]);
      }
      out[i * 2 * b + j] = lo;
      out[i * 2 * b + b + j] = hi;
    }
  }
  return out;
}

/** The `nUnits` brightest units and `wndMin`-minute windows, `perHour` per hour. */
export function zooms(peak: Float32Array, T: number, fps: number | null, nUnits = 10, wndMin = 5, perHour = 3): Zooms {
  // np.argsort(peak)[::-1]: ties go to the larger index first
  const order = Array.from(peak.keys()).sort((a, b) => peak[b] - peak[a] || b - a);
  const units = order.slice(0, nUnits);
  if (!fps) return { units, windows: [] };
  const step = Math.max(1, Math.trunc((3600 / perHour) * fps));
  const wnd = Math.trunc(wndMin * 60 * fps);
  const windows = [];
  for (let t0 = 0; t0 < T; t0 += step) windows.push({ start: t0, stop: Math.min(t0 + wnd, T) });
  return { units, windows };
}
