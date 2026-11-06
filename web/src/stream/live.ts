// Live view previews, worker side: built only while `live.on` (the job page's switch); every hook
// checks the flag first, so a run with the switch off does no preview work at all.
import type { Tile } from "../browser/derived/tiles";
import { readNpy } from "../browser/npy";
import type { Csr } from "../browser/npz";
import type { LivePayload } from "./protocol";

export const live = { on: false };

/** True at most `hz` times a second; `force` (a step's first or last preview) always passes. */
export function makeThrottle(hz: number, now: () => number = () => performance.now()) {
  let last = Number.NEGATIVE_INFINITY;
  return (force = false): boolean => {
    const t = now();
    if (!force && t - last < 1000 / hz) return false;
    last = t;
    return true;
  };
}

/** Block mean by `f = ceil(max(h, w) / max)`, then a 0.5-99.5 % stretch to u8. */
export function previewImage(
  src: ArrayLike<number>,
  h: number,
  w: number,
  max = 256,
): { image: Uint8Array; h: number; w: number; f: number } {
  const f = Math.max(1, Math.ceil(Math.max(h, w) / max));
  const [dh, dw] = [Math.ceil(h / f), Math.ceil(w / f)];
  const sum = new Float32Array(dh * dw);
  const cnt = new Float32Array(dh * dw);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = Math.floor(y / f) * dw + Math.floor(x / f);
      sum[i] += src[y * w + x];
      cnt[i] += 1;
    }
  for (let i = 0; i < sum.length; i++) sum[i] /= cnt[i];
  const sorted = Float32Array.from(sum).sort();
  const lo = sorted[Math.floor(0.005 * (sorted.length - 1))];
  const hi = sorted[Math.ceil(0.995 * (sorted.length - 1))];
  const image = new Uint8Array(sum.length);
  const span = hi > lo ? 255 / (hi - lo) : 0;
  for (let i = 0; i < sum.length; i++) image[i] = Math.max(0, Math.min(255, Math.round((sum[i] - lo) * span)));
  return { image, h: dh, w: dw, f };
}

/** Frame `frame` of a `take_tiles()` block of `n` frames (tile after tile, each `n x pixels`). */
export function frameFromBlock(block: Uint8Array, n: number, frame: number, grid: Tile[], w: number): Uint8Array {
  const h = grid.reduce((m, t) => Math.max(m, t.y1), 0);
  const out = new Uint8Array(h * w);
  let o = 0;
  for (const t of grid) {
    const tw = t.x1 - t.x0;
    const px = (t.y1 - t.y0) * tw;
    const base = o + frame * px;
    for (let y = t.y0; y < t.y1; y++) {
      const row = base + (y - t.y0) * tw;
      out.set(block.subarray(row, row + tw), y * w + t.x0);
    }
    o += n * px;
  }
  return out;
}

export type TileFactorsView = { rank: number; w: Float32Array; v: Float32Array; s: Float32Array; mean: Float32Array };

/** `TileFactors::to_bytes` (poolio.rs): u32 rank, k, len(w), len(v), len(s), len(mean), len(sigma),
 * len(energy); then w (P x r, scaled by s), v (T x r), s, mean, sigma as f32; energy f64. */
export function parseTileFactors(b: Uint8Array): TileFactorsView {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const len = (i: number) => dv.getUint32(4 * i, true);
  let at = 32;
  const f32 = (n: number) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = dv.getFloat32(at + 4 * i, true);
    at += 4 * n;
    return out;
  };
  const rank = len(0);
  const w = f32(len(2));
  const v = f32(len(3));
  const s = f32(len(4));
  const mean = f32(len(5));
  return { rank, w, v, s, mean };
}

/** The mean image of a frame from its tiles' factors (`TileFactors.mean` of each tile). */
export function meanImage(tiles: Uint8Array[], grid: Tile[], h: number, w: number): Float32Array {
  const out = new Float32Array(h * w);
  grid.forEach((t, k) => {
    const m = parseTileFactors(tiles[k]).mean;
    const tw = t.x1 - t.x0;
    for (let y = t.y0; y < t.y1; y++) out.set(m.subarray((y - t.y0) * tw, (y - t.y0 + 1) * tw), y * w + t.x0);
  });
  return out;
}

/** At most `max` points: block means (the input as is when it is short enough). */
export function decimate(x: ArrayLike<number>, max: number): Float32Array {
  if (x.length <= max) return Float32Array.from(x);
  const out = new Float32Array(max);
  for (let i = 0; i < max; i++) {
    const [a, b] = [Math.floor((i * x.length) / max), Math.floor(((i + 1) * x.length) / max)];
    let s = 0;
    for (let j = a; j < b; j++) s += x[j];
    out[i] = s / (b - a);
  }
  return out;
}

/** `<name>_indptr/_indices/_data.npy` (int64, int64, f32) as a `Csr` of `cols` columns. */
export function csrFromFiles(files: Map<string, Uint8Array>, name: string, cols: number): Csr | null {
  const get = (part: string) => files.get(`${name}_${part}.npy`);
  const [p, i, d] = [get("indptr"), get("indices"), get("data")];
  if (!p || !i || !d) return null;
  const indptr = Int32Array.from(readNpy(p).data);
  return {
    shape: [indptr.length - 1, cols],
    indptr,
    indices: Int32Array.from(readNpy(i).data),
    data: Float32Array.from(readNpy(d).data),
  };
}

/** The `n` units (rows of `c`, `k x t`) with the highest peak, brightest first. */
export function brightest(c: ArrayLike<number>, k: number, t: number, n: number): number[] {
  const peak = Array.from({ length: k }, (_, u) => {
    let m = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < t; j++) m = Math.max(m, c[u * t + j]);
    return [m, u] as const;
  });
  return peak
    .sort((a, b) => b[0] - a[0])
    .slice(0, n)
    .map(([, u]) => u);
}

/** The tile size of the run: pass 1's "tile" (a window's pass 1 params are `params.json`, as
 * `window_inputs` names them; `pass1.json` in a reference folder). */
export function tileOf(files: Map<string, Uint8Array>): number {
  const b = files.get("params.json") ?? files.get("pass1.json");
  if (!b) throw new Error("no pass 1 params");
  return Number(JSON.parse(new TextDecoder().decode(b)).tile);
}

/** The buffers of a payload's typed arrays, to post it without copying. */
export function liveTransfer(p: LivePayload): Transferable[] {
  const out = new Set<ArrayBuffer>();
  const add = (v: unknown) => {
    if (ArrayBuffer.isView(v)) out.add(v.buffer as ArrayBuffer);
    else if (Array.isArray(v)) for (const x of v) if (ArrayBuffer.isView(x)) out.add(x.buffer as ArrayBuffer);
  };
  for (const v of Object.values(p)) add(v);
  return [...out];
}
