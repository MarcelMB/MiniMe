/** `np.percentile(a, q)` (linear) of a float32 array, given sorted: numpy takes the difference
 * of the two neighbours in float32, then interpolates from the nearer one in float64. */
export function percentile(sorted: Float32Array, q: number): number {
  const pos = (sorted.length - 1) * (q / 100);
  const i = Math.floor(pos);
  const t = pos - i;
  const [a, b] = [sorted[i], sorted[Math.min(i + 1, sorted.length - 1)]];
  const diff = Math.fround(b - a);
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

/** `np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)` for a float32 `f`: numpy
 * keeps every step in float32, so the same rounding happens here. */
export function toU8(f: Float32Array, lo: number, hi: number): Uint8Array {
  const out = new Uint8Array(f.length);
  const [lo32, s32] = [Math.fround(lo), Math.fround(255 / (hi - lo))];
  for (let i = 0; i < f.length; i++) {
    const v = Math.fround(Math.fround(f[i] - lo32) * s32);
    out[i] = Math.min(255, Math.max(0, Math.trunc(v)));
  }
  return out;
}

/** `np.linspace(0, stop, n).astype(int)`: the step first, then the multiple (the last one is `stop`). */
export function linspaceInt(stop: number, n: number): number[] {
  const step = n > 1 ? stop / (n - 1) : 0;
  return Array.from({ length: n }, (_, i) => (n > 1 && i === n - 1 ? stop : Math.trunc(i * step)));
}
