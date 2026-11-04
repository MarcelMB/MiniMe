import type { Csr } from "../npz";
import { externalContours } from "./contours";

/** `init.centroids`: (K, 2) intensity-weighted (row, col) centre of each footprint, NaN if empty. */
export function centroids(a: Csr, H: number, W: number): Float64Array {
  if (a.shape[1] !== H * W) throw new Error(`A has ${a.shape[1]} columns, expected H * W = ${H * W}`);
  const K = a.shape[0];
  const out = new Float64Array(K * 2);
  for (let k = 0; k < K; k++) {
    let [m, sy, sx] = [0, 0, 0];
    for (let j = a.indptr[k]; j < a.indptr[k + 1]; j++) {
      const [p, v] = [a.indices[j], a.data[j]];
      m += v;
      sy += v * Math.floor(p / W);
      sx += v * (p % W);
    }
    out[2 * k] = m > 0 ? sy / m : Number.NaN;
    out[2 * k + 1] = m > 0 ? sx / m : Number.NaN;
  }
  return out;
}

/** `contours.unit_polygons`: per unit, the outer outlines of its pixels >= `rel` x its peak, as
 * [x, y] points (pixel centres; a one-pixel island is a one-point polygon). Each unit is
 * outlined in the bounding box of those pixels, which gives the same polygons as the whole image
 * (the box only drops empty rows and columns) without a frame-sized mask per unit. */
export function unitPolygons(a: Csr, H: number, W: number, rel = 0.2): number[][][][] {
  const out: number[][][][] = [];
  for (let k = 0; k < a.shape[0]; k++) {
    const [lo, hi] = [a.indptr[k], a.indptr[k + 1]];
    let max = Number.NEGATIVE_INFINITY;
    for (let j = lo; j < hi; j++) max = Math.max(max, a.data[j]);
    if (hi === lo || max <= 0) {
      out.push([]);
      continue;
    }
    // main's A is float32: numpy 2 multiplies float32 by a Python float in float32
    const thr = Math.fround(Math.fround(rel) * max);
    let [y0, y1, x0, x1] = [H, -1, W, -1];
    for (let j = lo; j < hi; j++) {
      if (!(a.data[j] >= thr)) continue;
      const [y, x] = [Math.floor(a.indices[j] / W), a.indices[j] % W];
      [y0, y1, x0, x1] = [Math.min(y0, y), Math.max(y1, y), Math.min(x0, x), Math.max(x1, x)];
    }
    if (y1 < 0) {
      out.push([]);
      continue;
    }
    const [h, w] = [y1 - y0 + 1, x1 - x0 + 1];
    const m = new Uint8Array(h * w);
    for (let j = lo; j < hi; j++)
      if (a.data[j] >= thr) m[(Math.floor(a.indices[j] / W) - y0) * w + (a.indices[j] % W) - x0] = 1;
    out.push(externalContours(m, h, w).map((c) => c.map(([x, y]) => [x + x0, y + y0])));
  }
  return out;
}
