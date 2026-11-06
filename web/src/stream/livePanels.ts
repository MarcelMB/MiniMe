// Live view panels from what each stage already has (its output files, tile blocks, factors): only
// called while `live.on`. Coordinates stay in full-resolution pixels; images are previews.
import type { Tile } from "../browser/derived/tiles";
import { centroids, unitPolygons } from "../browser/derived/units";
import { readNpy } from "../browser/npy";
import {
  brightest,
  csrFromFiles,
  decimate,
  frameFromBlock,
  makeThrottle,
  parseTileFactors,
  previewImage,
} from "./live";
import type { LiveGrid, LivePayload } from "./protocol";

const TRACE = 500;
const COMPS = 3;
const TRACES = 5;

/** (n, cols) int64/float rows → (row, col) pairs as float32, from columns `r` and `c`. */
function pairs(b: Uint8Array | undefined, r = 0, c = 1): Float32Array {
  if (!b) return new Float32Array(0);
  const { shape, data } = readNpy(b);
  const [n, cols] = [shape[0] ?? 0, shape[1] ?? 1];
  const out = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) [out[2 * i], out[2 * i + 1]] = [data[i * cols + r], data[i * cols + c]];
  return out;
}

export function motionPanel(
  block: Uint8Array,
  n: number,
  grid: Tile[],
  h: number,
  w: number,
  shifts: Float32Array,
  at: number,
  last = false,
): LivePayload {
  const p = previewImage(frameFromBlock(block, n, n - 1, grid, w), h, w);
  return { panel: "motion", frame: p.image, h: p.h, w: p.w, f: p.f, at, shifts, ...(last ? { last } : {}) };
}

export class Pass2Live {
  private ranks: Int32Array;
  private ok = makeThrottle(4);
  constructor(
    private label: string,
    private grid: LiveGrid,
    private n: number,
  ) {
    this.ranks = new Int32Array(n).fill(-1);
  }

  /** A finished tile; returns a payload when the throttle lets it through (always for the last tile). */
  tile(k: number, rank: number, factors: Uint8Array): LivePayload | null {
    this.ranks[k] = rank;
    if (!this.ok(k === this.n - 1)) return null;
    const { tile } = this.grid;
    const cols = Math.ceil(this.grid.w / tile);
    const [ty, tx] = [Math.floor(k / cols) * tile, (k % cols) * tile];
    const [th, tw] = [Math.min(tile, this.grid.h - ty), Math.min(tile, this.grid.w - tx)];
    const t = parseTileFactors(factors);
    const r = t.rank;
    const P = th * tw;
    const T = r ? t.v.length / r : 0;
    const comps: Uint8Array[] = [];
    const traces: Float32Array[] = [];
    for (let j = 0; j < Math.min(COMPS, r); j++) {
      comps.push(
        previewImage(
          Float32Array.from({ length: P }, (_, p) => t.w[p * r + j]),
          th,
          tw,
          tile,
        ).image,
      );
      traces.push(
        decimate(
          Float32Array.from({ length: T }, (_, i) => t.v[i * r + j]),
          TRACE,
        ),
      );
    }
    return { panel: "pass2", label: this.label, grid: this.grid, ranks: this.ranks.slice(), k, comps, th, tw, traces };
  }
}

export function seedsPanel(
  label: string,
  out: Map<string, Uint8Array>,
  maxProj: Uint8Array,
  h: number,
  w: number,
): LivePayload {
  const p = previewImage(maxProj, h, w);
  return {
    panel: "seeds",
    label,
    image: p.image,
    h: p.h,
    w: p.w,
    f: p.f,
    seeds: pairs(out.get("seeds_init.npy")),
    merged: pairs(out.get("seeds_merged.npy")),
  };
}

export function unitsPanel(
  label: string,
  step: "init" | "cnmf",
  out: Map<string, Uint8Array>,
  mean: Float32Array,
  h: number,
  w: number,
): LivePayload | null {
  const [aName, cName] = step === "init" ? ["A_merge", "C_merge.npy"] : ["A", "C.npy"];
  const a = csrFromFiles(out, aName, h * w);
  const cb = out.get(cName);
  if (!a || !cb) return null;
  const c = readNpy(cb);
  const [k, t] = [c.shape[0], c.shape[1]];
  const ids = brightest(c.data, k, t, TRACES);
  const p = previewImage(mean, h, w);
  return {
    panel: "units",
    label,
    step,
    image: p.image,
    h: p.h,
    w: p.w,
    f: p.f,
    polygons: unitPolygons(a, h, w),
    count: a.shape[0],
    ids,
    traces: ids.map((u) => decimate(c.data.subarray(u * t, (u + 1) * t), TRACE)),
  };
}

/** Window `window`'s units for the stitch panel; `first` is its first unit's index over all windows. */
export function stitchWindowPanel(
  window: number,
  first: number,
  files: Map<string, Uint8Array>,
  h: number,
  w: number,
): LivePayload | null {
  const a = csrFromFiles(files, "A", h * w);
  if (!a) return null;
  return { panel: "stitch", window, first, polygons: unitPolygons(a, h, w), centroids: centroids(a, h, w) };
}

export function stitchFinalPanel(out: Map<string, Uint8Array>): LivePayload | null {
  const e = out.get("edges.npy");
  const p = out.get("A_indptr.npy");
  if (!e || !p) return null;
  return { panel: "stitch", window: -1, edges: Float64Array.from(readNpy(e).data), units: readNpy(p).shape[0] - 1 };
}

const dead = new Set<string>();

/** Run `build` and post its payload; on an error log once and turn that panel off for the run. */
export function guard(
  panel: string,
  build: () => LivePayload | null,
  post: (p: LivePayload) => void,
  log: (m: string) => void,
): void {
  if (dead.has(panel)) return;
  try {
    const p = build();
    if (p) post(p);
  } catch (e) {
    dead.add(panel);
    log(`live view: ${panel}: ${e instanceof Error ? e.message : String(e)}`);
  }
}
