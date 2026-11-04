/** The browser pipeline's outputs as a MiniMe run folder (spec section 3): the files `minime
 * serve` and `minime score` read, plus the window factors the browser viewer uses for recon
 * frames and the motion residual. */
import { centroids } from "./derived/units";
import { type Dir, walk, writeJSON } from "./dir";
import { readNpy, writeNpy } from "./npy";
import { type Csr, csrNpz, npzBlob } from "./npz";
import { pyFloat } from "./num";

/** One window's kept outputs, by file name (`max_proj.npy`, `W.npy`, ...). */
export type WindowParts = { t0: number; t1: number; files: Map<string, Uint8Array> };

/** The pass-2 factors of a window that go to `factors/wNN/`. */
const KEEP = ["W.npy", "V.npy", "s.npy", "mean.npy", "k.npy", "rank.npy", "block_max.npy", "max_proj.npy"];

/** What a window keeps of its stage outputs for the writer: `KEEP`, the noise, the seed stage's
 * tables and the init footprints. The rest (tile CRCs, merge pairs, ...) is dropped. */
export const WINDOW_FILES: readonly string[] = [
  ...KEEP,
  "sigma.npy",
  "seeds_init.npy",
  "mask_pnr.npy",
  "mask_gmm.npy",
  "seeds_merged.npy",
  "A_merge_data.npy",
  "A_merge_indices.npy",
  "A_merge_indptr.npy",
];

const arr = (w: WindowParts, name: string) => {
  const b = w.files.get(name);
  return b ? readNpy(b) : null;
};

/** The rows of `sets` (flat (n, 2) positions, one set per window, in time order) that are more
 * than `radius` px from every row kept from an earlier set. A window's own rows are never
 * compared with each other: its seed stages have already decided which of them stay. */
function dedupe(sets: Float64Array[], radius: number): Float32Array {
  const kept: number[] = [];
  const r2 = radius * radius;
  for (const s of sets) {
    const before = kept.length;
    for (let i = 0; i < s.length; i += 2) {
      let near = false;
      for (let j = 0; j < before && !near; j += 2) near = (kept[j] - s[i]) ** 2 + (kept[j + 1] - s[i + 1]) ** 2 <= r2;
      if (!near) kept.push(s[i], s[i + 1]);
    }
  }
  return Float32Array.from(kept);
}

/** The first two columns of an (n, c) table as flat (n, 2) positions (the engine's initial seeds
 * carry a third column). */
function positions(a: { shape: number[]; data: ArrayLike<number> }, keep?: (i: number) => boolean): Float64Array {
  const cols = a.shape[1] ?? 2;
  const out: number[] = [];
  for (let i = 0; i < a.shape[0]; i++) if (!keep || keep(i)) out.push(a.data[cols * i], a.data[cols * i + 1]);
  return Float64Array.from(out);
}

/** Each seed step over the session, as (n, 2) row/col: windows in time order, a seed within
 * `radius` px of one already kept is the same seed (the earliest window's position stays).
 * `pnr` is the initial seeds that passed the PNR threshold, as main's `seeds/pnr.npy` (the
 * opt-in GMM filter comes after it and is not part of that step). */
export function sessionSeeds(
  windows: WindowParts[],
  radius = 3,
): { init: Float32Array; pnr: Float32Array; merged: Float32Array } {
  const init: Float64Array[] = [];
  const pnr: Float64Array[] = [];
  const merged: Float64Array[] = [];
  for (const w of windows) {
    const s = arr(w, "seeds_init.npy");
    const mp = arr(w, "mask_pnr.npy");
    const m = arr(w, "seeds_merged.npy");
    if (!s || !mp || !m) throw new Error(`window ${w.t0}: the seed stage gave no seeds`);
    init.push(positions(s));
    pnr.push(positions(s, (i) => mp.data[i] !== 0));
    merged.push(positions(m));
  }
  return { init: dedupe(init, radius), pnr: dedupe(pnr, radius), merged: dedupe(merged, radius) };
}

function csrOf(w: WindowParts, prefix: string, P: number): Csr {
  const get = (k: string) => {
    const a = arr(w, `${prefix}_${k}.npy`);
    if (!a) throw new Error(`window ${w.t0}: no ${prefix}_${k}.npy`);
    return a.data;
  };
  const indptr = Int32Array.from(get("indptr"));
  return {
    shape: [indptr.length - 1, P],
    data: Float32Array.from(get("data")),
    indices: Int32Array.from(get("indices")),
    indptr,
  };
}

type Row = { idx: Int32Array; val: Float32Array; norm: number };

/** Dot product of two sparse rows whose pixels ascend (as in every CSR the engines and scipy write). */
function dot(a: Row, b: Row): number {
  let d = 0;
  for (let i = 0, j = 0; i < a.idx.length && j < b.idx.length; )
    if (a.idx[i] === b.idx[j]) d += a.val[i++] * b.val[j++];
    else if (a.idx[i] < b.idx[j]) i++;
    else j++;
  return d;
}

/** The windows' init footprints over the session: a footprint with cosine >= `cosThres` to one
 * kept from an earlier window is the same unit (the earliest window's footprint stays; units of
 * one window are never compared, the init merge has seen them). */
export function sessionInit(windows: WindowParts[], P: number, cosThres: number): Csr {
  const rows: Row[] = [];
  for (const w of windows) {
    const a = csrOf(w, "A_merge", P);
    const fresh: Row[] = [];
    for (let k = 0; k < a.shape[0]; k++) {
      const idx = a.indices.slice(a.indptr[k], a.indptr[k + 1]);
      const val = a.data.slice(a.indptr[k], a.indptr[k + 1]);
      let sq = 0;
      for (const v of val) sq += v * v;
      const row = { idx, val, norm: Math.sqrt(sq) };
      const same = rows.some((r) => row.norm > 0 && r.norm > 0 && dot(r, row) / (row.norm * r.norm) >= cosThres);
      if (!same) fresh.push(row);
    }
    rows.push(...fresh);
  }
  const indptr = new Int32Array(rows.length + 1);
  rows.forEach((r, k) => {
    indptr[k + 1] = indptr[k] + r.idx.length;
  });
  const indices = new Int32Array(indptr[rows.length]);
  const data = new Float32Array(indptr[rows.length]);
  rows.forEach((r, k) => {
    indices.set(r.idx, indptr[k]);
    data.set(r.val, indptr[k]);
  });
  return { shape: [rows.length, P], data, indices, indptr };
}

/** A float32 as pandas writes it: the fewest digits that read back as the same float32. */
function float32Text(x: number): string {
  if (!Number.isFinite(x)) return "";
  for (let p = 1; p <= 9; p++) {
    const v = Number(x.toPrecision(p));
    if (Math.fround(v) === x) return pyFloat(v);
  }
  return pyFloat(x);
}

/** `cnmf/units.csv` as main's `_write_cnmf` writes it: `unit_id, centroid_h, centroid_w, size_px,
 * peak`. The centroid is `init.centroids` (float64, blank when the footprint is empty), the
 * numbers print like pandas' `to_csv`. */
export function unitsCsv(a: Csr, H: number, W: number, peak: Float32Array): string {
  const cent = centroids(a, H, W);
  const num = (v: number) => (Number.isNaN(v) ? "" : pyFloat(v));
  const lines = ["unit_id,centroid_h,centroid_w,size_px,peak"];
  for (let k = 0; k < a.shape[0]; k++)
    lines.push(
      `${k},${num(cent[2 * k])},${num(cent[2 * k + 1])},${a.indptr[k + 1] - a.indptr[k]},${float32Text(peak[k])}`,
    );
  return `${lines.join("\n")}\n`;
}

export type RunParts = {
  /** `session.json`'s object (window length and each stage's parameters). */
  session: Record<string, unknown>;
  /** The flat parameters (MiniMe's names) plus `raw_dir` and `fps`. */
  params: Record<string, unknown>;
  report: Record<string, unknown>;
  H: number;
  W: number;
  T: number;
  tile: number;
  windows: WindowParts[];
  /** The motion stage's `shifts.npy`, `template.npy` and `quality.npy`, and its parameters. */
  motion: { shifts: Uint8Array; template: Uint8Array; quality: Uint8Array; params: Record<string, unknown> };
  /** The stitch's outputs: `A_{data,indices,indptr}.npy`, `sn.npy`, `bright.npy`. */
  stitch: Map<string, Uint8Array>;
  /** The streaming pass's outputs: `b.npy`, `Amerged_{data,indices,indptr}.npy`. */
  stream: Map<string, Uint8Array | Blob>;
  /** `C.npy`, (K, T) float32, read from where the pass wrote it. */
  traces: Blob;
  /** `progress.jsonl`'s lines. */
  progress: string[];
};

const need = (m: Map<string, Uint8Array>, k: string): Uint8Array => {
  const b = m.get(k);
  if (!b) throw new Error(`the run gave no ${k}`);
  return b;
};

const f32Image = (H: number, W: number, name: string, data: ArrayLike<number>): Uint8Array => {
  if (data.length !== H * W) throw new Error(`${name} holds ${data.length} values, not H * W = ${H * W}`);
  return writeNpy([H, W], Float32Array.from(data));
};

/** The run folder of spec section 3. `report.json` goes last: a run is listed only when it has
 * `params.json` and `report.json`, so a folder whose run failed halfway is never listed. */
export async function writeRun(out: Dir, r: RunParts): Promise<void> {
  if (!r.windows.length) throw new Error("the run has no windows");
  const P = r.H * r.W;
  await writeJSON(out, "params.json", { ...r.params, ...r.session, source: "browser" });
  await out.write("progress.jsonl", `${r.progress.join("\n")}\n`);
  const motion = await walk(out, "motion", { create: true });
  await writeJSON(motion, "meta.json", { format_version: 1, params: r.motion.params, raw_hash: "" });
  await motion.write(
    "arrays.npz",
    await npzBlob({ shifts: r.motion.shifts, template: r.motion.template, quality: r.motion.quality }),
  );

  const factors = await walk(out, "factors", { create: true });
  const ranks = r.windows.map((w) => Array.from(readNpy(need(w.files, "rank.npy")).data as ArrayLike<number>));
  await writeJSON(factors, "meta.json", {
    H: r.H,
    W: r.W,
    T: r.T,
    tile: r.tile,
    ranks: ranks[0].map((_, j) => Math.max(...ranks.map((rk) => rk[j]))),
    windows: r.windows.map((w) => ({ t0: w.t0, t1: w.t1 })),
  });
  const maxProj = new Float32Array(P).fill(Number.NEGATIVE_INFINITY);
  for (const w of r.windows) {
    const m = readNpy(need(w.files, "max_proj.npy")).data;
    for (let i = 0; i < P; i++) maxProj[i] = Math.max(maxProj[i], m[i]);
  }
  await factors.write("max_proj.npy", f32Image(r.H, r.W, "max_proj", maxProj));
  await factors.write("sn.npy", f32Image(r.H, r.W, "sn", readNpy(need(r.stitch, "sn.npy")).data));
  for (const [i, w] of r.windows.entries()) {
    const d = await walk(factors, `w${String(i).padStart(2, "0")}`, { create: true });
    for (const k of KEEP) await d.write(k, need(w.files, k));
  }

  const cnmf = await walk(out, "cnmf", { create: true });
  const A = csrOf({ t0: 0, t1: 0, files: r.stitch }, "A", P);
  await cnmf.write("A.npz", await csrNpz(A));
  await cnmf.write("C.npy", r.traces);
  const peak = Float32Array.from(readNpy(need(r.stitch, "bright.npy")).data);
  await cnmf.write("units.csv", unitsCsv(A, r.H, r.W, peak));
  const b = r.stream.get("b.npy");
  if (b instanceof Uint8Array) await cnmf.write("b.npy", f32Image(r.H, r.W, "b", readNpy(b).data));
  const bytes = new Map([...r.stream].filter((e): e is [string, Uint8Array] => e[1] instanceof Uint8Array));
  if (bytes.has("Amerged_data.npy"))
    await cnmf.write("Amerged.npz", await csrNpz(csrOf({ t0: 0, t1: 0, files: bytes }, "Amerged", P)));

  const seeds = sessionSeeds(r.windows);
  const sd = await walk(out, "seeds", { create: true });
  for (const k of ["init", "pnr", "merged"] as const)
    await sd.write(`${k}.npy`, writeNpy([seeds[k].length / 2, 2], seeds[k]));
  const stitch = r.session.stitch as { cos_thres?: number } | undefined;
  const init = sessionInit(r.windows, P, Number(stitch?.cos_thres ?? 0.7));
  await (await walk(out, "init", { create: true })).write("A.npz", await csrNpz(init));
  await writeJSON(out, "report.json", r.report);
}
