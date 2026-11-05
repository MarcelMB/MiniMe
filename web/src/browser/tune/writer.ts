/** A window-2 rerun as a run folder (`window2.Window2Session.run`'s layout): the session-level
 * files written anew, `factors/meta.json`, `max_proj.npy`, `sn.npy`, `motion/`, `seeds/` and
 * `init/` copied from the source run (the windows' factors stay there: `params.source_run` points
 * the registry at them). `WindowEngine` does not return the windows' seeds and init, so the
 * rerun carries the source run's `seeds/` and `init/`: the Seeds page of a rerun shows the run's
 * seeds, as the registry follows `source_run`. */
import { type Dir, exists, readBytes, walk, writeJSON } from "../dir";
import { readNpy, writeNpy } from "../npy";
import { type Csr, csrNpz } from "../npz";
import type { RunInfo } from "../runs";
import { unitsCsv } from "../writer";
import type { Params } from "./params";

export type RerunParts = {
  session: Record<string, unknown>;
  params: Params;
  sourceRun: string;
  rawDir: string;
  fps: number | null;
  H: number;
  W: number;
  T: number;
  tile: number;
  windows: { t0: number; t1: number }[];
  stitch: Map<string, Uint8Array>;
  stream: Map<string, Uint8Array | Blob>;
  traces: Blob;
  yra: Blob;
  report: Record<string, unknown>;
  progress: string[];
};

const need = (m: Map<string, Uint8Array | Blob>, k: string): Uint8Array => {
  const b = m.get(k);
  if (!(b instanceof Uint8Array)) throw new Error(`the rerun gave no ${k}`);
  return b;
};

function csrOf(m: Map<string, Uint8Array | Blob>, prefix: string, P: number): Csr {
  const indptr = Int32Array.from(readNpy(need(m, `${prefix}_indptr.npy`)).data);
  return {
    shape: [indptr.length - 1, P],
    data: Float32Array.from(readNpy(need(m, `${prefix}_data.npy`)).data),
    indices: Int32Array.from(readNpy(need(m, `${prefix}_indices.npy`)).data),
    indptr,
  };
}

async function copyTree(from: Dir, to: Dir, rel: string): Promise<void> {
  if (!(await hasDir(from, rel))) return;
  const src = await walk(from, rel);
  const dst = await walk(to, rel, { create: true });
  for (const e of await src.entries()) {
    if (e.kind === "file") await dst.write(e.name, await readBytes(src, e.name));
  }
}

const hasDir = (d: Dir, rel: string): Promise<boolean> =>
  walk(d, rel).then(
    () => true,
    () => false,
  );

const f32 = (shape: number[], data: ArrayLike<number>): Uint8Array => writeNpy(shape, Float32Array.from(data));

/** `params.json` first (the folder is a MiniMe run from then on, so a failed rerun is replaced,
 * not refused), the outputs, `report.json` last. */
export async function writeRerun(out: Dir, src: RunInfo, r: RerunParts): Promise<void> {
  const P = r.H * r.W;
  await writeJSON(out, "params.json", {
    ...r.params,
    ...r.session,
    source: "browser",
    raw_dir: r.rawDir,
    fps: r.fps,
    source_run: r.sourceRun,
  });
  await out.write("progress.jsonl", `${r.progress.join("\n")}\n`);
  const factors = await walk(out, "factors", { create: true });
  await writeJSON(factors, "meta.json", {
    H: r.H,
    W: r.W,
    T: r.T,
    tile: r.tile,
    ranks: src.meta.ranks,
    windows: r.windows,
  });
  await factors.write("max_proj.npy", await readBytes(src.dir, "factors/max_proj.npy"));
  const sn = r.stitch.get("sn.npy");
  if (sn) await factors.write("sn.npy", f32([r.H, r.W], readNpy(sn).data));
  else if (await exists(src.dir, "factors/sn.npy"))
    await factors.write("sn.npy", await readBytes(src.dir, "factors/sn.npy"));
  for (const d of ["motion", "seeds", "init"]) await copyTree(src.dir, out, d);
  const cnmf = await walk(out, "cnmf", { create: true });
  const A = csrOf(r.stitch, "A", P);
  await cnmf.write("A.npz", await csrNpz(A));
  await cnmf.write("C.npy", r.traces);
  await cnmf.write("YrA.npy", r.yra);
  await cnmf.write("units.csv", unitsCsv(A, r.H, r.W, Float32Array.from(readNpy(need(r.stitch, "bright.npy")).data)));
  const b = r.stream.get("b.npy");
  if (b instanceof Uint8Array) await cnmf.write("b.npy", f32([r.H, r.W], readNpy(b).data));
  if (r.stream.has("Amerged_data.npy")) await cnmf.write("Amerged.npz", await csrNpz(csrOf(r.stream, "Amerged", P)));
  await writeJSON(out, "report.json", r.report);
}

/** `report.json` of a rerun: the stages, main's counts and the windows. */
export function rerunReport(
  stages: { stage: string; seconds: number }[],
  windows: Record<string, unknown>[],
  stitch: Record<string, unknown>,
  stream: Record<string, unknown>,
  T: number,
  H: number,
  W: number,
  wallTotal: number,
  sourceRun: string,
  heapPeakMb: number,
): Record<string, unknown> {
  const sum = (key: string) =>
    windows.reduce((n, w) => n + Number((w.counts as Record<string, unknown>)?.[key] ?? 0), 0);
  return {
    stages,
    counts: {
      seeds_init: sum("seeds_init"),
      seeds_pnr: sum("seeds_pnr"),
      seeds_merged: sum("seeds_merged"),
      windows: windows.length,
      units_windows: stitch.units_in,
      units_stitched: stitch.units_out,
      units_kept: stream.kept,
      units_merged: stream.merged ?? null,
      units_final: stitch.units_out,
    },
    frames: T,
    H,
    W,
    windows,
    wall_total: wallTotal,
    source_run: sourceRun,
    wasm_heap_peak_mb: heapPeakMb,
    where: "browser",
  };
}
