/** Scoring a run folder's units (`serve.tune.compute_flags` / `compute_pairs`) through the wasm
 * `score_units`, and the cell diameter from its footprints. */
import type { Summary } from "../../stream/tuneProtocol";
import type { Checks, Flags, Pairs } from "../../types";
import { fileAt } from "../dir";
import { readNpyHeader } from "../npy";
import { readCsr } from "../npz";
import { HttpError } from "../router";
import type { RunInfo } from "../runs";
import { wasm } from "../wasm";

/** Python's order of the flag reasons (the core's JSON writes its keys sorted). */
const REASONS = ["low_pnr", "in_pieces", "bad_size", "duplicate"];

export type Scored = {
  flags: Flags & Summary;
  pairs: Pairs;
  csv: string;
  summary: Summary;
  good: [number, number][];
};

async function loadA(r: RunInfo) {
  const f = await fileAt(r.dir, "cnmf/A.npz").catch(() => {
    throw new HttpError(404, "cnmf/A.npz is not available for this run");
  });
  return readCsr(f);
}

/** `cnmf/YrA.npy` whole, (K, T) float32. */
async function loadYrA(r: RunInfo): Promise<{ data: Float32Array; K: number; T: number }> {
  const f = await fileAt(r.dir, "cnmf/YrA.npy").catch(() => {
    throw new HttpError(404, "YrA traces are not available for this run");
  });
  const bytes = new Uint8Array(await f.arrayBuffer());
  const h = readNpyHeader(bytes);
  if (h.descr !== "<f4" || h.shape.length !== 2 || h.fortran)
    throw new HttpError(500, "cnmf/YrA.npy is not a C-order (K, T) float32 array");
  const [K, T] = h.shape;
  const data = new Float32Array(K * T);
  new Uint8Array(data.buffer).set(bytes.subarray(h.offset, h.offset + K * T * 4));
  return { data, K, T };
}

export function checkDiameter(d: number, H: number, W: number): void {
  if (!(d > 0)) throw new HttpError(400, `diameter must be a number of px > 0, got ${d}`);
  if (!Number.isFinite(d) || d > Math.min(H, W))
    throw new HttpError(
      400,
      `diameter must be a finite number of px of at most ${Math.min(H, W)} (the smaller field-of-view side), got ${d}`,
    );
}

export async function estimateDiameter(r: RunInfo): Promise<number> {
  const a = await loadA(r);
  if (a.shape[0] < 10)
    throw new HttpError(
      400,
      `only ${a.shape[0]} units in ${r.vpath}: too few to estimate the cell diameter, pass --diameter`,
    );
  const core = await wasm();
  return core.estimate_diameter(a.data, a.indices, a.indptr, a.shape[1]);
}

export async function scoreRun(
  r: RunInfo,
  checks: Checks,
  minDistFrac: number,
  reference: [number, number][],
): Promise<Scored> {
  const a = await loadA(r);
  const y = await loadYrA(r);
  const core = await wasm();
  const files = core.score_units(
    a.data,
    a.indices,
    a.indptr,
    a.shape[1],
    y.data,
    y.K,
    y.T,
    r.meta.H,
    r.meta.W,
    JSON.stringify(checks),
    minDistFrac,
    reference.length ? JSON.stringify(reference) : "",
  ) as Map<string, Uint8Array>;
  const text = (k: string) => new TextDecoder().decode(files.get(k) as Uint8Array);
  const flags = JSON.parse(text("flags.json")) as Flags & Summary;
  const known = REASONS.filter((k) => k in flags.reasons);
  const extra = Object.keys(flags.reasons).filter((k) => !REASONS.includes(k));
  flags.reasons = Object.fromEntries([...known, ...extra].map((k) => [k, flags.reasons[k]]));
  return {
    flags,
    pairs: JSON.parse(text("pairs.json")),
    csv: text("unit_flags.csv"),
    summary: JSON.parse(text("summary.json")),
    good: JSON.parse(text("good.json")),
  };
}
