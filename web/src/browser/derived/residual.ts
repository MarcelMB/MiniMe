/** The motion residual of a run (main's `derived.compute_residual`): how far each block's max
 * projection sits from the whole run's, which a registration that held keeps near 0 px. */
import type { RunCache } from "../cache";
import { exists, readBytes } from "../dir";
import { readNpy } from "../npy";
import { HttpError } from "../router";
import type { RunInfo } from "../runs";
import { wasm } from "../wasm";
import { windowDir } from "../writer";

const f32 = async (r: RunInfo, rel: string): Promise<{ shape: number[]; data: Float32Array }> => {
  if (!(await exists(r.dir, rel)))
    throw new HttpError(404, `the motion residual is not available for this run: ${rel}`);
  const a = readNpy(await readBytes(r.dir, rel));
  return { shape: a.shape, data: a.data instanceof Float32Array ? a.data : Float32Array.from(a.data) };
};

/** The files of block maxima in time order: every window's `factors/wNN/block_max.npy` of a browser
 * run, else the one `factors/block_max.npy` of a Python run (its blocks span the whole run); none
 * for a run that kept neither. */
export async function blockFiles(r: RunInfo): Promise<string[]> {
  const windows = r.meta.windows ?? [{ t0: 0, t1: r.meta.T }];
  const perWindow = windows.map((_, w) => `factors/${windowDir(w)}/block_max.npy`);
  if (await exists(r.dir, perWindow[0])) return perWindow;
  return (await exists(r.dir, "factors/block_max.npy")) ? ["factors/block_max.npy"] : [];
}

/** Compute `residual.npy` (n_blocks, 3: dy, dx, correlation) into the run's cache: every block
 * file's blocks (`blockFiles`, in time order) against the session's `factors/max_proj.npy`, with
 * the pass-1 parameters of `params.json` (the `pass1` object of a browser run, else the flat
 * parameters). File by file, so that only one window's blocks are in memory at a time. */
export async function computeResidual(r: RunInfo, cache: RunCache): Promise<void> {
  const { H, W } = r.meta;
  const [core, maxProj] = [await wasm(), await f32(r, "factors/max_proj.npy")];
  if (maxProj.data.length !== H * W)
    throw new HttpError(500, `factors/max_proj.npy holds ${maxProj.data.length} values, not H * W = ${H * W}`);
  const params = JSON.stringify(r.params.pass1 ?? r.params);
  const parts: Float32Array[] = [];
  let n = 0;
  for (const file of await blockFiles(r)) {
    const blocks = await f32(r, file);
    const [nb, h, wd] = blocks.shape;
    if (blocks.shape.length !== 3 || h !== H || wd !== W)
      throw new HttpError(500, `${file} is shaped (${blocks.shape.join(", ")}), not (n, ${H}, ${W})`);
    parts.push(core.block_residual(blocks.data, nb, H, W, maxProj.data, params));
    n += nb;
  }
  if (!n) throw new HttpError(500, "the run has no block maxima");
  const out = new Float32Array(3 * n);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  await cache.saveNpy("residual.npy", [n, 3], out);
}
