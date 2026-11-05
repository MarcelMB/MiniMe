import { exists, fileAt, readBytes } from "../dir";
import { HttpError } from "../router";
import type { RunInfo } from "../runs";
import { windowDir } from "../writer";

export type WindowFiles = { i: number; t0: number; t1: number; files: Map<string, Uint8Array> };
export const WINDOW_FACTOR_FILES = [
  "W.npy",
  "V.npy",
  "mean.npy",
  "sigma.npy",
  "energy.npy",
  "rank.npy",
  "block_max.npy",
  "max_proj.npy",
];

export const windowsOf = (r: RunInfo): { t0: number; t1: number }[] => r.meta.windows ?? [{ t0: 0, t1: r.meta.T }];

/** The eight factor files of window `i` of `r` (its source run's for a rerun). */
export async function loadWindowFiles(r: RunInfo, i: number): Promise<WindowFiles> {
  const d = `factors/${windowDir(i)}`;
  for (const name of ["sigma.npy", "energy.npy"])
    if (!(await exists(r.factors, `${d}/${name}`)))
      throw new HttpError(
        400,
        `this run was made before auto-tune: run it again to tune or rerun it (${d}/${name} is missing)`,
      );
  const files = new Map<string, Uint8Array>();
  for (const name of WINDOW_FACTOR_FILES) {
    if (!(await exists(r.factors, `${d}/${name}`))) throw new HttpError(404, `${d}/${name} is missing from this run`);
    files.set(name, await readBytes(r.factors, `${d}/${name}`));
  }
  const { t0, t1 } = windowsOf(r)[i];
  return { i, t0, t1, files };
}

/** Bytes of window 0's factor files (what one window costs a tune worker before its working set). */
export async function windowBytes(r: RunInfo): Promise<number> {
  let n = 0;
  for (const name of WINDOW_FACTOR_FILES) n += (await fileAt(r.factors, `factors/${windowDir(0)}/${name}`)).size;
  return n;
}
