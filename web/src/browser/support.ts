import { fmtBytes } from "../format";

/** What this browser offers MiniMe: the system folder dialog (File System Access, Chromium only)
 * and OPFS ("Browser storage"; without it the app cannot run), and what a run needs of storage. */

export function support(): { fsAccess: boolean; opfs: boolean } {
  return {
    fsAccess: "showDirectoryPicker" in globalThis,
    opfs: typeof globalThis.navigator?.storage?.getDirectory === "function",
  };
}

/** The system's folder dialog, asking for write access at once: a permission prompt later in a
 * request would have no user activation. `id` makes the dialog open where it last was for that id. */
export function pickDirectory(id: "minime" | "minime-out"): Promise<FileSystemDirectoryHandle> {
  // biome-ignore lint/suspicious/noExplicitAny: showDirectoryPicker is Chromium-only (not in the DOM lib)
  return (globalThis as any).showDirectoryPicker({ id, mode: "readwrite" });
}

/** The Web Lock a run holds from its start to its end, in whatever tab runs it: one run at a time
 * for the site (its tabs share browser storage). */
export const RUN_LOCK = "minime-run";

/** The unit count is not known before a run; a 3 h session finds about 400. */
export const ASSUMED_UNITS = 500;
/** `factors/wNN/` of one 3000-frame window takes about 28 MB at 600 x 600 px (measured on a browser run). */
const FACTOR_BYTES_PER_PIXEL = 80;

/** The most a run writes to browser storage, in bytes, for `K` units, `T` frames of `P` pixels
 * in `windows` windows:
 *   2 * K * T * 4        the traces twice: the scratch `stream_out/C.npy` is still there when
 *                        `cnmf/C.npy` is written
 * + windows * P * 80     the window factors (`factors/wNN/`)
 * + T * P                the tile store (a byte per pixel and frame), only with `keepStore` */
export function estimateRunBytes(K: number, T: number, P: number, windows: number, keepStore = false): number {
  return 2 * K * T * 4 + windows * P * FACTOR_BYTES_PER_PIXEL + (keepStore ? T * P : 0);
}

/** The words of a storage refusal, with the numbers: `need` bytes, `free` of them free. A run in a
 * folder on disk (`onDisk`) still passes its traces through browser storage. */
export function quotaMessage(need: number, free: number, onDisk = false): string {
  const room = `${fmtBytes(need)} of browser storage, ${fmtBytes(Math.max(0, free))} are free`;
  return onDisk
    ? `this run needs about ${room} for its traces on their way to the run folder; make room in the browser's storage for this site and start it again`
    : `this run needs about ${room}; pick a folder on disk under Runs → Change folder`;
}

/** The second storage check, once the stitch has found the run's `K` units (the first one, before
 * the run, had to guess them): the streaming pass's traces, K x T float32, go to browser storage
 * (the scratch `C.npy`) and, for a run in Browser storage, once more into the run folder
 * (`cnmf/C.npy`). Throws the refusal, with the numbers, before the pass allocates `C.npy` rather
 * than failing in it after hours of windows; nothing is checked where the browser cannot say how
 * much room there is. Works in a worker. */
export async function checkTraceRoom(
  K: number,
  T: number,
  inBrowserStorage: boolean,
  storage: Pick<StorageManager, "estimate"> | undefined = globalThis.navigator?.storage,
): Promise<void> {
  const { quota, usage } = (await storage?.estimate?.().catch(() => undefined)) ?? {};
  if (quota === undefined || usage === undefined) return;
  const need = (inBrowserStorage ? 2 : 1) * K * T * 4;
  if (need > quota - usage) throw new Error(quotaMessage(need, quota - usage, !inBrowserStorage));
}
