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
