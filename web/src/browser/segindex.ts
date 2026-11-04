import type { AviFile } from "../stream/pkg/minime_core.js";

/** Frame count, size and the byte range of each frame of one AVI. */
export type SegIndex = { n: number; height: number; width: number; offsets: number[]; sizes: number[] };

/** The index of a parsed AVI, which is freed (the wasm object holds the frame table). Shared by
 * the page (Node tests) and `avi.worker.ts`, so it imports nothing that needs a router or a DOM. */
export function segIndex(avi: AviFile): SegIndex {
  const n = avi.n_frames();
  const out = {
    n,
    height: avi.height(),
    width: avi.width(),
    offsets: Array.from({ length: n }, (_, i) => avi.offset(i)),
    sizes: Array.from({ length: n }, (_, i) => avi.size(i)),
  };
  avi.free();
  return out;
}
