// Raw frames of the session's AVIs, for the pool's frame workers (`frame.worker.ts`). The same
// helpers sit, private, in `stream.worker.ts` (`indexSegments`, `forFrames`); merge them when the
// two meet.
import { orderAvis } from "./order";
import { AviFile } from "./pkg/minime_core.js";

/** Frames per `File.slice` read. */
export const BLOCK = 64;

export type Segment = { file: File; avi: AviFile; read: (off: number, len: number) => Uint8Array };

/** Index every AVI (numeric order), reading through `File.slice`. */
export function indexSegments(avis: File[]): Segment[] {
  const reader = new FileReaderSync();
  return orderAvis(avis).map((file) => {
    const read = (off: number, len: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(off, off + len)));
    return { file, read, avi: AviFile.parse(read, file.size) };
  });
}

/** Calls `push` on raw frames `[from, to)` of the session, reading `BLOCK` frames per slice. */
export function forFrames(
  segs: Segment[],
  from: number,
  to: number,
  setStride: (s: number) => void,
  push: (chunk: Uint8Array) => void,
): void {
  let s0 = 0;
  for (const { avi, read } of segs) {
    const n = avi.n_frames();
    const [lo, hi] = [Math.max(from - s0, 0), Math.min(to - s0, n)];
    if (lo < hi) {
      setStride(avi.size(0) === avi.height() * avi.width() ? avi.width() : (avi.width() + 3) & ~3);
      for (let i = lo; i < hi; i += BLOCK) {
        const j = Math.min(i + BLOCK, hi);
        const start = avi.offset(i);
        const buf = read(start, avi.offset(j - 1) + avi.size(j - 1) - start);
        for (let k = i; k < j; k++) {
          const o = avi.offset(k) - start;
          push(buf.subarray(o, o + avi.size(k)));
        }
      }
    }
    s0 += n;
    if (s0 >= to) return;
  }
  if (s0 < to) throw new Error(`frames ${from}-${to} requested, the files hold ${s0}`);
}
