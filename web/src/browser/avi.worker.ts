// Indexes AVIs where FileReaderSync lets AviFile.parse read just the ranges it needs.
/// <reference lib="webworker" />
import init, { AviFile } from "../stream/pkg/minime_core.js";

self.onmessage = async (e: MessageEvent<File[]>) => {
  try {
    await init();
    const reader = new FileReaderSync();
    const index = e.data.map((file) => {
      const read = (off: number, len: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(off, off + len)));
      const avi = AviFile.parse(read, file.size);
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
    });
    self.postMessage({ index });
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
