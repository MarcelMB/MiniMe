// Indexes AVIs where FileReaderSync lets AviFile.parse read just the ranges it needs.
/// <reference lib="webworker" />
import init, { AviFile } from "../stream/pkg/minime_core.js";
import { segIndex } from "./segindex";

self.onmessage = async (e: MessageEvent<File[]>) => {
  try {
    await init();
    const reader = new FileReaderSync();
    const index = e.data.map((file) => {
      const read = (off: number, len: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(off, off + len)));
      return segIndex(AviFile.parse(read, file.size));
    });
    self.postMessage({ index });
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
