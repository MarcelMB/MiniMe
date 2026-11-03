/// <reference lib="webworker" />
import init, { AviFile, Engine } from "./pkg/minime_core.js";
import { orderAvis } from "./order";
import type { FromWorker, ToWorker } from "./protocol";

const post = (m: FromWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const BLOCK = 64; // frames per File.slice read

async function run({ avis, inputs }: ToWorker): Promise<void> {
  const tl = performance.now();
  const log = (what: string) =>
    post({ type: "log", message: `${what} at ${((performance.now() - tl) / 1e3).toFixed(1)} s` });
  const wasm = await init();
  log("wasm ready");
  const files = new Map<string, Uint8Array>();
  for (const f of inputs) files.set(f.name, new Uint8Array(await f.arrayBuffer()));
  log(`read ${files.size} input files`);
  const engine = new Engine(files);
  log("engine built");
  const [T, H, W] = [engine.frames(), engine.height(), engine.width()];
  post({ type: "start", frames: T, units: engine.units(), height: H, width: W });
  const reader = new FileReaderSync();
  const t0 = performance.now();
  let t = 0;
  let lastPost = 0;
  for (const file of orderAvis(avis)) {
    if (t === T) break;
    const read = (off: number, len: number) =>
      new Uint8Array(reader.readAsArrayBuffer(file.slice(off, off + len)));
    const avi = AviFile.parse(read, file.size);
    log(`${file.name}: ${avi.n_frames()} frames indexed`);
    if (avi.height() !== H || avi.width() !== W)
      throw new Error(`${file.name}: ${avi.height()}x${avi.width()} frames, inputs say ${H}x${W}`);
    const n = avi.n_frames();
    if (n) engine.set_stride(avi.size(0) === H * W ? W : (W + 3) & ~3);
    for (let i = 0; i < n && t < T; i += BLOCK) {
      const j = Math.min(i + BLOCK, n, i + (T - t));
      const start = avi.offset(i);
      const buf = read(start, avi.offset(j - 1) + avi.size(j - 1) - start);
      for (let k = i; k < j; k++) {
        const o = avi.offset(k) - start;
        engine.push_raw(buf.subarray(o, o + avi.size(k)));
        t++;
      }
      const now = performance.now();
      if (now - lastPost > 500) {
        lastPost = now;
        post({ type: "progress", done: t, frames: T, seconds: (now - t0) / 1e3, stateMb: engine.state_mb() });
      }
    }
    avi.free();
  }
  if (t !== T) throw new Error(`read ${t} frames, inputs say ${T}`);
  const passSeconds = (performance.now() - t0) / 1e3;
  const heapMb = wasm.memory.buffer.byteLength / 2 ** 20;
  log(`pass done, wasm heap ${heapMb.toFixed(0)} MB`);
  const timing = { pass: passSeconds, where: "browser", wasm_heap_mb: heapMb };
  const out = engine.finish(JSON.stringify(timing)) as Map<string, Uint8Array>;
  engine.free();
  post({ type: "done", files: out, seconds: (performance.now() - t0) / 1e3 }, [...out.values()].map((v) => v.buffer));
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  run(e.data).catch((err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }));
};
