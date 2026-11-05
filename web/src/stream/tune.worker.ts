/// <reference lib="webworker" />
// A tune worker: holds a few windows' WindowEngines and runs one setting on each of them.
import init, { WindowEngine } from "./pkg/minime_core.js";
import type { FromTuneWorker, ToTuneWorker, TrialOut } from "./tuneProtocol";

const post = (m: FromTuneWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);
let memory: WebAssembly.Memory | null = null;
let engines: { i: number; t0: number; t1: number; e: WindowEngine }[] = [];
const heapMb = () => (memory?.buffer.byteLength ?? 0) / 2 ** 20;
let queue: Promise<void> = Promise.resolve();

async function handle(m: ToTuneWorker): Promise<void> {
  switch (m.type) {
    case "load": {
      memory ??= (await init()).memory;
      for (const { e } of engines) e.free();
      engines = [];
      let bytes = 0;
      for (const w of m.windows) {
        const files = new Map<string, Uint8Array>(w.files);
        const e = new WindowEngine(m.runSession, w.t0, w.t1, files);
        bytes += e.heap_hint();
        engines.push({ i: w.i, t0: w.t0, t1: w.t1, e });
      }
      return post({ type: "ack", heapMb: heapMb(), bytes });
    }
    case "trial": {
      const caps = m.caps ? (JSON.parse(m.caps) as Record<string, { seeds_pnr: number; seeds_merged: number }>) : {};
      const results = engines.map(({ i, e }) => ({
        i,
        out: JSON.parse(
          e.trial(m.session, m.checks, m.reference, caps[String(i)] ? JSON.stringify(caps[String(i)]) : ""),
        ) as TrialOut,
      }));
      return post({ type: "trial", id: m.id, results, heapMb: heapMb() });
    }
    case "units": {
      const windows = engines.map(({ i, t0, t1, e }) => ({ i, t0, t1, bytes: e.window_bytes() }));
      return post(
        { type: "units", id: m.id, windows },
        windows.map((w) => w.bytes.buffer),
      );
    }
    case "hist": {
      const hist: number[] = [];
      for (const { e } of engines) {
        const h = Array.from(e.contrast_hist(m.seeds_wnd, m.seeds_stp, m.seeds_max_wnd, m.n_windows), Number);
        for (let c = 0; c < h.length; c++) hist[c] = (hist[c] ?? 0) + h[c];
      }
      return post({ type: "hist", id: m.id, hist });
    }
  }
}

self.onmessage = (e: MessageEvent<ToTuneWorker>) => {
  const id = "id" in e.data ? e.data.id : -1;
  queue = queue
    .then(() => handle(e.data))
    .catch((err: unknown) => {
      post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
    });
};
