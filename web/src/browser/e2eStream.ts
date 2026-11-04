// The streaming pipeline on files Playwright hands over, for `e2e/stream.spec.ts`: a new worker
// gets the same `ToWorker` message a job sends, and the outputs stay in the page as blobs that
// the spec reads by slices (the traces of a long session are far too big for one message).
// Only the browser build opened with `?e2e` loads this (`e2e.ts`).
import { defaultRunName } from "../launch/params";
import type { FromWorker, ToWorker } from "../stream/protocol";
import { RUN_LOCK } from "./support";

export type StreamOptions = { keepStore?: boolean; workers?: number; tileCrc?: boolean };
export type StreamResult = {
  files: { name: string; size: number }[];
  seconds: number;
  /** Where the run folder went in browser storage (`runs/<name>`); empty with reference inputs,
   * which only return their outputs. */
  folder: string;
};

const outputs = new Map<string, Blob>();
const PROGRESS_EVERY_S = 30;

/** Run the pipeline on `avis` (disk-backed `File`s) with, optionally, a reference-inputs folder.
 * From the AVIs alone the whole pipeline writes its run folder to browser storage. The run holds
 * the run lock, as a job does, so a tab opened meanwhile leaves its scratch folder alone. */
export function runStream(avis: File[], inputs: File[], opts: StreamOptions = {}): Promise<StreamResult> {
  return navigator.locks.request(RUN_LOCK, () => stream(avis, inputs, opts));
}

async function stream(avis: File[], inputs: File[], opts: StreamOptions): Promise<StreamResult> {
  let out: FileSystemDirectoryHandle | undefined;
  let folder = "";
  if (!inputs.length) {
    const name = defaultRunName(new Date());
    const runs = await (await navigator.storage.getDirectory()).getDirectoryHandle("runs", { create: true });
    out = await runs.getDirectoryHandle(name, { create: true });
    folder = `runs/${name}`;
  }
  const worker = new Worker(new URL("../stream/stream.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise<StreamResult>((resolve, reject) => {
      let lastProgress = 0;
      worker.onmessage = (e: MessageEvent<FromWorker>) => {
        const m = e.data;
        if (m.type === "log") console.log(`[worker] ${m.message}`);
        else if (m.type === "stage") console.log(`[stream] ${m.stage} ${m.event}${m.note ? ` (${m.note})` : ""}`);
        else if (m.type === "progress" && Date.now() / 1e3 - lastProgress >= PROGRESS_EVERY_S) {
          lastProgress = Date.now() / 1e3;
          console.log(`[stream] ${m.stage} ${m.done} / ${m.frames}, state ${m.stateMb.toFixed(0)} MB`);
        } else if (m.type === "done") {
          outputs.clear();
          for (const [name, bytes] of m.files) outputs.set(name, bytes instanceof Blob ? bytes : new Blob([bytes]));
          resolve({
            files: [...outputs].map(([name, blob]) => ({ name, size: blob.size })),
            seconds: m.seconds,
            folder,
          });
        } else if (m.type === "error") reject(new Error(m.message));
      };
      worker.onerror = (e) => reject(new Error(`worker failed: ${e.message || "could not start"}`));
      // one scratch folder for these runs, cleared by the next one (reference runs return their
      // traces from it, so it stays until then)
      const params = {
        ...(opts.workers ? { workers: opts.workers } : {}),
        ...(opts.tileCrc ? { tile_crc: true } : {}),
      };
      const msg: ToWorker = { avis, inputs, out, params, keepStore: opts.keepStore, scratch: "e2e-stream" };
      worker.postMessage(msg);
    });
  } finally {
    worker.terminate();
  }
}

/** `len` bytes of output `name` from `start`, base64 (empty past the end). */
export function readOutput(name: string, start: number, len: number): Promise<string> {
  const blob = outputs.get(name);
  if (!blob) return Promise.reject(new Error(`no output ${name}`));
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] ?? "");
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob.slice(start, start + len));
  });
}
