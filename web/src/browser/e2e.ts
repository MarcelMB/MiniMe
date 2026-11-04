// Hooks for Playwright (`e2e/browser.spec.ts`, `look.spec.ts`, `stream.spec.ts`), which cannot
// drive the system folder dialog or reach main's `api.ts` in the built page. They exist only in
// the browser build opened with `?e2e`; nothing else in the app reads them.
import { HandleDir } from "./dir";
import { readOutput, runStream } from "./e2eStream";
import { browserFetch } from "./fetch";
import { state } from "./state";

declare global {
  interface Window {
    /** Copy `files` into OPFS `e2e/<name>/raw/` and remember `e2e/<name>` as a picked folder;
     * returns its root name (a vpath starts with it). */
    __minimeAddFolder?: (name: string, files: File[]) => Promise<string>;
    /** Like `__minimeAddFolder` with the files at their own relative paths (`[path, file]`) under
     * `e2e/<name>/`: a run folder and its raw videos side by side. `raw` (a path under the folder)
     * is remembered as its last run's raw folder, which New run starts from. */
    __minimeAddTree?: (name: string, files: [string, File][], raw?: string) => Promise<string>;
    /** The streaming pipeline in a new worker (`e2eStream.ts`), and a slice of one of its outputs. */
    __minimeStream?: typeof runStream;
    __minimeStreamRead?: typeof readOutput;
    /** `api.ts`'s `getJSON` and `postJSON` over the in-page router, and `getArray` for an array
     * answer (asked again while it is computing). */
    __minimeApi?: {
      getJSON: (path: string) => Promise<unknown>;
      postJSON: (path: string, body: unknown) => Promise<unknown>;
      getArray: (path: string) => Promise<{ shape: number[]; dtype: string; data: number[] }>;
    };
  }
}

async function addTree(name: string, files: [string, File][], raw?: string): Promise<string> {
  const opfs = await navigator.storage.getDirectory();
  const e2e = await opfs.getDirectoryHandle("e2e", { create: true });
  await e2e.removeEntry(name, { recursive: true }).catch(() => undefined);
  const top = await e2e.getDirectoryHandle(name, { create: true });
  for (const [path, file] of files) {
    const parts = path.split("/");
    let dir = top;
    for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create: true });
    const w = await (await dir.getFileHandle(parts[parts.length - 1], { create: true })).createWritable();
    await w.write(file);
    await w.close();
  }
  const root = await state.folders.add(new HandleDir(top), "picked");
  if (raw) await state.folders.touch(root, `${root}/${raw}`);
  return root;
}
const addFolder = (name: string, files: File[]) =>
  addTree(
    name,
    files.map((f) => [`raw/${f.name}`, f]),
  );

async function call(path: string, init?: RequestInit): Promise<unknown> {
  const r = await browserFetch(`/api${path}`, init);
  const doc = await r.json();
  if (!r.ok) throw new Error(String(doc.detail));
  return doc;
}
async function getArray(path: string): Promise<{ shape: number[]; dtype: string; data: number[] }> {
  for (;;) {
    const r = await browserFetch(`/api${path}`);
    if (r.status === 202) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      continue;
    }
    if (!r.ok) throw new Error(String((await r.json()).detail));
    const dtype = r.headers.get("X-Dtype") ?? "";
    const buf = await r.arrayBuffer();
    const data =
      dtype === "uint8" ? new Uint8Array(buf) : dtype === "int32" ? new Int32Array(buf) : new Float32Array(buf);
    return { shape: (r.headers.get("X-Shape") ?? "").split(",").map(Number), dtype, data: Array.from(data) };
  }
}
const getJSON = (path: string) => call(path);
const postJSON = (path: string, body: unknown) => call(path, { method: "POST", body: JSON.stringify(body) });

if (import.meta.env.MODE === "browser" && new URLSearchParams(location.search).has("e2e")) {
  window.__minimeAddFolder = addFolder;
  window.__minimeAddTree = addTree;
  window.__minimeStream = runStream;
  window.__minimeStreamRead = readOutput;
  window.__minimeApi = { getJSON, postJSON, getArray };
}
