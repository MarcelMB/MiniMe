// Hooks for Playwright (`e2e/browser.spec.ts`), which cannot drive the system folder dialog or
// reach main's `api.ts` in the built page. They exist only in the browser build opened with
// `?e2e`; nothing else in the app reads them.
import { HandleDir } from "./dir";
import { browserFetch } from "./fetch";
import { state } from "./state";

declare global {
  interface Window {
    /** Copy `files` into OPFS `e2e/<name>/raw/` and remember `e2e/<name>` as a picked folder;
     * returns its root name (a vpath starts with it). */
    __minimeAddFolder?: (name: string, files: File[]) => Promise<string>;
    /** `api.ts`'s `getJSON` and `postJSON` over the in-page router, and `getArray` for an array
     * answer (asked again while it is computing). */
    __minimeApi?: {
      getJSON: (path: string) => Promise<unknown>;
      postJSON: (path: string, body: unknown) => Promise<unknown>;
      getArray: (path: string) => Promise<{ shape: number[]; dtype: string; data: number[] }>;
    };
  }
}

async function addFolder(name: string, files: File[]): Promise<string> {
  const opfs = await navigator.storage.getDirectory();
  const e2e = await opfs.getDirectoryHandle("e2e", { create: true });
  await e2e.removeEntry(name, { recursive: true }).catch(() => undefined);
  const top = await e2e.getDirectoryHandle(name, { create: true });
  const raw = await top.getDirectoryHandle("raw", { create: true });
  for (const f of files) {
    const w = await (await raw.getFileHandle(f.name, { create: true })).createWritable();
    await w.write(f);
    await w.close();
  }
  return state.folders.add(new HandleDir(top), "picked");
}

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
  window.__minimeApi = { getJSON, postJSON, getArray };
}
