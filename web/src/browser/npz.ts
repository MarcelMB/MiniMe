import { readNpy, writeNpy, writeNpyBytes } from "./npy";
import { readZip, zipBlob } from "./zip";

export type Csr = { shape: [number, number]; data: Float32Array; indices: Int32Array; indptr: Int32Array };

export async function readNpz(blob: Blob): Promise<Map<string, Uint8Array>> {
  const out = new Map<string, Uint8Array>();
  for (const [name, b] of await readZip(blob))
    out.set(name.replace(/\.npy$/, ""), new Uint8Array(await b.arrayBuffer()));
  return out;
}

export function npzBlob(arrays: Record<string, Uint8Array>): Promise<Blob> {
  return zipBlob(Object.entries(arrays).map(([k, v]) => ({ name: `${k}.npy`, data: new Blob([v]) })));
}

/** scipy `load_npz` of a CSR matrix; data as float32 (the viewer's type). */
export async function readCsr(blob: Blob): Promise<Csr> {
  const m = await readNpz(blob);
  const get = (k: string) => {
    const b = m.get(k);
    if (!b) throw new Error(`not a scipy sparse .npz (no ${k})`);
    return readNpy(b).data;
  };
  const shape = Array.from(get("shape")) as [number, number];
  return {
    shape,
    data: Float32Array.from(get("data")),
    indices: Int32Array.from(get("indices")),
    indptr: Int32Array.from(get("indptr")),
  };
}

/** scipy `save_npz` layout (stored, not compressed). */
export function csrNpz(a: Csr): Promise<Blob> {
  return npzBlob({
    indices: writeNpy([a.indices.length], a.indices),
    indptr: writeNpy([a.indptr.length], a.indptr),
    format: writeNpyBytes("csr"),
    shape: writeNpy([2], BigInt64Array.from(a.shape.map(BigInt))),
    data: writeNpy([a.data.length], a.data),
  });
}
