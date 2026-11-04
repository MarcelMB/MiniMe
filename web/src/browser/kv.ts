export interface KV {
  get<T>(k: string): Promise<T | undefined>;
  set(k: string, v: unknown): Promise<void>;
  del(k: string): Promise<void>;
  keys(): Promise<string[]>;
}

export function memoryKV(): KV {
  const m = new Map<string, unknown>();
  return {
    get: async <T>(k: string) => m.get(k) as T | undefined,
    set: async (k, v) => void m.set(k, v),
    del: async (k) => void m.delete(k),
    keys: async () => [...m.keys()],
  };
}

/** One object store in IndexedDB database `minime-<store>` (values are structured-cloned, so
 * directory handles survive reloads). */
export function idbKV(store: string): KV {
  const db = new Promise<IDBDatabase>((ok, no) => {
    const r = indexedDB.open(`minime-${store}`, 1);
    r.onupgradeneeded = () => r.result.createObjectStore("kv");
    r.onsuccess = () => ok(r.result);
    r.onerror = () => no(r.error);
  });
  const run = async <T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest): Promise<T> => {
    const s = (await db).transaction("kv", mode).objectStore("kv");
    return new Promise<T>((ok, no) => {
      const r = f(s);
      r.onsuccess = () => ok(r.result as T);
      r.onerror = () => no(r.error);
    });
  };
  return {
    get: (k) => run("readonly", (s) => s.get(k)),
    set: (k, v) => run("readwrite", (s) => s.put(v, k)),
    del: (k) => run("readwrite", (s) => s.delete(k)),
    keys: async () => (await run<IDBValidKey[]>("readonly", (s) => s.getAllKeys())).map(String),
  };
}
