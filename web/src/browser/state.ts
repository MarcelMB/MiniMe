import { RunCache } from "./cache";
import { type Dir, HandleDir } from "./dir";
import { Folders } from "./folders";
import { idbKV } from "./kv";
import { HttpError } from "./router";
import { Registry } from "./runs";

const opfsDir = async (name: string): Promise<Dir | null> => {
  if (!navigator.storage?.getDirectory) return null;
  const root = await navigator.storage.getDirectory();
  return new HandleDir(await root.getDirectoryHandle(name, { create: true }));
};

const overrides: { folders?: Folders; registry?: Registry } = {};
let cacheRoot: () => Promise<Dir> = async () => {
  const d = await opfsDir("cache");
  if (!d) throw new HttpError(500, "this browser has no private storage (OPFS) for the run cache");
  return d;
};

const lazy = <T>(make: () => T) => {
  let v: T | null = null;
  return () => (v ??= make());
};
const folders = lazy(() => new Folders(idbKV("folders"), () => opfsDir("runs")));
const registry = lazy(() => new Registry(state.folders, idbKV("runs"), () => cacheRoot()));

/** The back end's singletons; tests replace them with `configure`. */
export const state = {
  get folders(): Folders {
    return overrides.folders ?? folders();
  },
  get registry(): Registry {
    return overrides.registry ?? registry();
  },
  /** The run's derived views (outside the run folder, in the cache root). */
  async cache(id: string): Promise<RunCache> {
    return new RunCache(await (await cacheRoot()).dir(id, { create: true }));
  },
};

export function configure(s: { folders?: Folders; registry?: Registry; cacheRoot?: Dir }): void {
  if (s.folders) overrides.folders = s.folders;
  if (s.registry) overrides.registry = s.registry;
  if (s.cacheRoot) {
    const d = s.cacheRoot;
    cacheRoot = async () => d;
  }
}
