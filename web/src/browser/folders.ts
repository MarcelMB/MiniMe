import { type Dir, HandleDir, walk } from "./dir";
import type { KV } from "./kv";
import { HttpError } from "./router";
import { splitVpath } from "./vpath";

export type Root = { name: string; dir: Dir; kind: "picked" | "opfs" | "files"; time: number; raw?: string };
export const OPFS_NAME = "Browser storage";

/** The folders the app can see: picked ones (handles kept in `kv` across reloads), file lists
 * (this page load only) and OPFS. A vpath's first segment names its root. */
export class Folders {
  private live = new Map<string, Root>();
  constructor(
    private kv: KV,
    private opfs: () => Promise<Dir | null>,
  ) {}

  private async all(): Promise<Map<string, Root>> {
    const out = new Map<string, Root>();
    for (const k of await this.kv.keys()) {
      const r = await this.kv.get<Omit<Root, "dir"> & { handle: unknown; dir?: Dir }>(k);
      if (r?.dir) out.set(r.name, r as Root);
      else if (r?.handle) out.set(r.name, { ...r, dir: new HandleDir(r.handle as FileSystemDirectoryHandle) });
    }
    for (const [k, v] of this.live) out.set(k, v);
    const o = await this.opfs();
    if (o && !out.has(OPFS_NAME)) out.set(OPFS_NAME, { name: OPFS_NAME, dir: o, kind: "opfs", time: 0 });
    return out;
  }

  async list(): Promise<Root[]> {
    return [...(await this.all()).values()].sort((a, b) => b.time - a.time);
  }

  /** Remember `dir` under a unique name ("x", "x (2)", ...); returns the name. */
  async add(dir: Dir, kind: Root["kind"]): Promise<string> {
    const taken = await this.all();
    let name = dir.name;
    for (let i = 2; taken.has(name); i++) name = `${dir.name} (${i})`;
    const root: Root = { name, dir, kind, time: Date.now() + taken.size };
    if (kind === "picked" && "handle" in dir)
      await this.kv.set(name, { name, kind, time: root.time, handle: (dir as { handle: unknown }).handle });
    else if (kind === "picked")
      await this.kv.set(name, root); // tests: NodeDir kept as is
    else this.live.set(name, root);
    return name;
  }

  /** Mark a root as just used (and the raw folder of its last run). */
  async touch(name: string, raw?: string): Promise<void> {
    const r = (await this.all()).get(name);
    if (!r) return;
    r.time = Date.now();
    if (raw) r.raw = raw;
    if (this.live.has(name) || r.kind === "opfs") this.live.set(name, r);
    else
      await this.kv.set(name, {
        ...r,
        ...("handle" in r.dir ? { handle: (r.dir as { handle: unknown }).handle, dir: undefined } : {}),
      });
  }

  /** The folder `vpath` names. `mode` "readwrite" is for runs that write into it. */
  async resolve(vpath: string, mode: "read" | "readwrite" = "read"): Promise<Dir> {
    const { root, rel } = splitVpath(vpath);
    const r = (await this.all()).get(root);
    if (!r) throw new HttpError(404, `${root} is not a known folder; choose it again under Runs → Change folder`);
    if (r.kind === "files" && mode === "readwrite")
      throw new HttpError(403, `${root} was opened read-only; runs from it go to Browser storage`);
    if (!(await r.dir.permission(mode, false)))
      throw new HttpError(
        403,
        `Reconnect the folder ${root}: pick it again (Runs → Change folder) to give this page access`,
      );
    return walk(r.dir, rel);
  }

  /** Ask for permission again (needs a click). */
  async reconnect(name: string): Promise<boolean> {
    const r = (await this.all()).get(name);
    return r ? r.dir.permission("readwrite", true) : false;
  }

  /** A folder the system dialog just returned (it asked for write access): the root that is the
   * same folder, with its permission asked for again (it may have needed reconnecting), else a
   * new root. */
  async pick(dir: Dir): Promise<string> {
    const same = await this.sameRoot(dir);
    if (same === null) return this.add(dir, "picked");
    await this.reconnect(same);
    return same;
  }

  /** The name of the root that is the same entry as `dir` (handles only), or null. */
  private async sameRoot(dir: Dir): Promise<string | null> {
    if (!(dir instanceof HandleDir)) return null;
    for (const r of (await this.all()).values())
      if (r.dir instanceof HandleDir && (await r.dir.handle.isSameEntry(dir.handle).catch(() => false))) return r.name;
    return null;
  }
}
