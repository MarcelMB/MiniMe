import { type Dir, HandleDir, walk } from "./dir";
import type { KV } from "./kv";
import { HttpError } from "./router";
import { joinVpath, splitVpath } from "./vpath";

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
    // a browser may refuse private storage (a private window): the other folders still answer
    const o = await this.opfs().catch(() => null);
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
    await this.save(r);
  }

  /** Keep `r` as it is now: a picked folder's handle (not the `Dir`) across reloads. */
  private async save(r: Root): Promise<void> {
    if (this.live.has(r.name) || r.kind === "opfs") this.live.set(r.name, r);
    else
      await this.kv.set(r.name, {
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

  /** A folder the system dialog just returned (it asked for write access), as a vpath: the root
   * that is the same folder, now with the handle the dialog gave (it has the permission; the stored
   * one may need reconnecting); a folder inside a root, as its path there (so a raw folder picked
   * in a Runs folder has the parent its run goes to); else a new root. */
  async pick(dir: Dir): Promise<string> {
    const found = await this.within(dir);
    if (found === null) return this.add(dir, "picked");
    const { root, rel } = found;
    if (rel.length) return joinVpath(root.name, ...rel);
    if (root.kind === "picked") await this.save({ ...root, dir });
    await this.reconnect(root.name);
    return root.name;
  }

  /** The root `dir` is (`rel` empty) or lies inside (`rel` the names down to it), the nearest one;
   * handles only (`resolve` and `isSameEntry` say it, the names do not). A root that needs
   * reconnecting is not looked into: a path in it would answer 403. */
  private async within(dir: Dir): Promise<{ root: Root; rel: string[] } | null> {
    if (!(dir instanceof HandleDir)) return null;
    let best: { root: Root; rel: string[] } | null = null;
    for (const r of (await this.all()).values()) {
      if (!(r.dir instanceof HandleDir)) continue;
      if (await r.dir.handle.isSameEntry(dir.handle).catch(() => false)) return { root: r, rel: [] };
      if (!(await r.dir.permission("readwrite", false).catch(() => false))) continue;
      const rel = await r.dir.handle.resolve(dir.handle).catch(() => null);
      if (rel?.length && (best === null || rel.length < best.rel.length)) best = { root: r, rel };
    }
    return best;
  }
}
