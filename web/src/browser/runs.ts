import type { RunListing, Summary } from "../types";
import { type Dir, exists, fileAt, readJSON } from "./dir";
import type { Folders } from "./folders";
import { fpsFromFolder } from "./fps";
import type { KV } from "./kv";
import defaults from "./params.defaults.json";
import { HttpError } from "./router";
import { joinVpath, parentVpath, splitVpath } from "./vpath";

const MARKERS = ["params.json", "report.json"];
const SKIP = new Set(["Library"]);

export type RunInfo = {
  id: string;
  vpath: string;
  dir: Dir;
  params: Record<string, unknown>;
  // biome-ignore lint/suspicious/noExplicitAny: report.json is free-form (main reads it the same way)
  report: Record<string, any>;
  meta: { H: number; W: number; T: number; tile: number; ranks: number[]; windows?: { t0: number; t1: number }[] };
  mtime: number;
};

const isRun = async (d: Dir) => (await Promise.all(MARKERS.map((m) => exists(d, m)))).every(Boolean);
const norm = (v: unknown) => JSON.stringify(v);
/** A raw folder written by a browser run is a vpath; a Python run stores a disk path. */
const isDiskPath = (p: string) => p.startsWith("/") || /^[A-Za-z]:[\\/]/.test(p);

/** Runs by vpath and id; ids are random and kept in `ids` (vpath → id). */
export class Registry {
  constructor(
    private folders: Folders,
    private ids: KV,
  ) {}

  /** Run folders at or below `root` (never inside a run), newest report first. */
  async find(root: string, maxDepth = 4): Promise<string[]> {
    const found: { v: string; t: number }[] = [];
    const walk = async (d: Dir, v: string, depth: number): Promise<void> => {
      if (await isRun(d)) {
        found.push({ v, t: (await fileAt(d, "report.json")).lastModified });
        return;
      }
      if (depth === maxDepth) return;
      for (const e of await d.entries())
        if (e.kind === "directory" && !e.name.startsWith(".") && !SKIP.has(e.name))
          await walk(await d.dir(e.name), joinVpath(v, e.name), depth + 1).catch(() => undefined);
    };
    await walk(await this.folders.resolve(root), root, 0);
    return found.sort((a, b) => b.t - a.t).map((x) => x.v);
  }

  async open(vpath: string): Promise<RunInfo> {
    const dir = await this.folders.resolve(vpath);
    if (!(await isRun(dir))) throw new HttpError(404, `not a minime run: ${vpath}`);
    let id = await this.ids.get<string>(vpath);
    if (!id) {
      id = [...crypto.getRandomValues(new Uint8Array(6))].map((b) => b.toString(16).padStart(2, "0")).join("");
      await this.ids.set(vpath, id);
      await this.ids.set(`id:${id}`, vpath);
    }
    const meta = await readJSON<RunInfo["meta"]>(dir, "factors/meta.json");
    return {
      id,
      vpath,
      dir,
      params: await readJSON(dir, "params.json"),
      report: await readJSON(dir, "report.json"),
      meta: { ...meta, H: +meta.H, W: +meta.W, T: +meta.T, tile: +meta.tile, ranks: meta.ranks.map(Number) },
      mtime: (await fileAt(dir, "report.json")).lastModified / 1e3,
    };
  }

  async get(id: string): Promise<RunInfo> {
    const vpath = await this.ids.get<string>(`id:${id}`);
    if (!vpath) throw new HttpError(404, `unknown run ${id}`);
    return this.open(vpath);
  }

  /** `RunView.fps`: given at the start, else from the raw folder's files, else set in the viewer. */
  async fps(r: RunInfo): Promise<number | null> {
    if (r.params.fps) return Number(r.params.fps);
    const raw = r.params.raw_dir as string | undefined;
    if (raw && !isDiskPath(raw)) {
      const parent = parentVpath(raw);
      const f = await fpsFromFolder(
        await this.folders.resolve(raw).catch(() => null),
        parent ? await this.folders.resolve(parent).catch(() => null) : null,
      );
      if (f) return f;
    }
    return (await this.ids.get<number>(`fps:${r.id}`)) ?? null;
  }

  async listing(r: RunInfo): Promise<RunListing> {
    return {
      id: r.id,
      name: splitVpath(r.vpath).rel.split("/").pop() || r.vpath,
      path: r.vpath,
      units: r.report.counts?.units_final ?? null,
      T: r.meta.T,
      fps: await this.fps(r),
      wall_total: r.report.wall_total ?? null,
      mtime: r.mtime,
    };
  }

  async summary(r: RunInfo): Promise<Summary> {
    const changed: Summary["params_changed"] = {};
    for (const p of defaults.params)
      if (p.name in r.params && norm(r.params[p.name]) !== norm(p.default))
        changed[p.name] = { value: r.params[p.name], default: p.default };
    const has = (rel: string) => exists(r.dir, rel);
    return {
      ...(await this.listing(r)),
      H: r.meta.H,
      W: r.meta.W,
      tile: r.meta.tile,
      ranks: r.meta.ranks,
      raw_dir: (r.params.raw_dir as string | undefined) ?? null,
      params: r.params,
      params_changed: changed,
      stages: r.report.stages,
      counts: r.report.counts,
      peak_rss: r.report.peak_rss ?? null,
      peak_rss_children: r.report.peak_rss_children ?? null,
      bytes: r.report.bytes ?? {},
      workers: r.report.workers ?? null,
      motion: r.report.motion ?? {},
      available: {
        raw: await this.rawDir(r).then(
          () => true,
          () => false,
        ),
        pre: (await has("tiles/tile_000.part00")) || (await has("tiles/w00/tile_000.part00")),
        seeds_saved: await has("seeds/merged.npy"),
        init_saved: await has("init/A.npz"),
      },
    };
  }

  /** The run's raw-video folder (`RunView.has_raw`: it must hold `.avi` files): `params.raw_dir`
   * as a vpath (browser runs) — Python runs store an absolute disk path, which the browser cannot
   * open. */
  async rawDir(r: RunInfo): Promise<Dir> {
    const raw = r.params.raw_dir as string | undefined;
    if (!raw || isDiskPath(raw)) throw new HttpError(404, "raw videos not found");
    const dir = await this.folders.resolve(raw);
    if (!(await dir.entries()).some((e) => e.kind === "file" && /\.avi$/i.test(e.name)))
      throw new HttpError(404, "raw videos not found");
    return dir;
  }

  /** The frame rate set in the viewer (kept per run, outside the run folder). */
  async setFps(r: RunInfo, fps: number): Promise<void> {
    await this.ids.set(`fps:${r.id}`, fps);
  }
}
