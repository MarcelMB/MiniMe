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
  /** The folder that holds factors/wNN and tiles/: the run's own, or its source run's for a rerun. */
  factors: Dir;
  /** The source run's vpath for a rerun (params.source_run), else null. */
  sourceRun: string | null;
  params: Record<string, unknown>;
  // biome-ignore lint/suspicious/noExplicitAny: report.json is free-form (main reads it the same way)
  report: Record<string, any>;
  meta: { H: number; W: number; T: number; tile: number; ranks: number[]; windows?: { t0: number; t1: number }[] };
  mtime: number;
  /** Version of the run folder (`report.json` lastModified and size, like main's `RunView.stamp`),
   * for computing keys: `${id}:${stamp}:<what>`. */
  stamp: string;
};

const isRun = async (d: Dir) => (await Promise.all(MARKERS.map((m) => exists(d, m)))).every(Boolean);
const norm = (v: unknown) => JSON.stringify(v);
/** A raw folder written by a browser run is a vpath; a Python run stores a disk path. */
const isDiskPath = (p: string) => p.startsWith("/") || /^[A-Za-z]:[\\/]/.test(p);

/** Runs by vpath and id; ids are random and kept in `ids` (vpath → id), with the run's stamp
 * (`stamp:<id>`) and frame rate (`fps:<id>`). `cacheRoot` holds the runs' derived views, one
 * folder per id; without it there is nothing to drop when a run is rewritten. */
export class Registry {
  private stamping = new Map<string, Promise<void>>();

  constructor(
    private folders: Folders,
    private ids: KV,
    private cacheRoot?: () => Promise<Dir>,
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
    const rf = await fileAt(dir, "report.json");
    const stamp = `${rf.lastModified}-${rf.size}`;
    await this.invalidateStale(id, stamp);
    const meta = await readJSON<RunInfo["meta"]>(dir, "factors/meta.json");
    const params = await readJSON<Record<string, unknown>>(dir, "params.json");
    const sourceRun =
      typeof params.source_run === "string" && !isDiskPath(params.source_run) ? params.source_run : null;
    const factors = sourceRun ? await this.folders.resolve(sourceRun).catch(() => dir) : dir;
    return {
      id,
      vpath,
      dir,
      factors,
      sourceRun,
      params,
      report: JSON.parse(await rf.text()),
      meta: { ...meta, H: +meta.H, W: +meta.W, T: +meta.T, tile: +meta.tile, ranks: meta.ranks.map(Number) },
      mtime: rf.lastModified / 1e3,
      stamp,
    };
  }

  /** `RunView._invalidate_stale_cache`: a run folder that was rewritten loses its derived views
   * (the frame rate set in the viewer lives in `ids`, so it stays). The first sight of a run only
   * stores the stamp. One check per run at a time: requests open the run together, and a second
   * one must not delete what the first has begun to compute. */
  private invalidateStale(id: string, stamp: string): Promise<void> {
    const check = (this.stamping.get(id) ?? Promise.resolve()).then(async () => {
      const old = await this.ids.get<string>(`stamp:${id}`);
      if (old === stamp) return;
      if (old !== undefined && this.cacheRoot) await (await this.cacheRoot()).remove(id, { recursive: true });
      await this.ids.set(`stamp:${id}`, stamp);
    });
    this.stamping.set(
      id,
      check.catch(() => undefined),
    );
    return check;
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
      // a browser run's `report.json` has the wasm heap instead (main's summary has no such key,
      // so a Python run's summary stays main's)
      ...(typeof r.report.wasm_heap_peak_mb === "number" ? { peak_wasm: r.report.wasm_heap_peak_mb * 2 ** 20 } : {}),
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
