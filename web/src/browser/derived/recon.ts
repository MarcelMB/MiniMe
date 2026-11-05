/** Reconstructed and preprocessed frames of a run: the viewer's `recon` and `pre` sources
 * (main's `frames.recon_frame` and `frames.store_frame`), read from the folder of a browser run
 * (the factors in `factors/wNN/`, the tile store in `tiles/wNN/`, both per window with the
 * window's own frame numbers) and, for the preprocessed frames, from a Python run's flat tile
 * store (`tiles/`, global frame numbers, any number of part files). */
import { type Dir, exists, fileAt, walk } from "../dir";
import { readNpy } from "../npy";
import { HttpError } from "../router";
import type { RunInfo } from "../runs";
import { windowDir } from "../writer";
import { type Tile, tileGrid } from "./tiles";

const notAvailable = (what: string, file: string) =>
  new HttpError(404, `${what} is not available for this run: ${file}`);

/** Windows kept parsed in memory: the one a frame is in and the one before or after it. A
 * window's factors are tens of MB at full size; frames of the same window follow each other. */
const KEEP = 2;
/** A small LRU over a map of promises: a hit moves the entry last, an insert past `KEEP` drops the first. */
function lru<T>(cache: Map<string, Promise<T>>, key: string, make: () => Promise<T>): Promise<T> {
  let p = cache.get(key);
  if (p) cache.delete(key);
  else {
    p = make();
    p.catch(() => cache.delete(key)); // a failure is not remembered
  }
  cache.set(key, p);
  for (const old of cache.keys()) {
    if (cache.size <= KEEP) break;
    cache.delete(old);
  }
  return p;
}

type Windows = NonNullable<RunInfo["meta"]["windows"]>;
const windowsOf = (meta: RunInfo["meta"]): Windows => meta.windows ?? [{ t0: 0, t1: meta.T }];

/** The window frame `t` of the session is in, and its place in that window (`factors/meta.json`'s
 * `windows`; a run without them is one window). */
export function windowOf(meta: RunInfo["meta"], t: number): { w: number; local: number } {
  const windows = windowsOf(meta);
  const w = windows.findIndex((x) => t >= x.t0 && t < x.t1);
  if (w < 0) throw new HttpError(400, `frame ${t} is in no window`);
  return { w, local: t - windows[w].t0 };
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/** One window's factors, flattened over its tiles in grid order (`compress::outputs`): `W` is
 * (P, r) per tile and already scaled by the singular values, `V` (T, r) with T the window's
 * frames, `mean` (P); the `*At` arrays are each tile's start in them. */
type Factors = {
  W: ArrayLike<number>;
  V: ArrayLike<number>;
  mean: ArrayLike<number>;
  ranks: number[];
  wAt: number[];
  vAt: number[];
  mAt: number[];
};

const factorCache = new Map<string, Promise<Factors>>();
const FACTOR_FILES = ["W.npy", "V.npy", "mean.npy", "rank.npy"];

const factorDir = (w: number) => `factors/${windowDir(w)}`;

/** Whether the run folder holds the factors a reconstruction needs (a Python run's are not in this format). */
export const hasRecon = (r: RunInfo): Promise<boolean> => exists(r.factors, `${factorDir(0)}/W.npy`);

function factors(r: RunInfo, w: number): Promise<Factors> {
  return lru(factorCache, `${r.id}:${r.stamp}:${w}`, () => loadFactors(r, w));
}

async function loadFactors(r: RunInfo, w: number): Promise<Factors> {
  const [d, win] = [factorDir(w), windowsOf(r.meta)[w]];
  const files: ArrayLike<number>[] = [];
  for (const name of FACTOR_FILES) {
    if (!(await exists(r.factors, `${d}/${name}`))) throw notAvailable("recon frames", `${d}/${name}`);
    files.push(readNpy(new Uint8Array(await (await fileAt(r.factors, `${d}/${name}`)).arrayBuffer())).data);
  }
  const [W, V, mean, rank] = files;
  const { H, W: width, tile } = r.meta;
  const grid = tileGrid(H, width, tile);
  if (rank.length !== grid.length)
    throw new HttpError(
      500,
      `${d}/rank.npy holds ${rank.length} tiles, the run's ${H} x ${width} frames in ${tile}-px tiles have ${grid.length}`,
    );
  const T = win.t1 - win.t0;
  const [ranks, wAt, vAt, mAt] = [Array.from(rank), [0], [0], [0]];
  grid.forEach((g, b) => {
    const P = (g.y1 - g.y0) * (g.x1 - g.x0);
    wAt.push(wAt[b] + P * ranks[b]);
    vAt.push(vAt[b] + T * ranks[b]);
    mAt.push(mAt[b] + P);
  });
  const [nw, nv, nm] = [wAt[grid.length], vAt[grid.length], mAt[grid.length]];
  if (W.length !== nw || V.length !== nv || mean.length !== nm)
    throw new HttpError(
      500,
      `${d} holds ${W.length} W, ${V.length} V and ${mean.length} mean values, its ranks and ${T} frames need ${nw}, ${nv} and ${nm}`,
    );
  return { W, V, mean, ranks, wAt, vAt, mAt };
}

/** Frame `t` rebuilt from the factors of its window: per tile `mean[p] + sum_k W[p, k] V[t, k]`
 * (`factors.reconstruct_chunk`), (H, W) float32. */
export async function reconFrame(r: RunInfo, t: number): Promise<Float32Array> {
  const { w, local } = windowOf(r.meta, t);
  const f = await factors(r, w);
  const { H, W } = r.meta;
  const out = new Float32Array(H * W);
  tileGrid(H, W, r.meta.tile).forEach((g, b) => {
    const [rank, width] = [f.ranks[b], g.x1 - g.x0];
    const v = f.vAt[b] + local * rank; // V is (T, r): the frame's row
    for (let y = g.y0; y < g.y1; y++)
      for (let x = g.x0; x < g.x1; x++) {
        const p = (y - g.y0) * width + (x - g.x0);
        const row = f.wAt[b] + p * rank;
        let acc = f.mean[f.mAt[b] + p];
        for (let k = 0; k < rank; k++) acc += f.W[row + k] * f.V[v + k];
        out[y * W + x] = acc;
      }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Preprocessed frames (the uint8 tile store)
// ---------------------------------------------------------------------------

const pixels = (g: Tile) => (g.y1 - g.y0) * (g.x1 - g.x0);
const tileName = (k: number, part: number) => `tile_${String(k).padStart(3, "0")}.part${String(part).padStart(2, "0")}`;
const FLAT = "tiles";

/** One part file set of a Python run's flat store: a file per tile in `tiles/`, from session frame
 * `first` on, `frames` frames long. */
type Part = { part: number; first: number; frames: number };
/** How a run keeps its tile store: `flat` (a Python run's parts, global frame numbers), or null
 * for a browser run's `tiles/wNN/` per window. */
type Layout = { flat: Part[] | null };
const layouts = new Map<string, Promise<Layout>>();
const handles = new Map<string, Promise<File[]>>();

const firstTile = (r: RunInfo) => tileGrid(r.meta.H, r.meta.W, r.meta.tile)[0];

/** `frames.store_frame`'s part table: the parts of tile 0 in part order, each holding its size /
 * (pixels of tile 0) frames. */
async function flatParts(r: RunInfo): Promise<Part[]> {
  const px = pixels(firstTile(r));
  const numbers = (await (await walk(r.factors, FLAT)).entries())
    .flatMap((e) => (e.kind === "file" ? [/^tile_000\.part(\d+)$/.exec(e.name)] : []))
    .flatMap((m) => (m ? [Number(m[1])] : []))
    .sort((a, b) => a - b);
  const parts: Part[] = [];
  let first = 0;
  for (const part of numbers) {
    const frames = Math.floor((await fileAt(r.factors, `${FLAT}/${tileName(0, part)}`)).size / px);
    parts.push({ part, first, frames });
    first += frames;
  }
  return parts;
}

/** The layout of the run's tile store (a flat `tiles/tile_000.part00` wins over `tiles/w00/`), or
 * 404 when it kept none. */
function layout(r: RunInfo): Promise<Layout> {
  return lru(layouts, `${r.id}:${r.stamp}`, async () => {
    if (await exists(r.factors, `${FLAT}/${tileName(0, 0)}`)) return { flat: await flatParts(r) };
    const first = `${FLAT}/${windowDir(0)}/${tileName(0, 0)}`;
    if (await exists(r.factors, first)) return { flat: null };
    throw notAvailable("pre frames", first);
  });
}

/** Whether the run folder holds a tile store in either layout (`RunView.has_store`; a browser
 * run with `keep_store` has `tiles/wNN/`). */
export const hasPre = (r: RunInfo): Promise<boolean> =>
  layout(r).then(
    () => true,
    () => false,
  );

/** The tile files of one part set in `dir`, opened once (a `File` is a handle: its bytes are read
 * on `slice`). */
function tileFiles(r: RunInfo, dir: string, part: number): Promise<File[]> {
  const n = tileGrid(r.meta.H, r.meta.W, r.meta.tile).length;
  return lru(handles, `${r.id}:${r.stamp}:${dir}:${part}`, async () => {
    try {
      const d: Dir = await walk(r.factors, dir);
      return await Promise.all(Array.from({ length: n }, (_, k) => d.file(tileName(k, part))));
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) throw notAvailable("pre frames", `${dir}/${tileName(0, part)}`);
      throw e;
    }
  });
}

/** Frame `k` of the part set in `dir`: (H, W) float32 from the uint8 tiles, each `(n, th, tw)`
 * row-major. */
async function readFrame(r: RunInfo, dir: string, part: number, k: number): Promise<Float32Array> {
  const { H, W } = r.meta;
  const files = await tileFiles(r, dir, part);
  const frames = Math.floor(files[0].size / pixels(firstTile(r)));
  if (k >= frames) throw new HttpError(500, `${dir} holds ${frames} frames of the tile store, not frame ${k + 1}`);
  const out = new Float32Array(H * W);
  await Promise.all(
    tileGrid(H, W, r.meta.tile).map(async (g, b) => {
      const [wd, px] = [g.x1 - g.x0, pixels(g)];
      const bytes = new Uint8Array(await files[b].slice(k * px, (k + 1) * px).arrayBuffer());
      if (bytes.length !== px) throw new HttpError(500, `${dir}/${tileName(b, part)} is shorter than ${k + 1} frames`);
      for (let y = g.y0; y < g.y1; y++)
        for (let x = g.x0; x < g.x1; x++) out[y * W + x] = bytes[(y - g.y0) * wd + (x - g.x0)];
    }),
  );
  return out;
}

/** Frame `t` of the preprocessed tile store (`frames.store_frame`), (H, W) float32: from a flat
 * `tiles/` (a Python run: global frame numbers over its parts), else `tiles/wNN/` of the frame's
 * window (a browser run: the window's own numbers, one part). */
export async function preFrame(r: RunInfo, t: number): Promise<Float32Array> {
  const { flat } = await layout(r);
  if (flat) {
    const p = flat.find((x) => t >= x.first && t < x.first + x.frames);
    if (!p) {
      const total = flat.reduce((n, x) => n + x.frames, 0);
      throw new HttpError(500, `the tile store holds ${total} frames, not frame ${t + 1}`);
    }
    return readFrame(r, FLAT, p.part, t - p.first);
  }
  const { w, local } = windowOf(r.meta, t);
  return readFrame(r, `${FLAT}/${windowDir(w)}`, 0, local);
}
