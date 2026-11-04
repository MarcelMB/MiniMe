import { RawVideo } from "../avi";
import { computing } from "../compute";
import { linspaceInt, percentile, toU8 } from "../derived/contrast";
import { tileMap } from "../derived/tiles";
import { centroids, unitPolygons } from "../derived/units";
import { exists, fileAt, readBytes } from "../dir";
import { readNpy } from "../npy";
import { readCsr, readNpz } from "../npz";
import { pyRound, roundHalfEven } from "../num";
import { HttpError, router } from "../router";
import type { RunInfo } from "../runs";
import { state } from "../state";

const f32 = (shape: number[], data: ArrayLike<number>) => ({
  array: { shape, dtype: "float32" as const, data: data instanceof Float32Array ? data : Float32Array.from(data) },
});
/** 404 for a part of a run its folder does not hold (an older or Python-written run). */
const notAvailable = (what: string, file?: string) =>
  new HttpError(404, `${what} is not available for this run${file ? `: ${file}` : ""}`);

const IMAGES = ["max_proj", "background", "rank", "noise"];
/** The images read from a file of the run (`rank` is drawn from `factors/meta.json`). */
const IMAGE_FILES: Record<string, string> = {
  max_proj: "factors/max_proj.npy",
  background: "cnmf/b.npy",
  noise: "factors/sn.npy",
};

router.add("GET", "/runs/:rid/image/:name", async (_q, p) => {
  const r = await state.registry.get(p.rid);
  const { H, W } = r.meta;
  if (!IMAGES.includes(p.name)) throw new HttpError(404, `unknown image '${p.name}' (one of ${IMAGES.join(", ")})`);
  if (p.name === "rank") return f32([H, W], tileMap(H, W, r.meta.tile, r.meta.ranks));
  const file = IMAGE_FILES[p.name];
  if (!(await exists(r.dir, file))) throw notAvailable(p.name, file);
  const a = readNpy(await readBytes(r.dir, file));
  if (a.data.length !== H * W) throw new HttpError(500, `${file} holds ${a.data.length} values, not H * W = ${H * W}`);
  return f32([H, W], a.data);
});

router.add("GET", "/runs/:rid/motion", async (_q, p) => {
  const r = await state.registry.get(p.rid);
  if (!(await exists(r.dir, "motion/arrays.npz"))) throw notAvailable("the motion", "motion/arrays.npz");
  const z = await readNpz(await fileAt(r.dir, "motion/arrays.npz"));
  const [shifts, quality] = [z.get("shifts"), z.get("quality")];
  if (!shifts || !quality) throw notAvailable("the motion", "motion/arrays.npz has no shifts and quality");
  const sh = readNpy(shifts).data;
  const q = readNpy(quality).data;
  const out = new Float32Array(q.length * 3);
  for (let t = 0; t < q.length; t++) out.set([sh[2 * t], sh[2 * t + 1], q[t]], 3 * t);
  return f32([q.length, 3], out);
});

// Task 10 adds the computation of a missing residual; until then only a cached one is answered.
router.add("GET", "/runs/:rid/motion/residual", async (_q, p) => {
  const r = await state.registry.get(p.rid);
  const cache = await state.cache(r.id);
  if (!(await cache.has("residual.npy"))) throw notAvailable("the motion residual");
  const a = await cache.loadNpy("residual.npy");
  return f32(a.shape, a.data);
});

router.add("GET", "/runs/:rid/seeds", async (_q, p) => {
  const r = await state.registry.get(p.rid);
  if (!(await exists(r.dir, "seeds/merged.npy"))) throw notAvailable("the seeds");
  const out: Record<string, number[][]> = {};
  for (const k of ["init", "pnr", "merged"]) {
    const a = readNpy(await readBytes(r.dir, `seeds/${k}.npy`));
    const rows: number[][] = [];
    for (let i = 0; i < a.shape[0]; i++)
      rows.push([roundHalfEven(a.data[2 * i], 1), roundHalfEven(a.data[2 * i + 1], 1)]);
    out[k] = rows;
  }
  return { json: out };
});

router.add("GET", "/runs/:rid/units", async (q, p) => {
  const r = await state.registry.get(p.rid);
  const stage = q.query.get("stage") ?? "final";
  if (stage !== "final" && stage !== "init") throw new HttpError(400, `stage must be final or init, got '${stage}'`);
  const file = stage === "final" ? "cnmf/A.npz" : "init/A.npz";
  if (!(await exists(r.dir, file))) throw notAvailable(stage === "init" ? "the init footprints" : "the units");
  const A = await readCsr(await fileAt(r.dir, file));
  const K = A.shape[0];
  const peak = stage === "final" ? await peaks(r, K) : null;
  const { H, W } = r.meta;
  const cent = centroids(A, H, W);
  const polys = unitPolygons(A, H, W);
  return {
    json: {
      stage,
      units: polys.map((poly, k) => ({
        id: k,
        centroid: [pyRound(cent[2 * k], 2), pyRound(cent[2 * k + 1], 2)],
        size_px: A.indptr[k + 1] - A.indptr[k],
        peak: peak ? peak[k] : null,
        polygons: poly,
      })),
    },
  };
});

/** The `peak` column of `cnmf/units.csv`: one value per unit. */
async function peaks(r: RunInfo, K: number): Promise<number[]> {
  const file = "cnmf/units.csv";
  if (!(await exists(r.dir, file))) throw notAvailable("the unit peaks", file);
  const [head, ...rows] = (await (await fileAt(r.dir, file)).text()).trim().split("\n");
  const col = head.split(",").indexOf("peak");
  if (col < 0 || rows.length < K) throw notAvailable("the unit peaks", `${file} has no peak for every unit`);
  return rows.map((row) => Number(row.split(",")[col]));
}

type Frames = (t: number) => Promise<Float32Array>;
const sources = new Map<string, Promise<Frames>>();

/** Frame `t` of `source` as float32 (before the contrast); Task 10 adds "pre" and "recon". */
export function frameSource(r: RunInfo, source: "raw" | "pre" | "recon"): Promise<Frames> {
  const key = `${r.id}:${r.stamp}:${source}`;
  let s = sources.get(key);
  if (!s) {
    s = open(r, source);
    sources.set(key, s);
    s.catch(() => sources.delete(key)); // a failure is not remembered
  }
  return s;
}

async function open(r: RunInfo, source: string): Promise<Frames> {
  if (source !== "raw") throw notAvailable(`${source} frames`);
  const dir = await state.registry.rawDir(r).catch((e: unknown) => {
    if (e instanceof HttpError && e.status === 403) throw e; // the folder has to be reconnected
    throw new HttpError(404, `raw videos not found in ${String(r.params.raw_dir ?? "None")}`);
  });
  const v = await RawVideo.open(dir);
  if (v.height !== r.meta.H || v.width !== r.meta.W)
    throw new HttpError(
      400,
      `the raw videos' frames are ${v.height}x${v.width}, the run's are ${r.meta.H}x${r.meta.W}`,
    );
  return async (t) => Float32Array.from(await v.frame(t));
}

/** The frame number of `/frame/:t`: an integer, else 422 like FastAPI's `t: int`. */
function frameNumber(text: string): number {
  const t = text.trim() === "" ? Number.NaN : Number(text);
  if (!Number.isInteger(t)) throw new HttpError(422, "t must be an integer");
  return t;
}

router.add("GET", "/runs/:rid/frame/:t", async (q, p) => {
  const r = await state.registry.get(p.rid);
  const source = q.query.get("source") ?? "recon";
  if (source !== "raw" && source !== "pre" && source !== "recon")
    throw new HttpError(400, `unknown source '${source}' (one of raw, pre, recon)`);
  const t = frameNumber(p.t);
  const { H, W, T } = r.meta;
  if (!(t >= 0 && t < T)) throw new HttpError(400, `frame ${t} out of range 0..${T - 1}`);
  const get = await frameSource(r, source);
  const cache = await state.cache(r.id);
  const doc = await cache.loadJSON<Record<string, [number, number]>>("contrast.json", {});
  if (!doc[source])
    return computing(`${r.id}:${r.stamp}:contrast:${source}`, async () => {
      // 1st and 99.9th percentile of <= 50 frames spread over the run
      const ts = linspaceInt(T - 1, Math.min(50, T));
      const all = new Float32Array(ts.length * H * W);
      for (let i = 0; i < ts.length; i++) all.set(await get(ts[i]), i * H * W);
      all.sort();
      const [lo, hi] = [percentile(all, 1), percentile(all, 99.9)];
      const d = await cache.loadJSON<Record<string, [number, number]>>("contrast.json", {});
      d[source] = [lo, hi > lo ? hi : lo + 1];
      await cache.saveJSON("contrast.json", d);
    });
  const [lo, hi] = doc[source];
  return { array: { shape: [H, W], dtype: "uint8", data: toU8(await get(t), lo, hi) } };
});
