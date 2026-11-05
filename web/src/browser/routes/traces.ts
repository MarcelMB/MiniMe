import { computing } from "../compute";
import { overview, traceWindow, zooms } from "../derived/traces";
import type { readNpy } from "../npy";
import { HttpError, type Reply, router } from "../router";
import type { RunInfo } from "../runs";
import { state } from "../state";
import { traceRow, tracesShape } from "../tracefile";
import { runOf } from "./tune";

type Kind = "C" | "YrA";

const kindOf = (k: string | null): Kind => {
  const kind = k ?? "C";
  if (kind !== "C" && kind !== "YrA") throw new HttpError(400, `kind must be C or YrA, got '${kind}'`);
  return kind;
};

/** A query number the way FastAPI parses it (`start: int`, `lowpass_hz: float`): else 422. */
const queryNumber = (v: string, name: string, integer: boolean): number => {
  const x = v.trim() === "" ? Number.NaN : Number(v);
  if (!(integer ? Number.isInteger(x) : Number.isFinite(x)))
    throw new HttpError(422, `${name} must be ${integer ? "an integer" : "a number"}`);
  return x;
};

type Cached =
  | { r: RunInfo; ov: ReturnType<typeof readNpy>; reply?: undefined }
  | { r: RunInfo; ov?: undefined; reply: Reply };

/** The run's cached min/max overview, or the "computing" reply that starts it. */
async function cachedOverview(rid: string, rerun: string | null, kind: Kind, bins: number): Promise<Cached> {
  const r = await runOf(rid, rerun); // before the cache: it drops the cache of a rewritten run
  const cache = await state.cache(r.id);
  const name = `overview_${kind}_${bins}.npy`;
  if (await cache.has(name)) return { r, ov: await cache.loadNpy(name) };
  const { K, T } = await tracesShape(r, kind);
  return {
    r,
    reply: computing(`${r.id}:${r.stamp}:${name}`, async () => {
      const o = await overview((i) => traceRow(r, kind, i), K, T, bins);
      await cache.saveNpy(name, [K, 2, Math.max(1, Math.min(bins, T))], o);
    }),
  };
}

router.add("GET", "/runs/:rid/traces/overview", async (q, p) => {
  const kind = kindOf(q.query.get("kind"));
  const bins = queryNumber(q.query.get("bins") ?? "2000", "bins", true);
  if (!(bins >= 1 && bins <= 20000)) throw new HttpError(400, "bins must be in 1..20000");
  const o = await cachedOverview(p.rid, q.query.get("rerun"), kind, bins);
  if (!o.ov) return o.reply;
  return { array: { shape: o.ov.shape, dtype: "float32", data: o.ov.data as Float32Array } };
});

router.add("GET", "/runs/:rid/traces/:unit", async (q, p) => {
  const r = await runOf(p.rid, q.query.get("rerun"));
  const kind = kindOf(q.query.get("kind"));
  const row = await traceRow(r, kind, queryNumber(p.unit, "unit", true));
  const start = queryNumber(q.query.get("start") ?? "0", "start", true);
  const stop = q.query.has("stop") ? queryNumber(q.query.get("stop") as string, "stop", true) : row.length;
  const lp = q.query.has("lowpass_hz") ? queryNumber(q.query.get("lowpass_hz") as string, "lowpass_hz", false) : null;
  const y = traceWindow(row, start, stop, await state.registry.fps(r), lp);
  return { array: { shape: [y.length], dtype: "float32", data: y } };
});

router.add("GET", "/runs/:rid/zooms", async (q, p) => {
  const o = await cachedOverview(p.rid, q.query.get("rerun"), "C", 2000);
  if (!o.ov) return o.reply;
  const [n, , b] = o.ov.shape;
  const peak = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < b; j++) m = Math.max(m, o.ov.data[i * 2 * b + b + j] as number);
    peak[i] = m;
  }
  return { json: zooms(peak, o.r.meta.T, await state.registry.fps(o.r)) };
});
