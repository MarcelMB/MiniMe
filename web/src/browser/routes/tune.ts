import { SESSION_PARAMS } from "../../stream/session";
import type { Checks, CurvePoint, Proposal, Rerun } from "../../types";
import { computing } from "../compute";
import { type Dir, exists, fileAt, readJSON } from "../dir";
import { pyFloat } from "../num";
import { HttpError, router } from "../router";
import type { RunInfo } from "../runs";
import { state } from "../state";
import { bestPerValue } from "../tune/doc";
import { FACTOR_PARAMS, MIN_DIST_FRAC, norm } from "../tune/params";
import { scoreRun } from "../tune/score";
import type { Trial } from "../tune/types";

export const AUTO = "auto";

/** The run `rid`, or its window-2 rerun `rerun` (`reruns/<name>`) as a run view (`routes_runs._run`). */
export async function runOf(rid: string, rerun: string | null): Promise<RunInfo> {
  const run = await state.registry.get(rid);
  if (!rerun) return run;
  const vpath = `${run.vpath}/reruns/${rerun}`;
  if (/[/\\]/.test(rerun) || !(await exists(run.dir, `reruns/${rerun}/report.json`)))
    throw new HttpError(404, `run ${rid} has no finished rerun '${rerun}'`);
  return state.registry.open(vpath);
}

/** `window2.WINDOW2_PARAMS` as main lists them in a row: the app's own list plus the two seed-window lengths (shown, not editable). */
const ROW_PARAMS = SESSION_PARAMS.filter((k) => !FACTOR_PARAMS.includes(k) || k === "seeds_wnd" || k === "seeds_stp");

const changed = (run: Record<string, unknown>, rerun: Record<string, unknown>) =>
  Object.fromEntries(
    ROW_PARAMS.filter((k) => k in rerun && norm(rerun[k]) !== norm(run[k])).map((k) => [k, [run[k] ?? null, rerun[k]]]),
  );

async function row(run: RunInfo, d: Dir, name: string): Promise<Rerun> {
  const out: Rerun = { name, autotune: await exists(d, "autotune.json"), status: "running" };
  if (await exists(d, "report.json")) {
    const rep = await readJSON<Record<string, unknown>>(d, "report.json");
    const params = await readJSON<Record<string, unknown>>(d, "params.json");
    out.status = "done";
    out.counts = (rep.counts as Record<string, number>) ?? {};
    out.wall_total = (rep.wall_total as number) ?? null;
    out.params = Object.fromEntries(ROW_PARAMS.filter((k) => k in params).map((k) => [k, params[k]]));
    out.changed = changed(run.params, params) as Rerun["changed"];
  } else if (await exists(d, "error.txt")) {
    out.status = "failed";
    out.error = (await (await fileAt(d, "error.txt")).text()).trim().split("\n").slice(-1);
  }
  return out;
}

/** One row per `reruns/<name>` with a `params.json`, oldest first. */
export async function reruns(run: RunInfo): Promise<Rerun[]> {
  const root = await run.dir.dir("reruns").catch(() => null); // `exists` is for files
  if (!root) return [];
  const dirs = [];
  for (const e of await root.entries()) {
    if (e.kind !== "directory") continue;
    const d = await root.dir(e.name);
    if (!(await exists(d, "params.json"))) continue;
    const stamp = await fileAt(d, "report.json").catch(() => fileAt(d, "params.json"));
    dirs.push({ name: e.name, d, t: stamp.lastModified });
  }
  dirs.sort((a, b) => a.t - b.t);
  const out = [];
  for (const x of dirs) out.push(await row(run, x.d, x.name));
  return out;
}

/** CSV text into rows of fields: a field starting with `"` runs to the closing `"` (`""` is a quote), commas and newlines inside. */
function csvRows(text: string): string[][] {
  const rows: string[][] = [];
  let rec: string[] = [];
  let f = "";
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (c === '"' && f === "") {
      i++;
      for (; i < text.length; i++) {
        if (text[i] !== '"') f += text[i];
        else if (text[i + 1] === '"') {
          f += '"';
          i++;
        } else break;
      }
      i++; // the closing quote
    } else if (c === ",") {
      rec.push(f);
      f = "";
      i++;
    } else if (c === "\n" || c === "\r") {
      rec.push(f);
      rows.push(rec);
      rec = [];
      f = "";
      i += c === "\r" && text[i + 1] === "\n" ? 2 : 1;
    } else {
      f += c;
      i++;
    }
  }
  if (f !== "" || rec.length) {
    rec.push(f);
    rows.push(rec);
  }
  return rows;
}

const SEARCHED_KEYS = [
  "seeds_diff_thres",
  "pnr_thres",
  "merge_corr",
  "init_merge_corr",
  "inita_corr",
  "sparse_snr",
  "loop_merge_corr",
];

/** `sweep.csv` back into trials (the columns `sweepCsv` writes). */
export function readSweep(text: string): Trial[] {
  const [head, ...rows] = csvRows(text).filter((r) => r.length > 1 || r[0] !== "");
  if (!head) return [];
  const num = (v: string) => (v === "" ? null : Number(v));
  const value = (v: string) => (v === "auto" ? v : Number(v));
  return rows.map((line) => {
    const c = Object.fromEntries(head.map((k, i) => [k, line[i] ?? ""]));
    return {
      step: c.step,
      values: Object.fromEntries(SEARCHED_KEYS.map((k) => [k, value(c[k])])),
      counts: {},
      score:
        c.score === ""
          ? null
          : {
              good: Number(c.good),
              flagged: Number(c.flagged),
              lost: Number(c.lost),
              score: Number(c.score),
              low_pnr: Number(c.low_pnr),
              in_pieces: Number(c.in_pieces),
              bad_size: Number(c.bad_size),
              duplicate: Number(c.duplicate),
            },
      size10_median: num(c.size10_median),
      seconds: Number(c.seconds),
      error: c.error || null,
    };
  });
}

/** A column that mixes "auto" with numbers (`pnr_thres`) is read by pandas as text, so main's curves carry `"1.0"`, not 1. */
function pandasObjectColumns(trials: Trial[]): Trial[] {
  const words = SEARCHED_KEYS.filter((k) => trials.some((t) => typeof t.values[k] === "string"));
  return trials.map((t) => {
    const values = { ...t.values };
    for (const k of words) if (typeof values[k] === "number") values[k] = pyFloat(values[k] as number);
    return { ...t, values };
  });
}

/** `autotune.json` of `reruns/auto` plus the sweep curves, or null before an auto-tune. */
export async function proposal(run: RunInfo): Promise<Proposal | null> {
  if (!(await exists(run.dir, `reruns/${AUTO}/autotune.json`))) return null;
  const auto = await (await run.dir.dir("reruns")).dir(AUTO);
  const doc = await readJSON<Proposal>(auto, "autotune.json");
  const trials = pandasObjectColumns(readSweep(await (await fileAt(auto, "sweep.csv")).text()));
  const curves: Record<string, CurvePoint[]> = {};
  for (const k of Object.keys(doc.proposed)) {
    const c = bestPerValue(trials, k);
    if (c.length) curves[k] = c;
  }
  return { ...doc, curves };
}

async function checksOf(run: RunInfo): Promise<{ checks: Checks; minDistFrac: number } | null> {
  const doc = await proposal(run);
  return doc ? { checks: doc.checks, minDistFrac: doc.min_dist_frac ?? MIN_DIST_FRAC } : null;
}

async function hashKey(what: string, ...parts: unknown[]): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(parts));
  const h = await crypto.subtle.digest("SHA-1", bytes);
  return `${what}_${[...new Uint8Array(h)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 10)}.json`;
}

router.add("GET", "/runs/:rid/reruns", async (_q, p) => ({
  json: { reruns: await reruns(await state.registry.get(p.rid)) },
}));

router.add("GET", "/runs/:rid/autotune", async (_q, p) => {
  const doc = await proposal(await state.registry.get(p.rid));
  if (!doc) throw new HttpError(404, "this run has no auto-tune yet");
  return { json: doc };
});

for (const what of ["flags", "pairs"] as const) {
  router.add("GET", `/runs/:rid/${what}`, async (q, p) => {
    const run = await state.registry.get(p.rid);
    const ch = await checksOf(run);
    if (!ch) throw new HttpError(404, "the unit checks need an auto-tune of this run first");
    const r = await runOf(p.rid, q.query.get("rerun"));
    const cache = await state.cache(r.id);
    const flagsName = await hashKey("flags", ch.checks);
    const pairsName = await hashKey("pairs", ch.checks, ch.minDistFrac);
    const name = what === "flags" ? flagsName : pairsName;
    if (await cache.has(name)) return { json: await cache.loadJSON(name, null) };
    return computing(`${r.id}:${r.stamp}:score_${await hashKey("score", ch.checks, ch.minDistFrac)}`, async () => {
      const s = await scoreRun(r, ch.checks, ch.minDistFrac, []);
      await cache.saveJSON(flagsName, s.flags);
      await cache.saveJSON(pairsName, s.pairs);
    });
  });
}
