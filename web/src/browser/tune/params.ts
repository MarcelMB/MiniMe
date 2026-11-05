import { DEFAULTS, SESSION_PARAMS } from "../../stream/session";
import { HttpError } from "../router";

/** `window2.FACTOR_PARAMS` (what shaped the saved factors) plus the two seed-window lengths, which
 * fixed pass 1's block edges: a change needs a full run. */
export const FACTOR_PARAMS: readonly string[] = [
  "median_ksize",
  "tophat_wnd",
  "motion_max_shift",
  "motion_down",
  "template_frames",
  "motion_before_tophat",
  "motion_bandpass",
  "template_source",
  "template_chunks",
  "motion_fill",
  "tile",
  "svd_k0",
  "svd_power_iter",
  "rank_rho_min",
  "rank_z",
  "noise_frames",
  "noise_freq",
  "seeds_wnd",
  "seeds_stp",
];
export const WINDOW2_PARAMS: readonly string[] = SESSION_PARAMS.filter((k) => !FACTOR_PARAMS.includes(k));
export const SEARCHED = [
  "seeds_diff_thres",
  "pnr_thres",
  "merge_corr",
  "init_merge_corr",
  "inita_corr",
  "sparse_snr",
  "loop_merge_corr",
] as const;
export type SearchedKey = (typeof SEARCHED)[number];
export const CORR_KEYS: readonly string[] = ["merge_corr", "init_merge_corr", "inita_corr", "loop_merge_corr"];
export const DERIVED = ["seeds_max_wnd", "inita_wnd", "dl_wnd", "merge_dist", "merge_min_dist", "size_thres"] as const;
export const MIN_DIST_FRAC = 0.4;
export const TUNE_FRAMES = 40000;
export const TUNE_OPTIONS = ["diameter", "dup_corr", "min_dist_frac", "tune_frames"] as const;
export type Params = Record<string, unknown>;
export type TuneArgs = { diameter?: number; dup_corr?: number; min_dist_frac?: number; tune_frames?: number };

export const norm = (v: unknown): string => JSON.stringify(v);

export function checkWindow2(changed: Params): void {
  for (const k of Object.keys(changed)) {
    if (FACTOR_PARAMS.includes(k)) throw new HttpError(400, `${k}: not a window-2 parameter (needs a full run)`);
    if (!SESSION_PARAMS.includes(k)) throw new HttpError(400, `${k}: not used by the browser app`);
  }
}

/** The size parameters still at their built-in default, i.e. the ones derived from the diameter. */
export function derivedKeys(base: Params, _diameter: number): string[] {
  return DERIVED.filter((k) => norm(base[k]) === norm((DEFAULTS as Params)[k]));
}

/** `base` with the size parameters it left at their defaults set from `diameter` (px). */
export function derivedParams(base: Params, diameter: number, minDistFrac = MIN_DIST_FRAC): Params {
  const d = Math.max(1, Math.round(diameter));
  const area = Math.PI * (diameter / 2) ** 2;
  const lo = Math.max(1, Math.round(0.25 * area));
  const fresh: Params = {
    seeds_max_wnd: d,
    inita_wnd: d,
    dl_wnd: d,
    merge_dist: d,
    merge_min_dist: Math.round(minDistFrac * diameter * 10) / 10,
    size_thres: [lo, null],
  };
  const out = { ...base };
  for (const k of derivedKeys(base, diameter)) out[k] = fresh[k];
  return out;
}

/** `jobs._tune_args`: the options given (blank or null = the default), checked as numbers. */
export function tuneArgs(raw: Record<string, unknown>): TuneArgs {
  const bad = Object.keys(raw)
    .filter((k) => !(TUNE_OPTIONS as readonly string[]).includes(k))
    .sort();
  if (bad.length) throw new HttpError(400, `unknown auto-tune option: ${bad.join(", ")}`);
  const out: TuneArgs = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v === null || v === undefined || v === "") continue;
    const x = typeof v === "number" ? v : Number(String(v).trim());
    if (!Number.isFinite(x) || String(v).trim() === "") throw new HttpError(400, `${k}: enter a number`);
    out[k as keyof TuneArgs] = k === "tune_frames" ? Math.trunc(x) : x;
  }
  return out;
}
