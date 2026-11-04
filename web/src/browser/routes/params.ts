import { pyFloat, pyRound } from "../num";
import defaults from "../params.defaults.json";
import { HttpError, router } from "../router";

const MIN_DIST_FRAC = 0.4; // src/minime/autotune/sweep.py

/** `routes_jobs.params_derived` with `derived_params(Params(), diameter)` (sweep.py): every
 * derived key is at its default, so all are set. Rounding is Python's builtin `round`. */
export function derivedParams(diameter: number): Record<string, unknown> {
  if (!(diameter > 0)) throw new HttpError(400, `diameter must be a number of px > 0, got ${pyFloat(diameter)}`);
  if (!Number.isFinite(diameter)) throw new HttpError(400, "diameter must be finite, got inf");
  const d = Math.max(1, pyRound(diameter));
  const area = Math.PI * (diameter / 2) ** 2;
  return {
    tophat_wnd: d,
    seeds_max_wnd: d,
    inita_wnd: d,
    dl_wnd: d,
    merge_dist: d,
    merge_min_dist: pyRound(MIN_DIST_FRAC * diameter, 1),
    size_thres: [Math.max(1, pyRound(0.25 * area)), null],
  };
}

/** A query value as Python's `float()` reads it (FastAPI's float parameter), null if it is not one. */
function parseFloatParam(raw: string | null): number | null {
  const s = raw?.trim().toLowerCase() ?? "";
  if (/^[+-]?nan$/.test(s)) return Number.NaN;
  if (/^[+-]?inf(inity)?$/.test(s)) return s.startsWith("-") ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  return /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(s) ? Number(s) : null;
}

router.add("GET", "/params/defaults", () => ({ json: defaults }));
router.add("GET", "/params/derived", (q) => {
  const diameter = parseFloatParam(q.query.get("diameter"));
  if (diameter === null) throw new HttpError(422, "diameter must be a number");
  return { json: { ...derivedParams(diameter) } };
});
