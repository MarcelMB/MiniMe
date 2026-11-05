import type { Preset } from "../launch/preset";

/** Python's auto-tune proposal for the synthetic GRIN demo session (`synth_v4_grin30`): only the
 * parameters it changed; every other one stays at its default. On the 2026-11-04 benchmark this
 * set scored F1 0.68 in Python and 0.76 in the browser. */
export const DEMO_VALUES: Record<string, unknown> = {
  seeds_max_wnd: 12,
  merge_dist: 12,
  inita_wnd: 12,
  loop_merge_corr: 0.75,
  merge_min_dist: 5,
  dl_wnd: 12,
  sparse_snr: 0.714,
  size_thres: [30, null],
};

export const DEMO_PRESET: Preset = {
  name: "Demo settings",
  note: `Python's auto-tune for the demo session: ${Object.keys(DEMO_VALUES).join(", ")}; every other parameter is at its default.`,
  values: DEMO_VALUES,
};
