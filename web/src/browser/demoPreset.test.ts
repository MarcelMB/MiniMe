import { describe, expect, it } from "vitest";
import { changedOnly } from "../launch/params";
import { DEMO_PRESET, DEMO_VALUES } from "./demoPreset";
import defaults from "./params.defaults.json";

const KEYS = [
  "seeds_max_wnd",
  "merge_dist",
  "inita_wnd",
  "loop_merge_corr",
  "merge_min_dist",
  "dl_wnd",
  "sparse_snr",
  "size_thres",
];

describe("demo preset", () => {
  it("differs from the defaults in exactly its eight auto-tuned parameters", () => {
    const merged: Record<string, unknown> = {};
    for (const m of defaults.params) merged[m.name] = m.default;
    Object.assign(merged, DEMO_PRESET.values);
    const changed = changedOnly(merged, defaults.params);
    expect(Object.keys(changed).sort()).toEqual([...KEYS].sort());
    expect(changed).toEqual(DEMO_VALUES);
  });

  it("names only parameters the defaults have", () => {
    const names = new Set(defaults.params.map((m) => m.name));
    for (const k of Object.keys(DEMO_VALUES)) expect(names.has(k), k).toBe(true);
  });
});
