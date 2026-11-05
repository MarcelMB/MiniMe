import { describe, expect, it } from "vitest";
import { DEFAULTS } from "../../stream/session";
import { checkWindow2, derivedKeys, derivedParams, FACTOR_PARAMS, norm, tuneArgs, WINDOW2_PARAMS } from "./params";

describe("window-2 parameters", () => {
  it("splits the session parameters into factor and window-2 ones", () => {
    expect(WINDOW2_PARAMS).toContain("sparse_snr");
    expect(WINDOW2_PARAMS).not.toContain("tile");
    expect(FACTOR_PARAMS).toContain("noise_freq");
    expect(new Set([...WINDOW2_PARAMS, ...FACTOR_PARAMS])).toEqual(new Set(Object.keys(DEFAULTS)));
  });
  it("refuses factor and unknown parameters with main's words", () => {
    expect(() => checkWindow2({ tophat_wnd: 5 })).toThrow("tophat_wnd: not a window-2 parameter (needs a full run)");
    expect(() => checkWindow2({ bogus: 1 })).toThrow("bogus: not used by the browser app");
    expect(() => checkWindow2({ sparse_snr: 0.5 })).not.toThrow();
  });
  it("derives the default sizes from the diameter and keeps tuned ones", () => {
    const base = { ...DEFAULTS, merge_dist: 3, size_thres: [10, null] };
    const p = derivedParams(base, 8.0);
    expect([p.merge_dist, p.size_thres]).toEqual([3, [10, null]]);
    expect([p.seeds_max_wnd, p.inita_wnd, p.dl_wnd]).toEqual([8, 8, 8]);
    expect(derivedKeys(base, 8.0)).toEqual(["seeds_max_wnd", "inita_wnd", "dl_wnd", "merge_min_dist"]);
    expect(p.merge_min_dist).toBe(3.2);
    expect(derivedParams(DEFAULTS, 12.4).size_thres).toEqual([30, null]); // round(0.25 * pi * 6.2^2) = 30
  });
  it("parses the auto-tune options as jobs._tune_args does", () => {
    expect(tuneArgs({ diameter: "7", tune_frames: null, dup_corr: "" })).toEqual({ diameter: 7 });
    expect(tuneArgs({ tune_frames: "40000.0" })).toEqual({ tune_frames: 40000 });
    expect(() => tuneArgs({ grid: 1 })).toThrow("unknown auto-tune option: grid");
    expect(() => tuneArgs({ diameter: "seven" })).toThrow("diameter: enter a number");
  });
  it("norm makes 2 and 2.0 one value", () => {
    expect(norm(2)).toBe(norm(2.0));
    expect(norm("auto")).not.toBe(norm(2));
  });
});
