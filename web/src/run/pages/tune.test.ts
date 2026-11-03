import { describe, expect, it } from "vitest";
import type { Checks } from "../../types";
import { rerunOptions, rerunQuery } from "../RunContext";
import { changedText, criterion, fmtValue } from "./Tune";

const checks: Checks = { diameter: 8, pnr_min: 12, piece_frac: 0.9, size_lo: 0.3, size_hi: 3, dup_corr: 0.8, keep_dist: 0.5 };

describe("tune page helpers", () => {
  it("formats values the way the form shows them", () => {
    expect(fmtValue("auto")).toBe("auto");
    expect(fmtValue([9, null])).toBe("9, ");
    expect(fmtValue(true)).toBe("true");
    expect(fmtValue(0.8)).toBe("0.8");
  });
  it("lists the changed parameters as run→rerun", () => {
    expect(changedText({ sparse_snr: [3.74, 0.96], merge_corr: [0.85, 0.8] })).toBe("sparse_snr 3.74→0.96 · merge_corr 0.85→0.8");
    expect(changedText(undefined)).toBe("");
  });
  it("spells out every flag criterion with the thresholds", () => {
    expect(criterion("low_pnr", checks)).toContain("below 12");
    expect(criterion("duplicate", checks)).toContain("closer than 8 px");
    expect(criterion("in_pieces", checks)).toContain("90 %");
    expect(criterion("bad_size", checks)).toContain("15–151 px");
  });
  it("builds the rerun query and switcher", () => {
    expect(rerunQuery(null)).toBe("");
    expect(rerunQuery("auto")).toBe("?rerun=auto");
    expect(rerunQuery("re run", false)).toBe("&rerun=re%20run");
    const opts = rerunOptions([
      { name: "auto", autotune: true, status: "done" },
      { name: "r2", autotune: false, status: "running" },
    ]);
    expect(opts.map((o) => o.value)).toEqual(["run", "auto", "r2"]);
    expect(opts[2].disabled).toBe(true);
  });
});
