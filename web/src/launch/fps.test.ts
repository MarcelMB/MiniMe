import { describe, expect, it } from "vitest";
import { fpsOverride } from "./fps";

describe("fpsOverride", () => {
  it("keeps the detected frame rate unless the user typed another", () => {
    expect(fpsOverride("", 20)).toEqual({ fps: null, error: "" });
    expect(fpsOverride("20", 20)).toEqual({ fps: null, error: "" });
    expect(fpsOverride(" 30 ", 20)).toEqual({ fps: 30, error: "" });
    expect(fpsOverride("19.8", null)).toEqual({ fps: 19.8, error: "" });
  });

  it("rejects values outside (0, 1000] Hz", () => {
    for (const t of ["0", "-5", "2000", "abc"])
      expect(fpsOverride(t, null).error).not.toBe("");
  });
});
