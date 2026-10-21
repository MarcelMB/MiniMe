import { describe, expect, it } from "vitest";
import { clampRange, lowpassState, overviewColumns, traceMode } from "./traceMode";

describe("trace mode", () => {
  it("uses the overview for wide views and full traces for narrow windows", () => {
    expect(traceMode(null, 216000)).toBe("overview");
    expect(traceMode([0, 216000], 216000)).toBe("overview");
    expect(traceMode([1000, 21000], 216000)).toBe("window");
    expect(traceMode(null, 3000)).toBe("window");
  });
  it("clamps ranges into the recording", () => {
    expect(clampRange(-50, 120.7, 100)).toEqual([0, 100]);
    expect(clampRange(30, 30, 100)).toEqual([30, 31]);
  });
});

describe("low-pass state", () => {
  it("is disabled without a frame rate", () => {
    const s = lowpassState(null, true, 5);
    expect(s.enabled).toBe(false);
    expect(s.hz).toBeNull();
    expect(s.reason).toMatch(/frame rate/);
  });
  it("clamps the cutoff to Nyquist and passes it only when on", () => {
    expect(lowpassState(20, true, 15)).toMatchObject({ enabled: true, hz: 10, max: 10 });
    expect(lowpassState(20, false, 5).hz).toBeNull();
  });
});

describe("overview columns", () => {
  it("reads min and max rows of one unit", () => {
    const ov = {
      shape: [2, 2, 3],
      dtype: "float32" as const,
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    };
    const c = overviewColumns(ov, 1);
    expect(Array.from(c.lo)).toEqual([6, 7, 8]);
    expect(Array.from(c.hi)).toEqual([9, 10, 11]);
  });
});
