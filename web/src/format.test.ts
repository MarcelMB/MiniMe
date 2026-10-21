import { describe, expect, it } from "vitest";
import { fmtBytes, fmtClock, fmtDuration, fmtInt, stageLabel } from "./format";

describe("format", () => {
  it("formats durations like the run report", () => {
    expect(fmtDuration(42.4)).toBe("42 s");
    expect(fmtDuration(1185)).toBe("19:45");
    expect(fmtDuration(3 * 3600 + 61)).toBe("3:01:01");
  });
  it("formats clocks with and without a frame rate", () => {
    expect(fmtClock(48200, 20)).toBe("0:40:10");
    expect(fmtClock(48200, null)).toBe("frame 48,200");
  });
  it("formats numbers, bytes and stage names", () => {
    expect(fmtInt(216000)).toBe("216,000");
    expect(fmtBytes(13.9e9)).toBe("13.9 GB");
    expect(fmtBytes(512e6)).toBe("512 MB");
    expect(stageLabel("pass1")).toBe("Pass 1 · preprocessing + motion");
    expect(stageLabel("cnmf")).toBe("CNMF");
    expect(stageLabel("pnr_refine")).toBe("pnr refine");
  });
});
