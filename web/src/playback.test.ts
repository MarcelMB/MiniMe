import { describe, expect, it } from "vitest";
import { advance } from "./playback";

describe("advance", () => {
  it("moves by speed x fps x dt and keeps the fraction", () => {
    expect(advance(0, 1000, 1, 20, 100, 0)).toEqual({ t: 2, carry: 0, ended: false });
    const a = advance(0, 1000, 1, 20, 30, 0); // 0.6 frames
    expect(a.t).toBe(0);
    expect(advance(a.t, 1000, 1, 20, 30, a.carry).t).toBe(1);
    expect(advance(0, 1000, 16, 20, 100, 0).t).toBe(32);
  });
  it("stops at the last frame", () => {
    expect(advance(995, 1000, 16, 20, 100, 0)).toEqual({ t: 999, carry: 0, ended: true });
  });
});
