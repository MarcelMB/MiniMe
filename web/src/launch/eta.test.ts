import { describe, expect, it } from "vitest";
import { Countdown, fmtLeft, runLeft, StageRate } from "./eta";
import type { StageProgress } from "./params";

const p = (
  done: number,
  frames: number,
  seconds: number,
  stage = "motion",
  unit: StageProgress["unit"] = "frames",
) => ({
  stage,
  done,
  frames,
  seconds,
  unit,
});

describe("StageRate", () => {
  it("gives no estimate during the warm-up, then the steady rate", () => {
    const r = new StageRate();
    for (let t = 0.5; t <= 10; t += 0.5) r.feed(p(50 * t, 9000, t));
    expect(r.left()).toBeNull(); // 10 s in: still warming up
    for (let t = 10.5; t <= 30; t += 0.5) r.feed(p(50 * t, 9000, t));
    expect(r.perSecond()).toBeCloseTo(50, 6);
    expect(r.left()).toBeCloseTo((9000 - 1500) / 50, 3);
  });

  it("forgets a slow start (the template phase) within a few time constants", () => {
    const r = new StageRate();
    for (let t = 0.5; t <= 10; t += 0.5) r.feed(p(20 * t, 9000, t)); // 20 fps for 10 s
    for (let t = 10.5; t <= 90; t += 0.5) r.feed(p(200 + 50 * (t - 10), 9000, t)); // then 50 fps
    expect(r.perSecond()).toBeGreaterThan(49.5);
  });

  it("rates a stage of steps by the mean over its finished windows", () => {
    const r = new StageRate();
    r.feed(p(1, 3, 8, "cnmf", "steps"));
    expect(r.left()).toBeCloseTo(16, 6);
    r.feed(p(2, 3, 20, "cnmf", "steps"));
    expect(r.left()).toBeCloseTo(10, 6);
  });
});

describe("Countdown", () => {
  it("counts down on its clock and ignores estimates within 5 s or 5%", () => {
    const c = new Countdown();
    c.update(180, 0);
    expect(c.left(10)).toBe(170);
    c.update(174, 10); // 4 s off: kept
    expect(c.left(10)).toBe(170);
  });

  it("moves 30% of the way down, 10% up, and straight to an estimate under half", () => {
    const c = new Countdown();
    c.update(200, 0);
    c.update(100 + 50, 50); // shown 150: no change needed
    c.update(120, 50); // 30 s lower → 150 - 9
    expect(c.left(50)).toBeCloseTo(141, 6);
    c.update(241, 50); // 100 s higher → 141 + 10
    expect(c.left(50)).toBeCloseTo(151, 6);
    c.update(40, 50); // under half: snap
    expect(c.left(50)).toBe(40);
  });
});

describe("fmtLeft", () => {
  it("rounds up to rough steps", () => {
    expect(fmtLeft(3)).toBe("a few seconds");
    expect(fmtLeft(31)).toBe("35 s");
    expect(fmtLeft(57)).toBe("1:00");
    expect(fmtLeft(161)).toBe("2:50");
    expect(fmtLeft(601)).toBe("11 min");
    expect(fmtLeft(3600)).toBe("1 h");
    expect(fmtLeft(4801)).toBe("1 h 21 min");
  });
});

describe("runLeft", () => {
  const plan = { frames: 9000, windows: 3 };
  const after1 = new Map<string, StageProgress>([
    ["motion", p(9400, 9400, 190)],
    ["pass1", p(3000, 9000, 70, "pass1")],
    ["pass2", p(100, 300, 28, "pass2", "tiles")],
    ["seeds", p(1, 3, 0.4, "seeds", "steps")],
    ["init", p(1, 3, 0.7, "init", "steps")],
    ["cnmf", p(1, 3, 8, "cnmf", "steps")],
  ]);

  it("waits for the first window of every per-window stage", () => {
    const m = new Map(after1);
    m.delete("cnmf");
    expect(runLeft(m, plan)).toBeNull();
    expect(runLeft(after1, null)).toBeNull();
  });

  it("adds the windows left at each stage's rate and the streaming pass at motion's speed", () => {
    const windows = 2 * (70 + 28 + 0.4 + 0.7 + 8);
    expect(runLeft(after1, plan)).toBeCloseTo(windows + (9000 * 190) / 9400, 6);
  });
});
