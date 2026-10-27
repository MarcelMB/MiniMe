import { describe, expect, it } from "vitest";
import {
  changedOnly,
  defaultRunName,
  failingCount,
  isFinal,
  overwriteAfterError,
  parseValue,
  stageRows,
} from "./params";

describe("parameter values", () => {
  it("parses each parameter type", () => {
    expect(parseValue("int", "7")).toBe(7);
    expect(parseValue("float", "0.8")).toBe(0.8);
    expect(parseValue("bool", true)).toBe(true);
    expect(parseValue("float | str", "auto")).toBe("auto");
    expect(parseValue("float | str", "3.5")).toBe(3.5);
    expect(parseValue("tuple[float | None, float | None]", ["25", ""])).toEqual(
      [25, null],
    );
    expect(parseValue("tuple[float, float]", ["1", "6"])).toEqual([1, 6]);
  });
  it("rejects bad numbers", () => {
    expect(() => parseValue("int", "7.5")).toThrow(/whole number/);
    expect(() => parseValue("float", "abc")).toThrow(/number/);
    expect(() => parseValue("tuple[float, float]", ["1", ""])).toThrow();
  });
  it("sends only values that differ from the defaults", () => {
    const metas = [
      { name: "workers", default: 1 },
      { name: "size_thres", default: [25, null] },
    ];
    expect(changedOnly({ workers: 6, size_thres: [25, null] }, metas)).toEqual({
      workers: 6,
    });
  });
});

describe("launcher helpers", () => {
  it("names runs by date", () => {
    expect(defaultRunName(new Date(2026, 9, 21, 14, 5))).toBe(
      "minime_2026-10-21_1405",
    );
  });
  it("turns progress events into stage rows", () => {
    const rows = stageRows([
      { stage: "probe", event: "start", t: 1 },
      { stage: "probe", event: "end", t: 2, seconds: 1.5 },
      { stage: "pass1", event: "start", t: 2 },
    ]);
    expect(rows).toEqual([
      { stage: "probe", seconds: 1.5, running: false, failed: false },
      { stage: "pass1", seconds: null, running: true, failed: false },
    ]);
  });
  it("marks a stage that raised as failed", () => {
    const rows = stageRows([
      { stage: "pass1", event: "start", t: 2 },
      { stage: "pass1", event: "error", t: 5, seconds: 3 },
    ]);
    expect(rows).toEqual([
      { stage: "pass1", seconds: 3, running: false, failed: true },
    ]);
  });
});

describe("launcher state rules", () => {
  it("knows final job states", () => {
    expect(isFinal("running")).toBe(false);
    expect(isFinal("queued")).toBe(false);
    for (const s of ["done", "failed", "cancelled", "interrupted"])
      expect(isFinal(s)).toBe(true);
  });
  it("keeps overwrite only after the already-holds-a-run refusal", () => {
    expect(
      overwriteAfterError(
        "x already holds a run; choose another name or overwrite",
        true,
      ),
    ).toBe(true);
    expect(
      overwriteAfterError("a queued or running job already writes to x", true),
    ).toBe(false);
    expect(overwriteAfterError("network down", true)).toBe(false);
  });
  it("counts invalid parameters", () => {
    expect(failingCount({ a: "bad", b: "", c: "worse" })).toBe(2);
  });
});
