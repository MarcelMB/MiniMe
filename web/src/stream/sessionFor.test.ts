import { describe, expect, it } from "vitest";
import { DEFAULTS, defaultSession, SESSION_PARAMS, sessionFor, unusedParams } from "./session";

/** A value that differs from the default, of the same kind. */
function changed(v: unknown): unknown {
  if (typeof v === "boolean") return !v;
  if (typeof v === "number") return v + 1;
  if (v === "auto") return 7;
  if (Array.isArray(v)) return v.map((x) => (typeof x === "number" ? x + 1 : 9));
  throw new Error(`no change for ${String(v)}`);
}

describe("sessionFor", () => {
  it("is the default session without parameters", () => {
    expect(sessionFor(600, 600, {})).toEqual(defaultSession(600, 600));
    expect(sessionFor(480, 752, {}, 2000)).toEqual(defaultSession(480, 752, 2000));
  });

  it("puts the given parameters over the defaults, in every stage that reads them", () => {
    const s = sessionFor(64, 64, { tile: 32, n_iter: 5, size_thres: [9, null] });
    expect([s.pass1.tile, s.pass2.tile, s.seeds.tile, s.init.tile, s.cnmf.tile]).toEqual([32, 32, 32, 32, 32]);
    expect(s.cnmf.n_iter).toBe(5);
    expect(s.cnmf.size_thres).toEqual([9, null]);
    expect(s.stream.size_thres).toEqual([9, null]);
    expect(s.pass1.median_ksize).toBe(DEFAULTS.median_ksize);
  });

  it("ignores a parameter the pipeline does not read", () => {
    expect(sessionFor(64, 64, { workers: 4, sub_batch: 10, nonsense: 1 })).toEqual(defaultSession(64, 64));
  });
});

describe("SESSION_PARAMS and unusedParams", () => {
  it("lists exactly the defaults", () => {
    expect([...SESSION_PARAMS].sort()).toEqual(Object.keys(DEFAULTS).sort());
  });

  // unusedParams is honest only if every listed parameter really reaches the session
  it.each(SESSION_PARAMS.map((k) => [k]))("%s changes the session", (k) => {
    const v = changed((DEFAULTS as Record<string, unknown>)[k]);
    expect(sessionFor(64, 64, { [k]: v })).not.toEqual(defaultSession(64, 64));
  });

  it("names the changed parameters the browser would ignore, sorted, but not workers", () => {
    expect(unusedParams({ tile: 32, workers: 4, sub_batch: 5, a_new_one: 1 })).toEqual(["a_new_one", "sub_batch"]);
    expect(unusedParams({})).toEqual([]);
  });
});
