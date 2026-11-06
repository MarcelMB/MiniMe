import { describe, expect, it } from "vitest";
import { href, parseHash } from "./router";

describe("router", () => {
  it("parses every route and round-trips", () => {
    for (const r of [
      { name: "start" },
      { name: "home" },
      { name: "new" },
      { name: "job", id: "abc" },
      { name: "run", id: "0123456789ab", page: "traces" },
    ] as const) {
      expect(parseHash(href(r))).toEqual(r);
    }
  });
  it("falls back to the landing page and to the overview page", () => {
    expect(parseHash("")).toEqual({ name: "start" });
    expect(parseHash("#/")).toEqual({ name: "start" });
    expect(parseHash("#/nowhere")).toEqual({ name: "start" });
    expect(parseHash("#/runs")).toEqual({ name: "home" });
    expect(parseHash("#/run/abc")).toEqual({ name: "run", id: "abc", page: "overview" });
    expect(parseHash("#/run/abc/bogus")).toEqual({ name: "run", id: "abc", page: "overview" });
  });
});
