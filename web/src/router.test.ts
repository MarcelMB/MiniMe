import { describe, expect, it } from "vitest";
import { href, parseHash } from "./router";

describe("router", () => {
  it("parses every route and round-trips", () => {
    for (const r of [
      { name: "home" },
      { name: "new" },
      { name: "job", id: "abc" },
      { name: "run", id: "0123456789ab", page: "traces" },
    ] as const) {
      expect(parseHash(href(r))).toEqual(r);
    }
  });
  it("falls back to home and to the overview page", () => {
    expect(parseHash("")).toEqual({ name: "home" });
    expect(parseHash("#/nowhere")).toEqual({ name: "home" });
    expect(parseHash("#/run/abc")).toEqual({ name: "run", id: "abc", page: "overview" });
    expect(parseHash("#/run/abc/bogus")).toEqual({ name: "run", id: "abc", page: "overview" });
  });
});
