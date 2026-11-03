import { describe, expect, it } from "vitest";
import { HttpError, type Req, Router } from "./router";

const req = (method: string, path: string): Req => ({ method, path, query: new URLSearchParams(), body: null });

describe("Router", () => {
  it("matches params in order of registration", async () => {
    const r = new Router();
    r.add("GET", "/runs/:rid/traces/overview", () => ({ json: "overview" }));
    r.add("GET", "/runs/:rid/traces/:unit", (_q, p) => ({ json: p }));
    expect((await r.handle(req("GET", "/runs/ab/traces/overview"))).json).toBe("overview");
    expect((await r.handle(req("GET", "/runs/ab/traces/7"))).json).toEqual({ rid: "ab", unit: "7" });
  });

  it("decodes path segments and 404s unknown paths and methods", async () => {
    const r = new Router();
    r.add("GET", "/runs/:rid", (_q, p) => ({ json: p.rid }));
    expect((await r.handle(req("GET", "/runs/a%20b"))).json).toBe("a b");
    await expect(r.handle(req("POST", "/runs/x"))).rejects.toMatchObject({ status: 404 });
    await expect(r.handle(req("GET", "/nope"))).rejects.toBeInstanceOf(HttpError);
  });
});
