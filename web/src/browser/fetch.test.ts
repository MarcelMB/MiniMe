import { describe, expect, it } from "vitest";
import { decodeArray } from "../api";
import { createFetch } from "./fetch";
import { HttpError, Router } from "./router";

describe("createFetch", () => {
  const r = new Router();
  r.add("GET", "/a", (q) => ({ json: { x: q.query.get("x") } }));
  r.add("POST", "/b", (q) => ({ json: q.body }));
  r.add("GET", "/arr", () => ({ array: { shape: [2, 2], dtype: "float32", data: new Float32Array([1, 2, 3, 4]) } }));
  r.add("GET", "/busy", () => ({ computing: 3.5 }));
  r.add("GET", "/bad", () => {
    throw new HttpError(400, "frame 9 out of range 0..3");
  });
  r.add("GET", "/boom", () => {
    throw new Error("kaput");
  });
  const f = createFetch(r);

  it("answers JSON with the query and body", async () => {
    expect(await (await f("/api/a?x=1")).json()).toEqual({ x: "1" });
    const res = await f("/api/b", { method: "POST", body: JSON.stringify({ y: 2 }) });
    expect(await res.json()).toEqual({ y: 2 });
  });

  it("answers arrays with X-Shape and X-Dtype", async () => {
    const res = await f("/api/arr");
    const a = decodeArray(await res.arrayBuffer(), res.headers.get("X-Shape")!, res.headers.get("X-Dtype")!);
    expect(a.shape).toEqual([2, 2]);
    expect(Array.from(a.data)).toEqual([1, 2, 3, 4]);
  });

  it("maps computing to 202 and errors to main's {detail}", async () => {
    const busy = await f("/api/busy");
    expect(busy.status).toBe(202);
    expect(await busy.json()).toEqual({ elapsed_s: 3.5 });
    const bad = await f("/api/bad");
    expect(bad.status).toBe(400);
    expect(await bad.json()).toEqual({ detail: "frame 9 out of range 0..3" });
    expect((await f("/api/boom")).status).toBe(500);
    expect((await f("/api/missing")).status).toBe(404);
  });
});
