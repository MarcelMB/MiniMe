import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, Computing, decodeArray, getArray, getJSON, readToken } from "./api";

function memStore() {
  const m = new Map<string, string>();
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) };
}

describe("token", () => {
  it("takes the token from the URL and remembers it", () => {
    const s = memStore();
    expect(readToken("?token=abc&root=/x", s)).toBe("abc");
    expect(readToken("", s)).toBe("abc");
  });
  it("is empty without one", () => {
    expect(readToken("", memStore())).toBe("");
  });
});

describe("decodeArray", () => {
  it("decodes float32 with its shape", () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const d = decodeArray(a.buffer, "2,3", "float32");
    expect(d.shape).toEqual([2, 3]);
    expect(Array.from(d.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });
  it("handles an empty array and rejects unknown dtypes", () => {
    expect(decodeArray(new ArrayBuffer(0), "0,2", "float32").shape).toEqual([0, 2]);
    expect(() => decodeArray(new ArrayBuffer(8), "1", "float64")).toThrow();
  });
});

describe("requests", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("turns 202 into Computing and errors into ApiError with the detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify({ status: "computing", elapsed_s: 3 }), { status: 202 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "unknown run x" }), { status: 404 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({ status: "failed", error: "boom" }), { status: 500 })),
    );
    await expect(getJSON("/a")).rejects.toBeInstanceOf(Computing);
    await expect(getJSON("/b")).rejects.toMatchObject({ status: 404, message: "unknown run x" });
    const e = (await getJSON("/c").catch((x) => x)) as Error;
    expect(e).toBeInstanceOf(ApiError);
    expect(e.message).toBe("boom");
  });

  it("decodes array responses from the headers", async () => {
    const body = new Uint8Array([1, 2, 3, 4]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(body, { status: 200, headers: { "X-Shape": "2,2", "X-Dtype": "uint8" } }),
      ),
    );
    const a = await getArray("/frame");
    expect(a.shape).toEqual([2, 2]);
    expect(Array.from(a.data)).toEqual([1, 2, 3, 4]);
    expect((fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls[0][0]).toBe("/api/frame");
  });
});
