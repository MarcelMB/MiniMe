import { describe, expect, it, vi } from "vitest";
import type { NDArray } from "./api";
import { FrameCache, STALE, latest } from "./frames";

const arr = (t: number): NDArray => ({ shape: [1], dtype: "uint8", data: new Uint8Array([t]) });

describe("FrameCache", () => {
  it("fetches each frame once and evicts the oldest", async () => {
    const f = vi.fn((_: string, t: number) => Promise.resolve(arr(t)));
    const c = new FrameCache(f, 2);
    await c.get("recon", 1);
    await c.get("recon", 1);
    expect(f).toHaveBeenCalledTimes(1);
    await c.get("recon", 2);
    await c.get("recon", 1); // touches 1: now 2 is the oldest
    await c.get("recon", 3);
    expect(c.has("recon", 1)).toBe(true);
    expect(c.has("recon", 2)).toBe(false);
  });
  it("forgets failed fetches and prefetches inside the recording", async () => {
    const f = vi.fn().mockRejectedValueOnce(new Error("x")).mockResolvedValue(arr(0));
    const c = new FrameCache(f, 50);
    await expect(c.get("raw", 0)).rejects.toThrow();
    expect(c.has("raw", 0)).toBe(false);
    c.prefetch("raw", 1, 5, 12, 3);
    expect(f.mock.calls.slice(1).map((x) => x[1]).sort()).toEqual([0, 2, 3, 4]);
  });
});

describe("latest", () => {
  it("lets only the most recent request through", async () => {
    const wrap = latest();
    let resolveFirst!: (v: number) => void;
    const first = wrap(new Promise<number>((r) => (resolveFirst = r)));
    const second = wrap(Promise.resolve(2));
    resolveFirst(1);
    expect(await second).toBe(2);
    expect(await first).toBe(STALE);
  });
});
