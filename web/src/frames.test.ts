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

describe("FrameCache limits", () => {
  const deferred = () => {
    let resolve!: (a: NDArray) => void;
    const promise = new Promise<NDArray>((r) => (resolve = r));
    return { promise, resolve };
  };

  it("keeps at most 4 prefetches in flight but never caps get()", async () => {
    const pending: ((a: NDArray) => void)[] = [];
    const f = vi.fn(
      () =>
        new Promise<NDArray>((r) => {
          pending.push(r);
        }),
    );
    const c = new FrameCache(f, 100);
    c.prefetch("raw", 50, 1000, 12, 3);
    expect(f).toHaveBeenCalledTimes(4);
    c.prefetch("raw", 50, 1000, 12, 3);
    expect(f).toHaveBeenCalledTimes(4);
    void c.get("raw", 500); // the current frame is never capped
    expect(f).toHaveBeenCalledTimes(5);
    pending[0](arr(0));
    await Promise.resolve();
    await Promise.resolve();
    c.prefetch("raw", 50, 1000, 12, 3);
    expect(f.mock.calls.length).toBeGreaterThan(5);
  });

  it("frees the prefetch slot when a fetch fails", async () => {
    const f = vi.fn(() => Promise.reject(new Error("x")));
    const c = new FrameCache(f, 100);
    c.prefetch("raw", 50, 1000, 12, 3);
    expect(f).toHaveBeenCalledTimes(4);
    await new Promise((r) => setTimeout(r, 0));
    c.prefetch("raw", 50, 1000, 12, 3);
    expect(f.mock.calls.length).toBeGreaterThan(4);
  });

  it("never evicts a pending entry; the oldest settled one goes instead", async () => {
    const d = deferred();
    const f = vi.fn((_: string, t: number) => (t === 1 ? d.promise : Promise.resolve(arr(t))));
    const c = new FrameCache(f, 2);
    void c.get("raw", 1); // pending and oldest
    await c.get("raw", 2);
    await c.get("raw", 3); // over capacity: 2 is the oldest settled
    expect(c.has("raw", 1)).toBe(true);
    expect(c.has("raw", 2)).toBe(false);
    expect(c.has("raw", 3)).toBe(true);
    await c.get("raw", 4);
    await c.get("raw", 5);
    expect(c.has("raw", 1)).toBe(true);
    d.resolve(arr(1));
  });

  it("lets the size exceed capacity while everything is pending", () => {
    const f = vi.fn(() => new Promise<NDArray>(() => {}));
    const c = new FrameCache(f, 2);
    for (let t = 0; t < 4; t++) void c.get("raw", t);
    for (let t = 0; t < 4; t++) expect(c.has("raw", t)).toBe(true);
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
  it("turns a stale rejection into STALE and keeps a current one", async () => {
    const wrap = latest();
    let rejectFirst!: (e: Error) => void;
    const first = wrap(new Promise<number>((_, r) => (rejectFirst = r)));
    const second = wrap(Promise.reject(new Error("now")));
    rejectFirst(new Error("late"));
    expect(await first).toBe(STALE);
    await expect(second).rejects.toThrow("now");
  });
});
