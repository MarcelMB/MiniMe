import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, Computing } from "./api";
import { pollResource, type Resource } from "./poll";

describe("pollResource", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("retries while computing, then delivers the data", async () => {
    const load = vi
      .fn<() => Promise<number>>()
      .mockRejectedValueOnce(new Computing(1))
      .mockRejectedValueOnce(new Computing(2.5))
      .mockResolvedValueOnce(42);
    const states: Resource<number>[] = [];
    pollResource(load, (s) => states.push(s), 1000);
    await vi.runAllTimersAsync();
    expect(load).toHaveBeenCalledTimes(3);
    expect(states.map((s) => s.computing)).toEqual([true, true, false]);
    expect(states[1].elapsed).toBe(2.5);
    expect(states.at(-1)!.data).toBe(42);
  });

  it("stops on an error and after cancel", async () => {
    const states: Resource<number>[] = [];
    pollResource(() => Promise.reject(new ApiError(500, "boom")), (s) => states.push(s), 1000);
    await vi.runAllTimersAsync();
    expect(states.at(-1)!.error).toBe("boom");

    const load = vi.fn().mockRejectedValue(new Computing(0));
    const cancel = pollResource(load, () => {}, 1000);
    await vi.advanceTimersByTimeAsync(10);
    cancel();
    await vi.advanceTimersByTimeAsync(5000);
    expect(load).toHaveBeenCalledTimes(1);
  });
});
