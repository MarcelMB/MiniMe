import { describe, expect, it, vi } from "vitest";
import { addEventProvider, openEvents } from "./events";

describe("openEvents (browser)", () => {
  it("routes /api/<path>?query to the first provider that takes it", () => {
    let emit: ((n: string, d: unknown) => void) | null = null;
    addEventProvider((path, e) => {
      if (path !== "/jobs/j1/events") return null;
      emit = e;
      return () => {};
    });
    const got: unknown[] = [];
    openEvents("/api/jobs/j1/events?token=", { log: (d) => got.push(d) }, () => {});
    emit!("log", "hello");
    expect(got).toEqual(["hello"]);
  });

  it("reports an error when no provider takes the path", async () => {
    const onError = vi.fn();
    openEvents("/api/jobs/none/events", {}, onError);
    await Promise.resolve();
    expect(onError).toHaveBeenCalledOnce();
  });
});
