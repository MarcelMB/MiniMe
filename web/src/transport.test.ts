import { afterEach, describe, expect, it, vi } from "vitest";
import { subscribe } from "./api";
import { openEvents } from "./transport";

class FakeEventSource {
  static last: FakeEventSource | null = null;
  listeners = new Map<string, (e: MessageEvent) => void>();
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeEventSource.last = this;
  }
  addEventListener(name: string, fn: (e: MessageEvent) => void) {
    this.listeners.set(name, fn);
  }
  close() {
    this.closed = true;
  }
  emit(name: string, data: unknown) {
    this.listeners.get(name)?.({ data: JSON.stringify(data) } as MessageEvent);
  }
}

describe("openEvents", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("parses each named event and closes before reporting an error", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const got: unknown[] = [];
    const onError = vi.fn();
    openEvents("/api/jobs/a/events?token=x", { stage: (d) => got.push(d) }, onError);
    const es = FakeEventSource.last!;
    es.emit("stage", { stage: "pass1", event: "start" });
    expect(got).toEqual([{ stage: "pass1", event: "start" }]);
    es.onerror?.();
    expect(es.closed).toBe(true);
    expect(onError).toHaveBeenCalledOnce();
  });

  it("subscribe adds /api and the token query", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const close = subscribe("/jobs/j1/events", {}, () => {});
    expect(FakeEventSource.last!.url).toBe("/api/jobs/j1/events?token=");
    close();
    expect(FakeEventSource.last!.closed).toBe(true);
  });
});
