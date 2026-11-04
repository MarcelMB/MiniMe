import { afterEach, describe, expect, it, vi } from "vitest";
import { fmtBytes } from "../format";
import { checkTraceRoom, estimateRunBytes, quotaMessage, support } from "./support";

describe("support", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("reports the system folder dialog (File System Access) and OPFS separately", () => {
    vi.stubGlobal("navigator", {});
    expect(support()).toEqual({ fsAccess: false, opfs: false });
    vi.stubGlobal("navigator", { storage: { getDirectory: async () => ({}) } });
    expect(support()).toEqual({ fsAccess: false, opfs: true });
    vi.stubGlobal("showDirectoryPicker", async () => ({}));
    expect(support()).toEqual({ fsAccess: true, opfs: true });
    vi.stubGlobal("navigator", { storage: {} });
    expect(support()).toEqual({ fsAccess: true, opfs: false });
  });

  it("copes with a page that has no navigator.storage at all", () => {
    vi.stubGlobal("navigator", undefined);
    expect(support().opfs).toBe(false);
  });
});

describe("estimateRunBytes", () => {
  // a 3 h session at 600 x 600 px, 20 fps: 72 windows of 3000 frames, about 400 units
  const [K, T, P, W] = [400, 216000, 360000, 72];

  it("counts the traces twice (scratch stream_out/C.npy, then cnmf/C.npy) and the window factors", () => {
    const traces = K * T * 4;
    expect(estimateRunBytes(K, T, P, W)).toBeGreaterThanOrEqual(2 * traces);
    expect(estimateRunBytes(K, T, P, W)).toBeGreaterThan(2 * traces + W * P * 20);
  });

  it("adds the tile store (a byte per pixel and frame) when it is kept", () => {
    expect(estimateRunBytes(K, T, P, W, true) - estimateRunBytes(K, T, P, W)).toBe(T * P);
  });

  it("grows with every input", () => {
    const base = estimateRunBytes(K, T, P, W);
    for (const more of [
      estimateRunBytes(K + 1, T, P, W),
      estimateRunBytes(K, T + 1, P, W),
      estimateRunBytes(K, T, P + 1, W),
      estimateRunBytes(K, T, P, W + 1),
    ])
      expect(more).toBeGreaterThan(base);
  });
});

describe("checkTraceRoom (the second storage check, once the stitch has found K units)", () => {
  const storage = (quota: number, usage: number) => ({ estimate: async () => ({ quota, usage }) });
  const [K, T] = [800, 216000]; // twice the units the pre-run check assumed, 3 h at 20 fps
  const traces = K * T * 4;

  it("counts the traces twice for a run in Browser storage and refuses with the real numbers", async () => {
    await expect(checkTraceRoom(K, T, true, storage(2 * traces + 1e6, 1e6))).resolves.toBeUndefined();
    await expect(checkTraceRoom(K, T, true, storage(2 * traces, 1e6))).rejects.toThrow(
      `this run needs about ${fmtBytes(2 * traces)} of browser storage, ${fmtBytes(2 * traces - 1e6)} are free; pick a folder on disk under Runs → Change folder`,
    );
  });

  it("counts the scratch copy alone for a run in a folder on disk", async () => {
    await expect(checkTraceRoom(K, T, false, storage(traces + 1, 0))).resolves.toBeUndefined();
    await expect(checkTraceRoom(K, T, false, storage(traces - 1, 0))).rejects.toThrow(
      quotaMessage(traces, traces - 1, true),
    );
    expect(quotaMessage(traces, 5e9, true)).toMatch(/for its traces on their way to the run folder/);
  });

  it("checks nothing where the browser cannot say", async () => {
    await expect(checkTraceRoom(K, T, true, undefined)).resolves.toBeUndefined();
    await expect(checkTraceRoom(K, T, true, { estimate: async () => ({}) })).resolves.toBeUndefined();
    await expect(
      checkTraceRoom(K, T, true, { estimate: () => Promise.reject(new Error("no")) }),
    ).resolves.toBeUndefined();
  });
});
