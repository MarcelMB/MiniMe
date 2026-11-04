import { afterEach, describe, expect, it, vi } from "vitest";
import { estimateRunBytes, support } from "./support";

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
