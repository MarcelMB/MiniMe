import { describe, expect, it } from "vitest";
import { defaultSession } from "./session";
import ref from "./session.ref.json";

describe("defaultSession", () => {
  // session.ref.json is what ref_stitch.py --session-inputs --stream wrote for the 600x600
  // GRIN synth with MiniMe's default parameters (2026-11-03)
  it("equals the session.json Python writes from the default parameters", () => {
    expect(defaultSession(600, 600)).toEqual(ref);
  });
  it("takes the frame size and window length", () => {
    const s = defaultSession(480, 752, 2000);
    expect(s.window_frames).toBe(2000);
    for (const stage of [s.pass1, s.pass2, s.seeds, s.init, s.cnmf, s.stream])
      expect([stage.H, stage.W]).toEqual([480, 752]);
  });
});
