import { describe, expect, it } from "vitest";
import { pipelineSteps } from "./Preview";

const v = { median_ksize: 3, motion_bandpass: [1, 6], motion_max_shift: 20, tophat_wnd: 8, tile: 64 };

describe("preprocessing order", () => {
  it("puts motion before the tophat by default, on the band-passed frame", () => {
    const labels = pipelineSteps({ ...v, motion_before_tophat: true }).map((s) => s.label);
    expect(labels.slice(0, 4)).toEqual(["median 3 px", "band-pass σ 1–6 px", "motion correction ≤ 20 px", "tophat 8 px"]);
    expect(pipelineSteps({ ...v, motion_before_tophat: true }).map((s) => s.shown)).toEqual([true, true, false, true, false, false]);
  });
  it("puts the tophat first when motion_before_tophat is off", () => {
    const labels = pipelineSteps({ ...v, motion_before_tophat: false }).map((s) => s.label);
    expect(labels.slice(0, 3)).toEqual(["median 3 px", "tophat 8 px", "motion correction ≤ 20 px"]);
  });
});
