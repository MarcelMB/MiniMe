import { describe, expect, it } from "vitest";
import { chunkWeights, decide, estimateSeconds, type SpeedTest, speedKey, splitWorkers } from "./plan";

const test = (fps: SpeedTest["fps"], bothGpu = 6): SpeedTest => ({
  key: "k",
  when: 0,
  gpu: "apple",
  gpuReason: "",
  workers: 8,
  cores: 10,
  H: 600,
  W: 600,
  fps,
  bothGpu,
  bothRates: [],
  seconds: 1,
});

describe("decide", () => {
  it("takes what was asked for, and the CPU when the GPU cannot be used", () => {
    expect(decide("cpu", 8, "", test({ cpu: 1, gpu: 9, both: 5 })).use).toBe("cpu");
    expect(decide("gpu", 8, "", null)).toMatchObject({ use: "gpu", gpuWorkers: 8 });
    expect(decide("both", 8, "", test({ cpu: 1, gpu: 9, both: 5 }, 3))).toMatchObject({ use: "both", gpuWorkers: 3 });
    expect(decide("both", 1, "", null)).toMatchObject({ use: "gpu", gpuWorkers: 1 });
    expect(decide("gpu", 8, "no WebGPU", null)).toMatchObject({ use: "cpu", why: "no WebGPU" });
    expect(decide("auto", 8, "self-test: differs", test({ cpu: 1, gpu: 9, both: 5 })).use).toBe("cpu");
  });

  it("auto picks the fastest measured mode", () => {
    expect(decide("auto", 8, "", test({ cpu: 800, gpu: 1500, both: 1200 })).use).toBe("gpu");
    expect(decide("auto", 8, "", test({ cpu: 800, gpu: 600, both: 1000 }))).toMatchObject({ use: "both", gpuWorkers: 6 });
    expect(decide("auto", 8, "", test({ cpu: 900, gpu: 600, both: 800 })).use).toBe("cpu");
    expect(decide("auto", 1, "", test({ cpu: 100, gpu: 90, both: 300 })).use).toBe("cpu"); // no split with one worker
    expect(decide("auto", 8, "", null).use).toBe("cpu");
  });
});

describe("splitWorkers", () => {
  it("puts as many workers on the GPU as keep it busy, at least one on each side", () => {
    expect(splitWorkers(8, 1500, 250)).toBe(6);
    expect(splitWorkers(8, 3000, 250)).toBe(7);
    expect(splitWorkers(8, 100, 250)).toBe(1);
    expect(splitWorkers(1, 100, 250)).toBe(1);
  });
});

describe("estimateSeconds", () => {
  it("scales a CPU run by the measured pass-1 rates over the device's share", () => {
    const e = estimateSeconds(9000, { cpu: 750, gpu: 1500, both: 0 });
    expect(e.cpu).toBeCloseTo(9000 / 750 / 0.42);
    expect(e.gpu).toBeCloseTo(e.cpu * (0.48 + 0.52 / 2));
    expect(e.both).toBe(0);
    expect(estimateSeconds(9000, { cpu: 0, gpu: 1, both: 1 })).toEqual({ cpu: 0, gpu: 0, both: 0 });
  });
});

describe("chunkWeights and speedKey", () => {
  it("weighs chunks only when the workers' rates differ", () => {
    expect(chunkWeights(4, [100, 101, 99, 100])).toBeUndefined();
    expect(chunkWeights(4, [200, 200, 100, 100])).toEqual([200, 200, 100, 100]);
    expect(chunkWeights(4, [200, 200])).toBeUndefined();
    expect(chunkWeights(2, [200, 0])).toBeUndefined();
  });

  it("keys a test by version, GPU, cores, frame size and workers", () => {
    expect(speedKey("apple · metal-3", 10, 600, 600, 8)).toBe("gpu-path-1|apple · metal-3|cores10|600x600|w8");
    expect(speedKey("", 4, 480, 752, 2)).toBe("gpu-path-1|no-gpu|cores4|480x752|w2");
  });
});
