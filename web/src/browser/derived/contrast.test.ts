import { expect, it } from "vitest";
import { linspaceInt, percentile, toU8 } from "./contrast";

it("percentile interpolates like numpy", () => {
  const s = Float32Array.from([1, 2, 3, 4]);
  expect(percentile(s, 1)).toBeCloseTo(1.03, 6);
  expect(percentile(s, 99.9)).toBeCloseTo(3.997, 6);
});
it("percentile reproduces numpy's float32 arithmetic to the last digit", () => {
  // np.percentile(a, [1, 99.9]) for two float32 arrays; a float64 lerp is off in the last digits
  const a = Float32Array.from([
    13.752328872680664, 72.87934875488281, 97.75906372070312, 104.1606674194336, 131.40802001953125, 206.02490234375,
  ]);
  expect([percentile(a, 1), percentile(a, 99.9)]).toEqual([16.708679962158204, 205.65181793212898]);
  const b = Float32Array.from([
    11.545174598693848, 12.433216094970703, 59.80010223388672, 110.91162872314453, 166.3541259765625,
    254.78990173339844,
  ]);
  expect([percentile(b, 1), percentile(b, 99.9)]).toEqual([11.58957667350769, 254.34772285461435]);
});
it("percentile of one value and of the ends", () => {
  expect(percentile(Float32Array.from([5]), 1)).toBe(5);
  expect(percentile(Float32Array.from([2, 9]), 0)).toBe(2);
  expect(percentile(Float32Array.from([2, 9]), 100)).toBe(9);
});

it("toU8 clips and truncates", () => {
  expect(Array.from(toU8(Float32Array.from([-1, 0, 0.5, 1, 2]), 0, 1))).toEqual([0, 0, 127, 255, 255]);
});
it("toU8 rounds in float32 like numpy", () => {
  // np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8): in float64 these come out one lower
  expect(toU8(Float32Array.of(246), 35.04789352416992, 249.36178703653403)[0]).toBe(251);
  expect(toU8(Float32Array.of(140), 31.618074417114258, 212.25461806979013)[0]).toBe(153);
  expect(toU8(Float32Array.of(83), 15.49869155883789, 161.3701669084048)[0]).toBe(118);
});

it("linspaceInt is np.linspace(0, stop, n).astype(int)", () => {
  expect(linspaceInt(299, 50).slice(8, 14)).toEqual([48, 54, 61, 67, 73, 79]);
  expect(linspaceInt(299, 50).at(-1)).toBe(299);
  expect(linspaceInt(52, 50).slice(14, 20)).toEqual([14, 15, 16, 18, 19, 20]);
  expect(linspaceInt(999, 50).slice(0, 5)).toEqual([0, 20, 40, 61, 81]);
  expect(linspaceInt(35999, 50).slice(-3)).toEqual([34529, 35264, 35999]);
  // T = 120: numpy's step 119/49 makes frames 50 and 101 where (i * 119) / 49 would truncate to 51 and 102
  const t120 = linspaceInt(119, 50);
  expect([t120[21], t120[42], t120[49]]).toEqual([50, 101, 119]);
  expect(linspaceInt(2, 3)).toEqual([0, 1, 2]);
  expect(linspaceInt(1, 2)).toEqual([0, 1]);
  expect(linspaceInt(0, 1)).toEqual([0]);
});
