import { expect, it } from "vitest";
import { roundHalfEven } from "./num";

it("rounds half to even like Python", () => {
  expect([0.5, 1.5, 2.5, -0.5, 2.6].map((x) => roundHalfEven(x))).toEqual([0, 2, 2, -0, 3]);
  expect(roundHalfEven(1.25, 1)).toBe(1.2);
  expect(roundHalfEven(3.0, 1)).toBe(3);
});
