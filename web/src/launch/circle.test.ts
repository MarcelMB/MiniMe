import { describe, expect, it } from "vitest";
import { diameterOf, dragCircle, toImagePixel } from "./circle";

describe("toImagePixel", () => {
  const rect = { left: 10, top: 20, width: 200, height: 100 };
  it("maps the letter-boxed image and clamps outside points", () => {
    // a 100 x 100 image fills the middle 100 px of a 200 x 100 box
    expect(toImagePixel(rect, 100, 100, 60, 20)).toEqual([-0.5, -0.5]);
    expect(toImagePixel(rect, 100, 100, 110, 70)).toEqual([49.5, 49.5]);
    expect(toImagePixel(rect, 100, 100, 500, 500)).toEqual([99.5, 99.5]);
  });
});

describe("dragCircle", () => {
  const c = { cx: 10, cy: 10, r: 4 };
  it("moves with the grab offset and resizes to the pointer", () => {
    expect(dragCircle(c, "move", 20, 15, [2, 1])).toEqual({ cx: 18, cy: 14, r: 4 });
    expect(dragCircle(c, "resize", 13, 14, [0, 0])).toEqual({ cx: 10, cy: 10, r: 5 });
    expect(dragCircle(c, "resize", 10, 10, [0, 0]).r).toBe(0.5);
  });
});

it("diameterOf rounds to half a pixel", () => {
  expect(diameterOf({ cx: 0, cy: 0, r: 4.3 })).toBe(8.5);
  expect(diameterOf({ cx: 0, cy: 0, r: 4.1 })).toBe(8);
});
