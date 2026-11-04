import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";
import { externalContours } from "./contours";

const cases = JSON.parse(readFileSync(join(__dirname, "../__fixtures__/contours.json"), "utf8")) as {
  h: number;
  w: number;
  mask: number[];
  contours: number[][][];
}[];

it.each(cases.map((c, i) => [i, c] as const))("matches cv2 on case %i", (_i, c) => {
  expect(externalContours(Uint8Array.from(c.mask), c.h, c.w)).toEqual(c.contours);
});

it("gives no contour for an empty mask and one point for a single pixel", () => {
  expect(externalContours(new Uint8Array(9), 3, 3)).toEqual([]);
  expect(externalContours(Uint8Array.from([0, 0, 0, 0, 1, 0, 0, 0, 0]), 3, 3)).toEqual([[[1, 1]]]);
});
