import { expect, it } from "vitest";
import { tileGrid, tileMap } from "./tiles";

it("lays out tiles row by row, the last row and column partial", () => {
  expect(tileGrid(5, 7, 3)).toEqual([
    { y0: 0, y1: 3, x0: 0, x1: 3 },
    { y0: 0, y1: 3, x0: 3, x1: 6 },
    { y0: 0, y1: 3, x0: 6, x1: 7 },
    { y0: 3, y1: 5, x0: 0, x1: 3 },
    { y0: 3, y1: 5, x0: 3, x1: 6 },
    { y0: 3, y1: 5, x0: 6, x1: 7 },
  ]);
});

it("fills each tile with its value", () => {
  const m = tileMap(3, 4, 2, [1, 2, 3, 4]);
  expect(Array.from(m)).toEqual([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4]);
});

it("refuses a value count that is not the tile count", () => {
  expect(() => tileMap(4, 4, 2, [1, 2, 3])).toThrow("3 values for 4 tiles");
});
