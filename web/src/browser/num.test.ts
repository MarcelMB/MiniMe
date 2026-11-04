import { describe, expect, it } from "vitest";
import { pyFloat, pyRound, roundHalfEven } from "./num";

// [x, digits, Python round(x, digits), numpy np.round(x, digits)], expected values from:
//   python -c "import numpy as np; c=[(0.05,1),(0.15,1),(0.35,1),(0.45,1),(0.95,1),(2.675,2),(1.005,2),(0.5,0),(1.5,0),
//   (2.5,0),(-0.5,0),(1.25,1),(-2.5,0),(-0.15,1),(2.6,0),(3.0,1)]; print([round(x,d) for x,d in c]); print([float(np.round(x,d)) for x,d in c])"
const table: [number, number, number, number][] = [
  [0.05, 1, 0.1, 0],
  [0.15, 1, 0.1, 0.2],
  [0.35, 1, 0.3, 0.4],
  [0.45, 1, 0.5, 0.4],
  [0.95, 1, 0.9, 1],
  [2.675, 2, 2.67, 2.68],
  [1.005, 2, 1, 1],
  [0.5, 0, 0, 0],
  [1.5, 0, 2, 2],
  [2.5, 0, 2, 2],
  [-0.5, 0, -0, -0],
  [1.25, 1, 1.2, 1.2],
  [-2.5, 0, -2, -2],
  [-0.15, 1, -0.1, -0.2],
  [2.6, 0, 3, 3],
  [3.0, 1, 3, 3],
];

describe("rounding", () => {
  it("roundHalfEven rounds half to even on the product, like np.round", () => {
    expect([0.5, 1.5, 2.5, -0.5, 2.6].map((x) => roundHalfEven(x))).toEqual([0, 2, 2, -0, 3]);
    expect(roundHalfEven(1.25, 1)).toBe(1.2);
    expect(roundHalfEven(3.0, 1)).toBe(3);
    for (const [x, d, , np] of table) expect(roundHalfEven(x, d), `np.round(${x}, ${d})`).toBe(np);
  });

  it("pyRound is Python's round: exact binary value, true ties to even", () => {
    for (const [x, d, py] of table) expect(pyRound(x, d), `round(${x}, ${d})`).toBe(py);
  });
});

describe("pyFloat", () => {
  it("writes floats as Python's str() does", () => {
    // python -c "print([str(x) for x in (-1.0, 0.0, -0.0, 2000.0, 2.5, 0.1, float('nan'), float('inf'), -float('inf'))])"
    expect(
      [-1, 0, -0, 2000, 2.5, 0.1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY].map(pyFloat),
    ).toEqual(["-1.0", "0.0", "-0.0", "2000.0", "2.5", "0.1", "nan", "inf", "-inf"]);
  });
});
