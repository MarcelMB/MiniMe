import { expect, it } from "vitest";
import { externalContours } from "./contours";
import { centroids, unitPolygons } from "./units";

const csr = (rows: [number, number][][], width: number) => {
  const indptr = [0];
  const indices: number[] = [];
  const data: number[] = [];
  for (const row of rows) {
    for (const [p, v] of row) {
      indices.push(p);
      data.push(v);
    }
    indptr.push(indices.length);
  }
  return {
    shape: [rows.length, width] as [number, number],
    data: Float32Array.from(data),
    indices: Int32Array.from(indices),
    indptr: Int32Array.from(indptr),
  };
};

it("weights pixels by footprint value and gives NaN for empty units", () => {
  const a = {
    shape: [2, 6] as [number, number],
    data: new Float32Array([1, 3]),
    indices: new Int32Array([0, 5]),
    indptr: new Int32Array([0, 2, 2]),
  };
  const c = centroids(a, 2, 3); // pixel 0 = (0,0), pixel 5 = (1,2)
  expect(Array.from(c.slice(0, 2))).toEqual([0.75, 1.5]);
  expect(Number.isNaN(c[2]) && Number.isNaN(c[3])).toBe(true);
});

it("refuses a footprint matrix whose width is not H * W", () => {
  expect(() => centroids(csr([[[0, 1]]], 5), 2, 3)).toThrow("A has 5 columns, expected H * W = 6");
});

it("outlines the pixels at or above a fifth of the unit's peak, and none for an empty unit", () => {
  // 3x4 image: unit 0 has a bright pixel (1,1), a dim neighbour above the threshold (1,2), and a
  // very dim one below it (0,0); unit 1 is empty; unit 2 has only non-positive values
  const a = csr(
    [
      [
        [0, 0.1],
        [5, 1],
        [6, 0.5],
      ],
      [],
      [[3, -1]],
    ],
    12,
  );
  expect(unitPolygons(a, 3, 4)).toEqual([
    [
      [
        [1, 1],
        [2, 1],
      ],
    ],
    [],
    [],
  ]);
});

it("has no centroids and no outlines for no units", () => {
  const a = csr([], 12);
  expect(centroids(a, 3, 4)).toHaveLength(0);
  expect(unitPolygons(a, 3, 4)).toEqual([]);
});

it("outlines a unit's box like the whole image, also at the frame edges", () => {
  // random-walk blobs on a 20 x 30 frame, outlined once by `unitPolygons` (bounding box) and once on a full mask
  let seed = 7;
  const rand = () => {
    seed = (seed * 48271) % 2147483647;
    return seed / 2147483647;
  };
  const [H, W] = [20, 30];
  const rows: [number, number][][] = [];
  for (let k = 0; k < 40; k++) {
    const px = new Map<number, number>();
    let [y, x] = [Math.floor(rand() * H), Math.floor(rand() * W)];
    for (let i = 0; i < 80; i++) {
      [y, x] = [
        Math.min(H - 1, Math.max(0, y + Math.floor(rand() * 3) - 1)),
        Math.min(W - 1, Math.max(0, x + Math.floor(rand() * 3) - 1)),
      ];
      px.set(y * W + x, rand());
    }
    rows.push([...px.entries()].sort((a, b) => a[0] - b[0]));
  }
  const a = csr(rows, H * W);
  const got = unitPolygons(a, H, W);
  rows.forEach((row, k) => {
    const max = Math.max(...Array.from(a.data.slice(a.indptr[k], a.indptr[k + 1])));
    const m = new Uint8Array(H * W);
    const thr = Math.fround(Math.fround(0.2) * max);
    for (let j = a.indptr[k]; j < a.indptr[k + 1]; j++) if (a.data[j] >= thr) m[a.indices[j]] = 1;
    expect(got[k], `unit ${k}`).toEqual(externalContours(m, H, W));
    expect(row.length).toBeGreaterThan(0);
  });
});
