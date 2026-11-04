// Directions 0..7 counter-clockwise from east, in screen coordinates (y down): E, NE, N, NW, W, SW, S, SE.
const DX = [1, 1, 0, -1, -1, -1, 0, 1];
const DY = [0, -1, -1, -1, 0, 1, 1, 1];

// Pixel states in the padded label image: 0 background, 1 object not on a traced border, then
// the marks of the traced border: LEFT on a border pixel whose east side is not background,
// RIGHT on a "right bound" (OpenCV's `nbd | -128`: the sign is what the scan reads).
const OBJECT = 1;
const LEFT = 2;
const RIGHT = -126;

/**
 * Outer borders of the 1-pixels of `mask` (h, w), as
 * `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`: points `[x, y]`, a run along one
 * of the 8 directions reduced to its end points (the start point always kept), components
 * inside the holes of another component skipped, the last one found first.
 *
 * A port of OpenCV's `cvFindNextContour` + `icvFetchContour` (Suzuki and Abe's border following).
 */
export function externalContours(mask: Uint8Array, h: number, w: number): number[][][] {
  const W = w + 2; // one background pixel all around
  const img = new Int8Array((h + 2) * W);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[(y + 1) * W + x + 1] = mask[y * w + x] ? OBJECT : 0;
  const out: number[][][] = [];
  for (let y = 1; y <= h; y++) {
    const row = y * W;
    let lnbd = 0; // column of the last border pixel met on this row (0 = the frame)
    let prev = 0;
    for (let x = 1; x <= w; x++) {
      let p = img[row + x];
      if (p === prev) continue;
      let trace = false;
      if (prev === 0 && p === OBJECT)
        trace = img[row + lnbd] <= 0; // a start; positive: inside another component
      else if (p === 0 && prev >= 1 && prev !== OBJECT) lnbd = x - 1; // a hole: not traced, but it moves lnbd
      if (trace) {
        out.push(follow(img, W, row + x, x - 1, y - 1));
        p = img[row + x];
      }
      prev = p;
      if (prev !== 0 && prev !== OBJECT) lnbd = x;
    }
  }
  return out.reverse();
}

/** Follows the outer border from its first pixel `i0` at `(x, y)` (the topmost, then leftmost). */
function follow(img: Int8Array, W: number, i0: number, x: number, y: number): number[][] {
  const delta = Array.from({ length: 16 }, (_, k) => DY[k & 7] * W + DX[k & 7]);
  const pts: number[][] = [];
  // the first neighbour, clockwise from north-west
  let s = 4;
  let i1 = -1;
  do {
    s = (s - 1) & 7;
    i1 = i0 + delta[s];
    if (img[i1] !== 0) break;
  } while (s !== 4);
  if (s === 4) {
    img[i0] = RIGHT; // a single pixel
    return [[x, y]];
  }
  let i3 = i0;
  let last = s ^ 4;
  for (;;) {
    const sEnd = s;
    let i4: number;
    do i4 = i3 + delta[++s];
    while (img[i4] === 0); // the next border pixel, counter-clockwise from the one before
    s &= 7;
    if (s >= 1 && s <= sEnd)
      img[i3] = RIGHT; // the search went round past east: a right bound
    else if (img[i3] === OBJECT) img[i3] = LEFT;
    if (s !== last) {
      pts.push([x, y]);
      last = s;
    }
    x += DX[s];
    y += DY[s];
    if (i4 === i0 && i3 === i1) return pts;
    i3 = i4;
    s = (s + 4) & 7;
  }
}
