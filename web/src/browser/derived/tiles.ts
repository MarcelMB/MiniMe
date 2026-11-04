export type Tile = { y0: number; y1: number; x0: number; x1: number };

/** `tiles.tile_grid`: the tiles of an (h, w) frame, row-major, the last row and column partial. */
export function tileGrid(h: number, w: number, tile: number): Tile[] {
  const out: Tile[] = [];
  for (let y0 = 0; y0 < h; y0 += tile)
    for (let x0 = 0; x0 < w; x0 += tile) out.push({ y0, y1: Math.min(h, y0 + tile), x0, x1: Math.min(w, x0 + tile) });
  return out;
}

/** `frames.tile_map`: (h, w) image with each tile filled with its value (e.g. its rank). */
export function tileMap(h: number, w: number, tile: number, values: number[]): Float32Array {
  const out = new Float32Array(h * w);
  const grid = tileGrid(h, w, tile);
  if (grid.length !== values.length) throw new Error(`${values.length} values for ${grid.length} tiles`);
  grid.forEach((t, i) => {
    for (let y = t.y0; y < t.y1; y++) out.fill(values[i], y * w + t.x0, y * w + t.x1);
  });
  return out;
}
