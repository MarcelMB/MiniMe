// The tile store of a window run on the pool: one file ("slab") per pass-1 chunk instead of one file
// per tile and chunk. A worker has a single sync access handle open (WebKit allows about 250 in all,
// which 100 tiles on 3 workers would exceed), and pass 2 reads every slab from several workers at
// once through `File` snapshots, which need no exclusive handle.
//
// A slab holds its chunk's frames in blocks of `FLUSH` frames (the last one shorter); a block is the
// tiles one after the other, each `frames x pixels` uint8, as `Pass1Engine.take_tiles()` hands them
// out. Tile `k` of the window is its pieces of every slab, in chunk order.

export const FLUSH = 100; // frames buffered before they are appended to the store

export const slabName = (chunk: number) => `slab_${String(chunk).padStart(2, "0")}`;

/** Byte ranges of tile `k` in a slab of `n` frames, in frame order; `pixels` are the tiles' sizes. */
export function pieces(n: number, pixels: number[], k: number): { at: number; len: number }[] {
  let before = 0;
  let all = 0;
  pixels.forEach((p, j) => {
    all += p;
    if (j < k) before += p;
  });
  const out: { at: number; len: number }[] = [];
  let base = 0;
  for (let f = 0; f < n; f += FLUSH) {
    const nb = Math.min(FLUSH, n - f);
    out.push({ at: base + nb * before, len: nb * pixels[k] });
    base += nb * all;
  }
  return out;
}

/** Tile `k`'s bytes (frames in order) from the chunks' slabs; `frames[c]` is chunk `c`'s frame count. */
export function readTile(
  slabs: File[],
  frames: number[],
  pixels: number[],
  k: number,
  reader: FileReaderSync,
): Uint8Array {
  const out = new Uint8Array(frames.reduce((n, f) => n + f, 0) * pixels[k]);
  let at = 0;
  frames.forEach((n, c) => {
    for (const p of pieces(n, pixels, k)) {
      out.set(new Uint8Array(reader.readAsArrayBuffer(slabs[c].slice(p.at, p.at + p.len))), at);
      at += p.len;
    }
  });
  return out;
}
