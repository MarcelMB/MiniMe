/** Zip archives as Blobs: read (stored or deflated, zip64) and write (stored, zip64 when an
 * entry or offset passes 4 GiB). Writing never copies entry data: the result is a Blob of
 * headers and the given Blobs. */
const u16 = (v: DataView, o: number) => v.getUint16(o, true);
const u32 = (v: DataView, o: number) => v.getUint32(o, true);
const u64 = (v: DataView, o: number) => Number(v.getBigUint64(o, true));

async function bytes(b: Blob, start: number, end: number): Promise<DataView> {
  return new DataView(await b.slice(start, end).arrayBuffer());
}

export async function readZip(blob: Blob): Promise<Map<string, Blob>> {
  const tail = Math.min(blob.size, 65_557);
  const t = await bytes(blob, blob.size - tail, blob.size);
  let eocd = -1;
  for (let i = tail - 22; i >= 0; i--)
    if (u32(t, i) === 0x06054b50) {
      eocd = i;
      break;
    }
  if (eocd < 0) throw new Error("not a zip file");
  let count = u16(t, eocd + 10);
  let cdOff = u32(t, eocd + 16);
  let cdSize = u32(t, eocd + 12);
  if (cdOff === 0xffffffff || count === 0xffff) {
    const loc = eocd - 20; // zip64 end-of-central-directory locator
    const z64 = await bytes(blob, Number(t.getBigUint64(loc + 8, true)), blob.size);
    count = u64(z64, 32);
    cdSize = u64(z64, 40);
    cdOff = u64(z64, 48);
  }
  const cd = await bytes(blob, cdOff, cdOff + cdSize);
  const out = new Map<string, Blob>();
  let o = 0;
  for (let n = 0; n < count; n++) {
    const method = u16(cd, o + 10);
    let csize = u32(cd, o + 20);
    let usize = u32(cd, o + 24);
    const nameLen = u16(cd, o + 28);
    const extraLen = u16(cd, o + 30);
    const commentLen = u16(cd, o + 32);
    let local = u32(cd, o + 42);
    const name = new TextDecoder().decode(new Uint8Array(cd.buffer, o + 46, nameLen));
    for (let e = o + 46 + nameLen; e < o + 46 + nameLen + extraLen; ) {
      const id = u16(cd, e);
      const len = u16(cd, e + 2);
      if (id === 0x0001) {
        let f = e + 4;
        if (usize === 0xffffffff) {
          usize = u64(cd, f);
          f += 8;
        }
        if (csize === 0xffffffff) {
          csize = u64(cd, f);
          f += 8;
        }
        if (local === 0xffffffff) local = u64(cd, f);
      }
      e += 4 + len;
    }
    const lh = await bytes(blob, local, local + 30);
    const dataStart = local + 30 + u16(lh, 26) + u16(lh, 28);
    const raw = blob.slice(dataStart, dataStart + csize);
    out.set(
      name,
      method === 0 ? raw : await new Response(raw.stream().pipeThrough(new DecompressionStream("deflate-raw"))).blob(),
    );
    o += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

/** CRC-32 (zlib) of a Blob, streamed. */
async function crc32(b: Blob): Promise<number> {
  const table = new Uint32Array(256).map((_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c;
  });
  let c = 0xffffffff;
  const reader = b.stream().getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (let i = 0; i < value.length; i++) c = table[(c ^ value[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

/** `forceZip64` writes the zip64 records for every entry and the end of the archive whatever the
 * sizes (what a 4 GiB archive gets), so the layout can be tested on small data. */
export async function zipBlob(
  entries: { name: string; data: Blob }[],
  opts: { forceZip64?: boolean } = {},
): Promise<Blob> {
  const parts: BlobPart[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const { name, data } of entries) {
    const nameBytes = new TextEncoder().encode(name);
    const crc = await crc32(data);
    const big = !!opts.forceZip64 || data.size >= 0xffffffff || offset >= 0xffffffff;
    // local header: sizes in a zip64 extra field when big
    const lx = big ? 20 : 0;
    const lh = new DataView(new ArrayBuffer(30 + nameBytes.length + lx));
    lh.setUint32(0, 0x04034b50, true);
    lh.setUint16(4, big ? 45 : 20, true);
    lh.setUint32(14, crc, true);
    lh.setUint32(18, big ? 0xffffffff : data.size, true);
    lh.setUint32(22, big ? 0xffffffff : data.size, true);
    lh.setUint16(26, nameBytes.length, true);
    lh.setUint16(28, lx, true);
    new Uint8Array(lh.buffer).set(nameBytes, 30);
    if (big) {
      const e = 30 + nameBytes.length;
      lh.setUint16(e, 0x0001, true);
      lh.setUint16(e + 2, 16, true);
      lh.setBigUint64(e + 4, BigInt(data.size), true);
      lh.setBigUint64(e + 12, BigInt(data.size), true);
    }
    // central header: sizes and offset in a zip64 extra field when big
    const cx = big ? 28 : 0;
    const ch = new DataView(new ArrayBuffer(46 + nameBytes.length + cx));
    ch.setUint32(0, 0x02014b50, true);
    ch.setUint16(4, big ? 45 : 20, true);
    ch.setUint16(6, big ? 45 : 20, true);
    ch.setUint32(16, crc, true);
    ch.setUint32(20, big ? 0xffffffff : data.size, true);
    ch.setUint32(24, big ? 0xffffffff : data.size, true);
    ch.setUint16(28, nameBytes.length, true);
    ch.setUint16(30, cx, true);
    ch.setUint32(42, big ? 0xffffffff : offset, true);
    new Uint8Array(ch.buffer).set(nameBytes, 46);
    if (big) {
      const e = 46 + nameBytes.length;
      ch.setUint16(e, 0x0001, true);
      ch.setUint16(e + 2, 24, true);
      ch.setBigUint64(e + 4, BigInt(data.size), true);
      ch.setBigUint64(e + 12, BigInt(data.size), true);
      ch.setBigUint64(e + 20, BigInt(offset), true);
    }
    parts.push(lh.buffer, data);
    central.push(new Uint8Array(ch.buffer));
    offset += lh.byteLength + data.size;
  }
  const cdSize = central.reduce((s, c) => s + c.length, 0);
  const z64 = !!opts.forceZip64 || offset >= 0xffffffff || entries.length >= 0xffff;
  const tail: ArrayBuffer[] = [];
  if (z64) {
    const r = new DataView(new ArrayBuffer(56));
    r.setUint32(0, 0x06064b50, true);
    r.setBigUint64(4, 44n, true);
    r.setUint16(12, 45, true);
    r.setUint16(14, 45, true);
    r.setBigUint64(24, BigInt(entries.length), true);
    r.setBigUint64(32, BigInt(entries.length), true);
    r.setBigUint64(40, BigInt(cdSize), true);
    r.setBigUint64(48, BigInt(offset), true);
    const l = new DataView(new ArrayBuffer(20));
    l.setUint32(0, 0x07064b50, true);
    l.setBigUint64(8, BigInt(offset + cdSize), true);
    l.setUint32(16, 1, true);
    tail.push(r.buffer, l.buffer);
  }
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, z64 ? 0xffff : entries.length, true);
  end.setUint16(10, z64 ? 0xffff : entries.length, true);
  end.setUint32(12, cdSize, true);
  end.setUint32(16, z64 ? 0xffffffff : offset, true);
  return new Blob([...parts, ...central, ...tail, end.buffer]);
}
