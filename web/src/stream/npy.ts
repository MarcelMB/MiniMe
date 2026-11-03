/** Read a little-endian `.npy` (version 1–3, C order) into a typed array; int64 comes back
 * as `Float64Array` (exact below 2^53, which indices and counts are). */
export function readNpy(bytes: Uint8Array): {
  shape: number[];
  data: Float32Array | Float64Array | Int32Array | Uint8Array;
} {
  const magic = String.fromCharCode(...bytes.subarray(1, 6));
  if (bytes[0] !== 0x93 || magic !== "NUMPY") throw new Error("not a .npy file");
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = (major === 1 ? 10 : 12) + headerLen;
  const header = new TextDecoder().decode(bytes.subarray(major === 1 ? 10 : 12, start));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const order = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || order !== "False" || shapeText === undefined) throw new Error(`unsupported .npy header: ${header}`);
  const shape = shapeText
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length)
    .map(Number);
  const n = shape.reduce((a, b) => a * b, 1);
  // a slice: the data is rarely aligned to its element size inside the file
  const body = bytes.slice(start);
  const dv = new DataView(body.buffer, body.byteOffset, body.byteLength);
  const read = <T extends Float32Array | Float64Array | Int32Array>(
    out: T,
    get: (i: number) => number,
    size: number,
  ): T => {
    for (let i = 0; i < n; i++) out[i] = get(i * size);
    return out;
  };
  switch (descr) {
    case "<f4":
      return { shape, data: read(new Float32Array(n), (o) => dv.getFloat32(o, true), 4) };
    case "<f8":
      return { shape, data: read(new Float64Array(n), (o) => dv.getFloat64(o, true), 8) };
    case "<i4":
      return { shape, data: read(new Int32Array(n), (o) => dv.getInt32(o, true), 4) };
    case "<i8":
      return { shape, data: read(new Float64Array(n), (o) => Number(dv.getBigInt64(o, true)), 8) };
    case "|u1":
    case "|b1":
      return { shape, data: body.subarray(0, n) };
    default:
      throw new Error(`unsupported .npy dtype ${descr}`);
  }
}
