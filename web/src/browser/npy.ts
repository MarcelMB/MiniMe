type Typed = Float32Array | Float64Array | Int32Array | Uint8Array;

/** The elements of a Fortran-order array (first index fastest) in C order. */
function toCOrder<T extends Typed>(data: T, shape: number[]): T {
  if (shape.length < 2) return data;
  const out = new (data.constructor as new (n: number) => T)(data.length);
  const stride = shape.map((_, i) => shape.slice(0, i).reduce((a, b) => a * b, 1));
  const at = shape.map(() => 0);
  for (let c = 0; c < data.length; c++) {
    out[c] = data[at.reduce((f, a, i) => f + a * stride[i], 0)];
    for (let i = shape.length - 1; i >= 0 && ++at[i] === shape[i]; i--) at[i] = 0;
  }
  return out;
}

/** Read a little-endian `.npy` (version 1–3) into a typed array in C order (a Fortran-order file,
 * which `np.save` writes for a transposed array, is reordered); int64 comes back as
 * `Float64Array` (exact below 2^53, which indices and counts are). */
export function readNpy(bytes: Uint8Array): { shape: number[]; data: Typed } {
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
  if (!descr || !order || shapeText === undefined) throw new Error(`unsupported .npy header: ${header}`);
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
  const done = (data: Typed) => ({ shape, data: order === "True" ? toCOrder(data, shape) : data });
  switch (descr) {
    case "<f4":
      return done(read(new Float32Array(n), (o) => dv.getFloat32(o, true), 4));
    case "<f8":
      return done(read(new Float64Array(n), (o) => dv.getFloat64(o, true), 8));
    case "<i4":
      return done(read(new Int32Array(n), (o) => dv.getInt32(o, true), 4));
    case "<i8":
      return done(read(new Float64Array(n), (o) => Number(dv.getBigInt64(o, true)), 8));
    case "|u1":
    case "|b1":
      return done(body.subarray(0, n));
    default:
      throw new Error(`unsupported .npy dtype ${descr}`);
  }
}

const DESCR = new Map<unknown, string>([
  [Float32Array, "<f4"],
  [Float64Array, "<f8"],
  [Int32Array, "<i4"],
  [Uint8Array, "|u1"],
  [BigInt64Array, "<i8"],
]);

function header(descr: string, shape: number[]): Uint8Array {
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let text = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${dims}), }`;
  const pad = 64 - ((10 + text.length + 1) % 64);
  text = `${text}${" ".repeat(pad % 64)}\n`;
  const out = new Uint8Array(10 + text.length);
  out.set([0x93, ...new TextEncoder().encode("NUMPY"), 1, 0]);
  new DataView(out.buffer).setUint16(8, text.length, true);
  out.set(new TextEncoder().encode(text), 10);
  return out;
}

/** A version-1 `.npy`, data 64-byte aligned (numpy's own layout). */
export function writeNpy(
  shape: number[],
  data: Float32Array | Float64Array | Int32Array | Uint8Array | BigInt64Array,
): Uint8Array {
  const descr = DESCR.get(data.constructor);
  if (!descr) throw new Error("unsupported array type");
  const h = header(descr, shape);
  const out = new Uint8Array(h.length + data.byteLength);
  out.set(h);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), h.length);
  return out;
}

/** A 0-d bytes scalar (`|S<n>`), e.g. scipy's `format` entry `b'csr'`. */
export function writeNpyBytes(s: string): Uint8Array {
  const b = new TextEncoder().encode(s);
  const h = header(`|S${b.length}`, []);
  const out = new Uint8Array(h.length + b.length);
  out.set(h);
  out.set(b, h.length);
  return out;
}

/** Shape, dtype and data offset of a `.npy` from its first bytes (4 KB are always enough). */
export function readNpyHeader(bytes: Uint8Array): { shape: number[]; descr: string; offset: number } {
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const len = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const text = new TextDecoder().decode(bytes.subarray(start, start + len));
  const descr = /'descr':\s*'([^']+)'/.exec(text)?.[1] ?? "";
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(text)?.[1] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  return { shape, descr, offset: start + len };
}
