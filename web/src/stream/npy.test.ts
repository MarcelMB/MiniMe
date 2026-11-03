import { describe, expect, it } from "vitest";
import { readNpy } from "./npy";

/** A version-1 `.npy` as numpy writes it (header padded to a multiple of 64 bytes). */
function npy(descr: string, shape: number[], body: Uint8Array): Uint8Array {
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${dims}), }`;
  while ((10 + header.length + 1) % 64) header += " ";
  header += "\n";
  const out = new Uint8Array(10 + header.length + body.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 0xff, header.length >> 8]);
  out.set(new TextEncoder().encode(header), 10);
  out.set(body, 10 + header.length);
  return out;
}

describe("readNpy", () => {
  it("reads float32, float64, int32 and int64 arrays with their shape", () => {
    const f32 = npy("<f4", [2, 2], new Uint8Array(new Float32Array([1, 2.5, -3, 4]).buffer));
    expect(readNpy(f32)).toEqual({ shape: [2, 2], data: new Float32Array([1, 2.5, -3, 4]) });
    const f64 = npy("<f8", [3], new Uint8Array(new Float64Array([0.1, 0.2, 0.3]).buffer));
    expect(readNpy(f64)).toEqual({ shape: [3], data: new Float64Array([0.1, 0.2, 0.3]) });
    const i32 = npy("<i4", [2], new Uint8Array(new Int32Array([-7, 9]).buffer));
    expect(readNpy(i32)).toEqual({ shape: [2], data: new Int32Array([-7, 9]) });
    const i64 = npy("<i8", [2], new Uint8Array(new BigInt64Array([0n, 1234567890123n]).buffer));
    expect(readNpy(i64)).toEqual({ shape: [2], data: new Float64Array([0, 1234567890123]) });
  });
  it("reads uint8 and bool", () => {
    expect(readNpy(npy("|u1", [3], new Uint8Array([1, 2, 3]))).data).toEqual(new Uint8Array([1, 2, 3]));
    expect(readNpy(npy("|b1", [2], new Uint8Array([1, 0]))).data).toEqual(new Uint8Array([1, 0]));
  });
  it("rejects other files", () => {
    expect(() => readNpy(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))).toThrow("not a .npy file");
  });
});
