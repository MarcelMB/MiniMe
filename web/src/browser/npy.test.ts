import { describe, expect, it } from "vitest";
import { readNpy, readNpyHeader, writeNpy } from "./npy";

describe("npy", () => {
  it("round-trips float32, int32, uint8 and int64", () => {
    const f = writeNpy([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
    expect(readNpy(f)).toEqual({ shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) });
    expect(Array.from(readNpy(writeNpy([2], new Int32Array([-1, 7]))).data)).toEqual([-1, 7]);
    expect(Array.from(readNpy(writeNpy([3], new Uint8Array([0, 9, 255]))).data)).toEqual([0, 9, 255]);
    expect(Array.from(readNpy(writeNpy([2], new BigInt64Array([3n, 4n]))).data)).toEqual([3, 4]);
  });

  it("reads a Fortran-order file (np.save of a transposed array) in C order", () => {
    // (2, 3) matrix [[1, 2, 3], [4, 5, 6]] stored column by column
    const fortran = (bytes: Uint8Array) => {
      const out = bytes.slice();
      out.set(new TextEncoder().encode("True "), new TextDecoder("latin1").decode(bytes).indexOf("False"));
      return out;
    };
    const m = readNpy(fortran(writeNpy([2, 3], new Float32Array([1, 4, 2, 5, 3, 6]))));
    expect(m).toEqual({ shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) });
    // (2, 3, 2): element [i, j, k] sits at i + 2j + 6k and must come out at 6i + 2j + k
    const f = new Int32Array(12);
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 3; j++) for (let k = 0; k < 2; k++) f[i + 2 * j + 6 * k] = 6 * i + 2 * j + k;
    expect(Array.from(readNpy(fortran(writeNpy([2, 3, 2], f))).data)).toEqual(Array.from({ length: 12 }, (_, i) => i));
    // one dimension: nothing to reorder
    expect(Array.from(readNpy(fortran(writeNpy([3], new Uint8Array([7, 8, 9])))).data)).toEqual([7, 8, 9]);
  });

  it("aligns the data to 64 bytes and reports where it starts", () => {
    const h = readNpyHeader(writeNpy([5, 7], new Float32Array(35)));
    expect(h).toEqual({ shape: [5, 7], descr: "<f4", offset: 128 });
  });
});
