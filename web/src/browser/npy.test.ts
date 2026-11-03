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

  it("aligns the data to 64 bytes and reports where it starts", () => {
    const h = readNpyHeader(writeNpy([5, 7], new Float32Array(35)));
    expect(h).toEqual({ shape: [5, 7], descr: "<f4", offset: 128 });
  });
});
