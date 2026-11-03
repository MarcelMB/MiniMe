import { describe, expect, it } from "vitest";
import { orderAvis, rate } from "./order";

describe("orderAvis", () => {
  it("keeps numbered AVIs in numeric order", () => {
    const names = ["10.avi", "2.avi", "notes.txt", "1.AVI", "x.avi"].map((name) => ({ name }));
    expect(orderAvis(names).map((f) => f.name)).toEqual(["1.AVI", "2.avi", "10.avi"]);
  });
});

describe("rate", () => {
  it("gives fps and time left", () => {
    expect(rate(100, 300, 10)).toEqual({ fps: 10, eta: 20 });
    expect(rate(0, 300, 0).eta).toBe(Infinity);
  });
});
