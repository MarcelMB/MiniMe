import { describe, expect, it } from "vitest";
import { orderAvis } from "./order";

describe("orderAvis", () => {
  it("keeps numbered AVIs in numeric order", () => {
    const names = ["10.avi", "2.avi", "notes.txt", "1.AVI", "x.avi"].map((name) => ({ name }));
    expect(orderAvis(names).map((f) => f.name)).toEqual(["1.AVI", "2.avi", "10.avi"]);
  });
});
