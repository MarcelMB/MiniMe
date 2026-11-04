import { expect, it } from "vitest";
import { joinVpath, parentVpath, splitVpath } from "./vpath";

it("splits, joins and finds parents of vpaths", () => {
  expect(splitVpath("synth/raw")).toEqual({ root: "synth", rel: "raw" });
  expect(splitVpath("Browser storage")).toEqual({ root: "Browser storage", rel: "" });
  expect(joinVpath("synth", "processed", "t")).toBe("synth/processed/t");
  expect(parentVpath("synth/raw")).toBe("synth");
  expect(parentVpath("synth")).toBeNull();
});
