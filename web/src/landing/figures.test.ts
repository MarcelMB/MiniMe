import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { FIGURES, type FigureName, figureSrc } from "./figures";

const PUBLIC = resolve(import.meta.dirname, "../../public");

describe("landing figures", () => {
  it("ships a light and a dark file for every figure", () => {
    for (const name of Object.keys(FIGURES) as FigureName[]) {
      const { light, dark } = figureSrc(name);
      expect(existsSync(resolve(PUBLIC, light)), light).toBe(true);
      expect(existsSync(resolve(PUBLIC, dark)), dark).toBe(true);
    }
  });
});
