import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { writeNpy } from "../npy";
import type { RunInfo } from "../runs";
import { NodeDir } from "../testing/nodeDir";
import { loadWindowFiles, windowBytes, windowsOf } from "./factors";

function fakeRun(skip?: string): RunInfo {
  const top = mkdtempSync(join(tmpdir(), "factors-"));
  const w = join(top, "factors/w00");
  mkdirSync(w, { recursive: true });
  for (const name of ["W", "V", "mean", "rank", "block_max", "max_proj", "sigma", "energy"])
    if (name !== skip) writeFileSync(join(w, `${name}.npy`), writeNpy([4], new Float32Array(4)));
  const dir = new NodeDir(top);
  return {
    id: "x",
    vpath: "x",
    dir,
    factors: dir,
    sourceRun: null,
    params: {},
    report: {},
    mtime: 0,
    stamp: "s",
    meta: { H: 2, W: 2, T: 10, tile: 2, ranks: [1], windows: [{ t0: 0, t1: 10 }] },
  };
}

describe("window factor files", () => {
  it("loads a window's eight files and measures their bytes", async () => {
    const r = fakeRun();
    const w = await loadWindowFiles(r, 0);
    expect([w.i, w.t0, w.t1, [...w.files.keys()].sort()]).toEqual([
      0,
      0,
      10,
      ["V.npy", "W.npy", "block_max.npy", "energy.npy", "max_proj.npy", "mean.npy", "rank.npy", "sigma.npy"],
    ]);
    expect(await windowBytes(r)).toBe(8 * (128 + 16));
    expect(windowsOf(r)).toEqual([{ t0: 0, t1: 10 }]);
  });
  it.each(["sigma", "energy"])("refuses a run made before auto-tune (no %s.npy)", async (name) => {
    await expect(loadWindowFiles(fakeRun(name), 0)).rejects.toMatchObject({
      status: 400,
      message: `this run was made before auto-tune: run it again to tune or rerun it (factors/w00/${name}.npy is missing)`,
    });
  });
});
