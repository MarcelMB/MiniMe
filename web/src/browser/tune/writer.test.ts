import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { writeNpy } from "../npy";
import type { RunInfo } from "../runs";
import { NodeDir } from "../testing/nodeDir";
import { rerunReport, writeRerun } from "./writer";

function sourceRun(): RunInfo {
  const top = mkdtempSync(join(tmpdir(), "src-"));
  const d = new NodeDir(top);
  return {
    id: "s",
    vpath: "root/processed/t",
    dir: d,
    factors: d,
    sourceRun: null,
    params: { raw_dir: "root/raw", fps: 20 },
    report: {},
    mtime: 0,
    stamp: "1",
    meta: { H: 2, W: 2, T: 6, tile: 2, ranks: [1], windows: [{ t0: 0, t1: 6 }] },
  };
}

describe("writeRerun", () => {
  it("writes a run folder with params.json first and report.json last", async () => {
    const src = sourceRun();
    for (const f of [
      "factors/meta.json",
      "factors/max_proj.npy",
      "factors/sn.npy",
      "motion/meta.json",
      "motion/arrays.npz",
      "seeds/init.npy",
      "init/A.npz",
    ]) {
      const [dir, name] = f.split("/");
      await (await src.dir.dir(dir, { create: true })).write(
        name,
        name.endsWith(".json") ? "{}" : new Uint8Array([1, 2, 3]),
      );
    }
    const out = new NodeDir(mkdtempSync(join(tmpdir(), "rerun-")));
    const csr = (prefix: string) =>
      new Map<string, Uint8Array>([
        [`${prefix}_data.npy`, writeNpy([2], new Float32Array([1, 1]))],
        [`${prefix}_indices.npy`, writeNpy([2], new Int32Array([0, 3]))],
        [`${prefix}_indptr.npy`, writeNpy([3], new Int32Array([0, 1, 2]))],
      ]);
    const stitch = new Map([
      ...csr("A"),
      ["bright.npy", writeNpy([2], new Float32Array([5, 6]))],
      ["sn.npy", writeNpy([4], new Float64Array(4))],
    ]);
    const stream = new Map<string, Uint8Array | Blob>([
      ...csr("Amerged"),
      ["b.npy", writeNpy([4], new Float32Array(4))],
    ]);
    await writeRerun(out, src, {
      session: { pass1: { tile: 2 } },
      params: { sparse_snr: 0.5 },
      sourceRun: src.vpath,
      rawDir: "root/raw",
      fps: 20,
      H: 2,
      W: 2,
      T: 6,
      tile: 2,
      windows: [{ t0: 0, t1: 6 }],
      stitch,
      stream,
      traces: new Blob([writeNpy([2, 6], new Float32Array(12))]),
      yra: new Blob([writeNpy([2, 6], new Float32Array(12))]),
      report: { stages: [], counts: { units_final: 2 } },
      progress: ['{"stage":"window2","event":"start","t":1}'],
    });
    const params = JSON.parse(readFileSync(join(out.path, "params.json"), "utf8"));
    expect(params).toMatchObject({
      sparse_snr: 0.5,
      source: "browser",
      source_run: "root/processed/t",
      raw_dir: "root/raw",
      fps: 20,
    });
    for (const f of [
      "cnmf/A.npz",
      "cnmf/C.npy",
      "cnmf/YrA.npy",
      "cnmf/units.csv",
      "cnmf/b.npy",
      "cnmf/Amerged.npz",
      "factors/meta.json",
      "factors/max_proj.npy",
      "factors/sn.npy",
      "motion/arrays.npz",
      "seeds/init.npy",
      "init/A.npz",
      "progress.jsonl",
      "report.json",
    ])
      expect(existsSync(join(out.path, f)), f).toBe(true);
    expect(existsSync(join(out.path, "factors/w00"))).toBe(false); // the windows' factors stay with the source run
    const rep = rerunReport(
      [{ stage: "window2", seconds: 1 }],
      [],
      { units_in: 4, units_out: 2 },
      { kept: 2, merged: 2 },
      6,
      2,
      2,
      3.5,
      "root/processed/t",
      100,
    );
    expect(rep).toMatchObject({
      counts: { units_windows: 4, units_stitched: 2, units_final: 2, windows: 0 },
      wall_total: 3.5,
      source_run: "root/processed/t",
      where: "browser",
      wasm_heap_peak_mb: 100,
    });
  });
});
