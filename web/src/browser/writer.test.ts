import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { fileAt, readBytes, readJSON } from "./dir";
import { Folders } from "./folders";
import { memoryKV } from "./kv";
import { readNpy, writeNpy } from "./npy";
import { readCsr } from "./npz";
import { Registry } from "./runs";
import { NodeDir } from "./testing/nodeDir";
import { type RunParts, sessionInit, sessionSeeds, unitsCsv, type WindowParts, writeRun } from "./writer";

const win = (t0: number, files: Record<string, Uint8Array>): WindowParts => ({
  t0,
  t1: t0 + 10,
  files: new Map(Object.entries(files)),
});
const pts = (...xy: number[]) => writeNpy([xy.length / 2, 2], Float64Array.from(xy));
const mask = (...m: number[]) => writeNpy([m.length], Uint8Array.from(m));
const npy = (name: string) => `${name}.npy`;
const withNpy = (w: WindowParts): WindowParts => ({ ...w, files: new Map([...w.files].map(([k, v]) => [npy(k), v])) });

describe("session-wide seeds and init", () => {
  it("keeps the earliest window's seed and drops later ones within 3 px", () => {
    const s = sessionSeeds(
      [
        win(0, { seeds_init: pts(10, 10, 40, 40), mask_pnr: mask(1, 0), seeds_merged: pts(10, 10) }),
        win(10, { seeds_init: pts(11, 12, 60, 60), mask_pnr: mask(1, 1), seeds_merged: pts(12, 11, 60, 60) }),
      ].map(withNpy),
    );
    expect(Array.from(s.init)).toEqual([10, 10, 40, 40, 60, 60]); // (11,12) is 2.2 px from (10,10)
    expect(Array.from(s.pnr)).toEqual([10, 10, 60, 60]);
    expect(Array.from(s.merged)).toEqual([10, 10, 60, 60]);
  });

  it("reads the engine's layout: (n, 3) int64 initial seeds, the PNR mask over them, a GMM mask over the survivors", () => {
    const init = writeNpy([3, 3], BigInt64Array.from([5, 5, 0, 20, 20, 1, 30, 30, 2].map(BigInt)));
    const w = withNpy(
      win(0, {
        seeds_init: init,
        mask_pnr: mask(1, 0, 1),
        mask_gmm: mask(1, 1), // two PNR survivors; main's seeds/pnr.npy ignores it
        seeds_merged: writeNpy([2, 2], BigInt64Array.from([5, 5, 30, 30].map(BigInt))),
      }),
    );
    const s = sessionSeeds([w]);
    expect(Array.from(s.init)).toEqual([5, 5, 20, 20, 30, 30]);
    expect(Array.from(s.pnr)).toEqual([5, 5, 30, 30]);
    expect(Array.from(s.merged)).toEqual([5, 5, 30, 30]);
  });

  it("keeps every seed of the first window, even two within 3 px (only later windows are thinned)", () => {
    const s = sessionSeeds(
      [
        win(0, { seeds_init: pts(10, 10, 11, 11), mask_pnr: mask(1, 1), seeds_merged: pts(10, 10, 11, 11) }),
        win(10, {
          seeds_init: pts(12, 12, 30, 30, 31, 31),
          mask_pnr: mask(1, 1, 1),
          seeds_merged: pts(30, 30, 31, 31),
        }),
      ].map(withNpy),
    );
    // (12,12) is within 3 px of the first window's seeds: dropped; the second window's own pair stays
    expect(Array.from(s.init)).toEqual([10, 10, 11, 11, 30, 30, 31, 31]);
    expect(Array.from(s.merged)).toEqual([10, 10, 11, 11, 30, 30, 31, 31]);
  });

  it("says which window lacks seeds", () => {
    expect(() => sessionSeeds([win(30, {})])).toThrow(/window 30/);
  });

  it("merges init footprints with cosine >= cos_thres", () => {
    const csr = (data: number[], indices: number[], indptr: number[]) => ({
      "A_merge_data.npy": writeNpy([data.length], Float32Array.from(data)),
      "A_merge_indices.npy": writeNpy([indices.length], Int32Array.from(indices)),
      "A_merge_indptr.npy": writeNpy([indptr.length], BigInt64Array.from(indptr.map(BigInt))),
    });
    const a = sessionInit(
      [
        { t0: 0, t1: 10, files: new Map(Object.entries(csr([1, 1], [0, 1], [0, 2]))) },
        { t0: 10, t1: 20, files: new Map(Object.entries(csr([1, 0.9, 1], [0, 1, 5], [0, 2, 3]))) },
      ],
      6,
      0.7,
    );
    expect(a.shape).toEqual([2, 6]); // window 2's first unit has cosine 0.998 with window 1's: dropped
    expect(Array.from(a.indices)).toEqual([0, 1, 5]);
  });

  it("keeps footprints of one window that overlap, and drops one that is below the threshold nowhere", () => {
    const csr = (data: number[], indices: number[], indptr: number[]) => ({
      "A_merge_data.npy": writeNpy([data.length], Float32Array.from(data)),
      "A_merge_indices.npy": writeNpy([indices.length], Int32Array.from(indices)),
      "A_merge_indptr.npy": writeNpy([indptr.length], Float64Array.from(indptr)),
    });
    // unit (0,1) vs (1,2): cosine 0.5 < 0.7, so both stay; units within a window are not compared
    const a = sessionInit(
      [{ t0: 0, t1: 10, files: new Map(Object.entries(csr([1, 1, 1, 1], [0, 1, 1, 2], [0, 2, 4]))) }],
      4,
      0.7,
    );
    expect(a.shape).toEqual([2, 4]);
    expect(Array.from(a.indptr)).toEqual([0, 2, 4]);
  });
});

it("writes units.csv with main's columns", () => {
  const a = {
    shape: [1, 4] as [number, number],
    data: new Float32Array([1, 1]),
    indices: new Int32Array([0, 3]),
    indptr: new Int32Array([0, 2]),
  };
  expect(unitsCsv(a, 2, 2, new Float32Array([5.5]))).toBe(
    "unit_id,centroid_h,centroid_w,size_px,peak\n0,0.5,0.5,2,5.5\n",
  );
});

it("writes a float32 peak the way pandas does and leaves an empty footprint's centroid blank", () => {
  const a = {
    shape: [2, 4] as [number, number],
    data: new Float32Array([1, 1]),
    indices: new Int32Array([0, 3]),
    indptr: new Int32Array([0, 2, 2]),
  };
  const rows = unitsCsv(a, 2, 2, new Float32Array([0.1, 3]))
    .trim()
    .split("\n");
  expect(rows[1]).toBe("0,0.5,0.5,2,0.1");
  expect(rows[2]).toBe("1,,,0,3.0");
});

// --- writeRun --------------------------------------------------------------------------

const H = 4;
const W = 4;
const T = 10;
const f32 = (shape: number[], ...v: number[]) => writeNpy(shape, Float32Array.from(v));
const i64 = (...v: number[]) => writeNpy([v.length], BigInt64Array.from(v.map(BigInt)));

function csrFiles(prefix: string, data: number[], indices: number[], indptr: number[]): [string, Uint8Array][] {
  return [
    [`${prefix}_data.npy`, writeNpy([data.length], Float64Array.from(data))],
    [`${prefix}_indices.npy`, i64(...indices)],
    [`${prefix}_indptr.npy`, i64(...indptr)],
  ];
}

/** One window of a 4 x 4 frame, two tiles, two units (pixels 0-1 and 14-15). */
function runParts(traces: Blob = new Blob([f32([2, T], ...Array.from({ length: 2 * T }, (_, i) => i))])): RunParts {
  const unitA = csrFiles("A_merge", [1, 1, 1, 1], [0, 1, 14, 15], [0, 2, 4]);
  const w: WindowParts = {
    t0: 0,
    t1: T,
    files: new Map<string, Uint8Array>([
      ["max_proj.npy", f32([H, W], ...Array.from({ length: H * W }, (_, i) => i))],
      ["block_max.npy", f32([1, H, W], ...new Array(H * W).fill(1))],
      ["rank.npy", i64(3, 5)],
      ["W.npy", f32([4], 1, 2, 3, 4)],
      ["V.npy", f32([4], 1, 2, 3, 4)],
      ["s.npy", f32([2], 1, 2)],
      ["mean.npy", f32([2], 1, 2)],
      ["k.npy", i64(1, 1)],
      ["sigma.npy", f32([2], 1, 2)],
      ["seeds_init.npy", writeNpy([2, 3], BigInt64Array.from([0, 0, 0, 3, 3, 1].map(BigInt)))],
      ["mask_pnr.npy", mask(1, 1)],
      ["mask_gmm.npy", mask(1, 1)],
      ["seeds_merged.npy", writeNpy([2, 2], BigInt64Array.from([0, 0, 3, 3].map(BigInt)))],
      ...unitA,
    ]),
  };
  return {
    session: { window_frames: 3000, stitch: { cos_thres: 0.7 } },
    params: { tile: 2, raw_dir: "picked/raw" },
    report: { stages: [{ stage: "pass1", seconds: 1 }], counts: { units_final: 2 }, wall_total: 3, windows: [] },
    H,
    W,
    T,
    tile: 2,
    windows: [w],
    motion: {
      shifts: writeNpy([T, 2], new Float64Array(2 * T)),
      template: f32([H, W], ...new Array(H * W).fill(0)),
      quality: f32([T], ...new Array(T).fill(1)),
      params: { motion_max_shift: 20 },
    },
    stitch: new Map<string, Uint8Array>([
      ...csrFiles("A", [1, 1, 1, 1], [0, 1, 14, 15], [0, 2, 4]),
      [
        "sn.npy",
        writeNpy(
          [H * W],
          Float64Array.from({ length: H * W }, (_, i) => i / 10),
        ),
      ],
      ["bright.npy", writeNpy([2], Float64Array.from([7, 9]))],
    ]),
    stream: new Map<string, Uint8Array | Blob>([
      [
        "b.npy",
        writeNpy(
          [H * W],
          Float64Array.from({ length: H * W }, (_, i) => i * 2),
        ),
      ],
      ...csrFiles("Amerged", [1, 1], [0, 1], [0, 2]),
    ]),
    traces,
    progress: ['{"stage":"pass1","event":"start","t":1}', '{"stage":"pass1","event":"end","t":2,"seconds":1}'],
  };
}

async function listAll(d: NodeDir, prefix = ""): Promise<string[]> {
  const out: string[] = [];
  for (const e of await d.entries())
    if (e.kind === "directory") out.push(...(await listAll((await d.dir(e.name)) as NodeDir, `${prefix}${e.name}/`)));
    else out.push(`${prefix}${e.name}`);
  return out.sort();
}

const SPEC_FILES = [
  "params.json",
  "report.json",
  "progress.jsonl",
  "motion/meta.json",
  "motion/arrays.npz",
  "factors/meta.json",
  "factors/max_proj.npy",
  "factors/sn.npy",
  ...["W", "V", "s", "mean", "k", "rank", "block_max", "max_proj"].map((k) => `factors/w00/${k}.npy`),
  "cnmf/A.npz",
  "cnmf/C.npy",
  "cnmf/units.csv",
  "cnmf/b.npy",
  "cnmf/Amerged.npz",
  "seeds/init.npy",
  "seeds/pnr.npy",
  "seeds/merged.npy",
  "init/A.npz",
].sort();

describe("writeRun", () => {
  const newDir = () => new NodeDir(mkdtempSync(join(tmpdir(), "writer-")));

  it("writes the run folder of spec section 3, and a Registry opens it", async () => {
    const top = newDir();
    const out = await top.dir("run", { create: true });
    await writeRun(out, runParts());
    expect(await listAll(out as NodeDir)).toEqual(SPEC_FILES);

    const meta = await readJSON<Record<string, unknown>>(out, "factors/meta.json");
    expect(meta).toEqual({ H, W, T, tile: 2, ranks: [3, 5], windows: [{ t0: 0, t1: T }] });
    const params = await readJSON<Record<string, unknown>>(out, "params.json");
    expect(params).toMatchObject({ tile: 2, raw_dir: "picked/raw", source: "browser", window_frames: 3000 });
    const motion = await readJSON<Record<string, unknown>>(out, "motion/meta.json");
    expect(motion).toEqual({ format_version: 1, params: { motion_max_shift: 20 }, raw_hash: "" });

    // footprints, peak and the (H, W) images
    const A = await readCsr(await fileAt(out, "cnmf/A.npz"));
    expect(A.shape).toEqual([2, H * W]);
    expect(Array.from(A.indices)).toEqual([0, 1, 14, 15]);
    expect((await (await fileAt(out, "cnmf/units.csv")).text()).split("\n")[1]).toBe("0,0.0,0.5,2,7.0");
    const b = readNpy(await readBytes(out, "cnmf/b.npy"));
    expect(b.shape).toEqual([H, W]);
    expect(b.data).toBeInstanceOf(Float32Array);
    expect(readNpy(await readBytes(out, "cnmf/C.npy")).shape).toEqual([2, T]);
    expect(readNpy(await readBytes(out, "factors/sn.npy")).shape).toEqual([H, W]);
    expect(Array.from(readNpy(await readBytes(out, "factors/max_proj.npy")).data)).toEqual(
      Array.from({ length: H * W }, (_, i) => i),
    );
    expect((await readCsr(await fileAt(out, "cnmf/Amerged.npz"))).shape).toEqual([1, H * W]);
    expect(readNpy(await readBytes(out, "seeds/pnr.npy")).shape).toEqual([2, 2]);
    expect((await readCsr(await fileAt(out, "init/A.npz"))).shape).toEqual([2, H * W]);
    const progress = await (await out.file("progress.jsonl")).text();
    expect(progress.trim().split("\n")).toHaveLength(2);

    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(top, "picked");
    const run = await new Registry(folders, memoryKV()).open(`${name}/run`);
    expect(run.meta.T).toBe(T);
    expect(run.report.counts.units_final).toBe(2);
  });

  it("writes report.json last, so a run that failed halfway is never listed", async () => {
    const top = newDir();
    const out = await top.dir("run", { create: true });
    const failing = new (class extends Blob {
      stream(): ReadableStream<Uint8Array> {
        throw new Error("the OPFS file is gone");
      }
      arrayBuffer(): Promise<ArrayBuffer> {
        return Promise.reject(new Error("the OPFS file is gone"));
      }
    })(["x"]);
    await expect(writeRun(out, runParts(failing))).rejects.toThrow(/gone/);
    const files = await listAll(out as NodeDir);
    expect(files).not.toContain("report.json");
    expect(files).toContain("params.json"); // the rest was written before the failure
    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(top, "picked");
    expect(await new Registry(folders, memoryKV()).find(name)).toEqual([]);
  });

  it("names the missing output of a stage that gave none", async () => {
    const out = await newDir().dir("run", { create: true });
    const r = runParts();
    r.stitch.delete("sn.npy");
    await expect(writeRun(out, r)).rejects.toThrow(/sn\.npy/);
  });
});
