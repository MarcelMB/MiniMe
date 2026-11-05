import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { Computing, getArray, getJSON } from "../../api";
import { linspaceInt, percentile, toU8 } from "../derived/contrast";
import { writeNpy } from "../npy";
import { csrNpz } from "../npz";
import { state } from "../state";
import { loadContract } from "../testing/contract";
import { DATA, setupE2eRun } from "../testing/e2eRun";
import { NodeDir } from "../testing/nodeDir";
import { settled } from "../testing/settled";
import { windowDir } from "../writer";

const py = loadContract(join(DATA, "contract-python"));

type Units = {
  stage: string;
  units: { id: number; centroid: unknown[]; size_px: number; peak: number | null; polygons: number[][][] }[];
};

/** The e2e run, changed by `edit` (given the copy's folder). */
const variant = (edit: (run: string) => void = () => undefined): string => {
  const run = join(mkdtempSync(join(tmpdir(), "view-")), "t");
  cpSync(join(DATA, "processed/t"), run, { recursive: true });
  edit(run);
  return run;
};
/** The folder a `setupE2eRun` picked, to change files after the run is open. */
const topOf = async (name: string): Promise<string> => ((await state.folders.resolve(name)) as NodeDir).path;
const notFound = (path: string, message: string) =>
  expect(getJSON(path)).rejects.toMatchObject({ status: 404, message });

describe("view routes on the e2e run", () => {
  let rid = "";
  beforeAll(async () => {
    rid = (await setupE2eRun()).rid;
  });

  it("takes ?rerun= on the data routes and 404s for a rerun that has no report", async () => {
    const a = await getArray(`/runs/${rid}/image/background?rerun=r1`);
    expect(a.shape).toEqual((await getArray(`/runs/${rid}/image/background`)).shape);
    await notFound(`/runs/${rid}/units?rerun=nope`, `run ${rid} has no finished rerun 'nope'`);
  });

  it("answers the four images as (H, W) float32, the noise one only for browser runs", async () => {
    for (const name of ["max_proj", "background", "rank"]) {
      const a = await getArray(`/runs/${rid}/image/${name}`);
      expect([a.shape, a.dtype]).toEqual([[64, 64], "float32"]);
    }
    await notFound(`/runs/${rid}/image/noise`, "noise is not available for this run: factors/sn.npy");
    await notFound(`/runs/${rid}/image/bogus`, "unknown image 'bogus' (one of max_proj, background, rank, noise)");
  });

  it("answers the motion as (T, 3)", async () => {
    expect((await getArray(`/runs/${rid}/motion`)).shape).toEqual([300, 3]);
  });

  it("computes a Python run's motion residual from its flat factors/block_max.npy, as main's", async () => {
    await expect(getArray(`/runs/${rid}/motion/residual`)).rejects.toBeInstanceOf(Computing);
    const a = await settled(() => getArray(`/runs/${rid}/motion/residual`));
    expect([a.shape, a.dtype]).toEqual([[11, 3], "float32"]);
    const want = py.find("GET /runs/{rid}/motion/residual").npy();
    for (let i = 0; i < 33; i++) expect(Math.abs(a.data[i] - want.data[i])).toBeLessThan(1e-4);
  });

  it("answers seeds and units with one entry per footprint", async () => {
    const seeds = await getJSON<Record<string, number[][]>>(`/runs/${rid}/seeds`);
    expect(Object.keys(seeds)).toEqual(["init", "pnr", "merged"]);
    expect(seeds.init.length).toBe(66);
    expect(seeds.merged[0]).toHaveLength(2);
    const fin = await getJSON<Units>(`/runs/${rid}/units`);
    expect(fin.units.map((u) => u.id)).toEqual(Array.from({ length: 15 }, (_, k) => k));
    expect(fin.units.every((u) => typeof u.peak === "number")).toBe(true);
    const init = await getJSON<Units>(`/runs/${rid}/units?stage=init`);
    expect([init.stage, init.units.length, init.units[0].peak]).toEqual(["init", 17, null]);
  });

  it("checks the stage, the source and the frame number like main", async () => {
    await expect(getJSON(`/runs/${rid}/units?stage=x`)).rejects.toMatchObject({
      status: 400,
      message: "stage must be final or init, got 'x'",
    });
    const bad = (path: string, status: number, message: string) =>
      expect(getJSON(`/runs/${rid}${path}`)).rejects.toMatchObject({ status, message });
    await bad("/frame/0?source=gray", 400, "unknown source 'gray' (one of raw, pre, recon)");
    await bad("/frame/300?source=raw", 400, "frame 300 out of range 0..299");
    await bad("/frame/-1?source=raw", 400, "frame -1 out of range 0..299");
    await bad("/frame/abc?source=raw", 422, "t must be an integer");
    await bad("/frame/1.5?source=raw", 422, "t must be an integer");
    await bad("/frame/0", 404, "recon frames is not available for this run"); // the default source
    await bad("/frame/0?source=recon", 404, "recon frames is not available for this run"); // Python factors
  });

  it("answers the preprocessed frames from the flat tile store of a Python run", async () => {
    await expect(getArray(`/runs/${rid}/frame/10?source=pre`)).rejects.toBeInstanceOf(Computing);
    const a = await settled(() => getArray(`/runs/${rid}/frame/10?source=pre`));
    expect([a.shape, a.dtype]).toEqual([[64, 64], "uint8"]);
    const b = await getArray(`/runs/${rid}/frame/11?source=pre`); // the contrast is cached now
    expect(Array.from(b.data)).not.toEqual(Array.from(a.data));
    expect((await getJSON<{ available: { pre: boolean } }>(`/runs/${rid}/summary`)).available.pre).toBe(true);
  });

  it("computes the contrast once, then answers frames at once", async () => {
    await expect(getArray(`/runs/${rid}/frame/10?source=raw`)).rejects.toBeInstanceOf(Computing);
    const a = await settled(() => getArray(`/runs/${rid}/frame/10?source=raw`));
    expect([a.shape, a.dtype]).toEqual([[64, 64], "uint8"]);
    const doc = await (await state.cache(rid)).loadJSON<Record<string, number[]>>("contrast.json", {});
    expect(doc.raw[1]).toBeGreaterThan(doc.raw[0]);
    const b = await getArray(`/runs/${rid}/frame/11?source=raw`); // no Computing: the contrast is cached
    expect(Array.from(b.data)).not.toEqual(Array.from(a.data));
  });

  it("answers an unknown run with 404", async () => {
    for (const path of ["image/rank", "motion", "seeds", "units", "frame/0?source=raw"])
      await expect(getJSON(`/runs/000000000000/${path}`)).rejects.toMatchObject({ status: 404 });
  });
});

describe("a run folder without its optional files", () => {
  let rid = "";
  let name = "";
  beforeAll(async () => {
    const run = variant((d) => {
      for (const dir of ["seeds", "init", "tiles", "motion"]) rmSync(join(d, dir), { recursive: true });
      for (const f of ["factors/max_proj.npy", "factors/block_max.npy", "cnmf/b.npy"]) rmSync(join(d, f));
    });
    ({ rid, name } = await setupE2eRun(run));
    // setupE2eRun points raw_dir into the picked folder: a Python run holds a disk path
    const pj = join(await topOf(name), "processed/t/params.json");
    writeFileSync(pj, JSON.stringify({ ...JSON.parse(readFileSync(pj, "utf8")), raw_dir: "/data/raw" }));
  });

  it("answers 404 'not available for this run', never 500", async () => {
    const r = `/runs/${rid}`;
    await notFound(`${r}/seeds`, "the seeds is not available for this run");
    await notFound(`${r}/units?stage=init`, "the init footprints is not available for this run");
    await notFound(`${r}/motion`, "the motion is not available for this run: motion/arrays.npz");
    await notFound(`${r}/image/max_proj`, "max_proj is not available for this run: factors/max_proj.npy");
    await notFound(`${r}/image/background`, "background is not available for this run: cnmf/b.npy");
    await notFound(`${r}/motion/residual`, "the motion residual is not available for this run");
    await notFound(`${r}/frame/0?source=pre`, "pre frames is not available for this run");
    await notFound(`${r}/frame/0?source=raw`, "raw videos not found in /data/raw");
  });

  it("still answers what the folder holds", async () => {
    const r = `/runs/${rid}`;
    expect((await getArray(`${r}/image/rank`)).shape).toEqual([64, 64]);
    expect((await getJSON<Units>(`${r}/units`)).units).toHaveLength(15);
  });
});

describe("other broken run folders", () => {
  it("has no raw frames when the picked raw folder holds no videos", async () => {
    const { rid, name } = await setupE2eRun();
    rmSync(join(await topOf(name), "raw"), { recursive: true });
    await notFound(`/runs/${rid}/frame/0?source=raw`, `raw videos not found in ${name}/raw`);
  });

  it("keeps the 403 of a raw folder that has to be reconnected", async () => {
    const { rid, name } = await setupE2eRun();
    // the raw videos in a root of their own, whose permission is gone
    const rawRoot = mkdtempSync(join(tmpdir(), "rawroot-"));
    cpSync(join(DATA, "raw"), join(rawRoot, "raw"), { recursive: true });
    const dir = new NodeDir(rawRoot);
    const root = await state.folders.add(dir, "picked");
    const pj = join(await topOf(name), "processed/t/params.json");
    writeFileSync(pj, JSON.stringify({ ...JSON.parse(readFileSync(pj, "utf8")), raw_dir: `${root}/raw` }));
    dir.permission = async () => false;
    await expect(getJSON(`/runs/${rid}/frame/0?source=raw`)).rejects.toMatchObject({
      status: 403,
      message: expect.stringMatching(/^Reconnect the folder /),
    });
  });

  it("refuses raw frames of another size than the run's", async () => {
    const { rid } = await setupE2eRun(
      variant((d) => {
        const f = join(d, "factors/meta.json");
        writeFileSync(f, JSON.stringify({ ...JSON.parse(readFileSync(f, "utf8")), H: 32 }));
      }),
    );
    await expect(getJSON(`/runs/${rid}/frame/0?source=raw`)).rejects.toMatchObject({
      status: 400,
      message: "the raw videos' frames are 64x64, the run's are 32x64",
    });
  });

  it("answers 500 with the file name for an image that is not H * W", async () => {
    const { rid } = await setupE2eRun(
      variant((d) => writeFileSync(join(d, "factors/max_proj.npy"), writeNpy([2, 2], new Float32Array(4)))),
    );
    await expect(getJSON(`/runs/${rid}/image/max_proj`)).rejects.toMatchObject({
      status: 500,
      message: "factors/max_proj.npy holds 4 values, not H * W = 4096",
    });
  });

  it("says what is wrong with block maxima that do not fit the run, with 500, once computed", async () => {
    const { rid } = await setupE2eRun(
      variant((d) => {
        const w0 = join(d, "factors", windowDir(0));
        mkdirSync(w0, { recursive: true });
        writeFileSync(join(w0, "block_max.npy"), writeNpy([2, 8, 8], new Uint8Array(128)));
      }),
    );
    await expect(getArray(`/runs/${rid}/motion/residual`)).rejects.toBeInstanceOf(Computing);
    await expect(settled(() => getArray(`/runs/${rid}/motion/residual`))).rejects.toMatchObject({
      status: 500,
      message: "factors/w00/block_max.npy is shaped (2, 8, 8), not (n, 64, 64)",
    });
  });

  it("has no unit peaks without a peak for every unit", async () => {
    for (const csv of ["unit_id,size_px\n0,5\n", "unit_id,peak\n0,1.5\n"]) {
      const { rid } = await setupE2eRun(variant((d) => writeFileSync(join(d, "cnmf/units.csv"), csv)));
      await notFound(
        `/runs/${rid}/units`,
        "the unit peaks is not available for this run: cnmf/units.csv has no peak for every unit",
      );
    }
    const { rid } = await setupE2eRun(variant((d) => rmSync(join(d, "cnmf/units.csv"))));
    await notFound(`/runs/${rid}/units`, "the unit peaks is not available for this run: cnmf/units.csv");
  });
});

describe("a run without units", () => {
  let rid = "";
  beforeAll(async () => {
    const empty = new Uint8Array(
      await (
        await csrNpz({
          shape: [0, 64 * 64],
          data: new Float32Array(0),
          indices: new Int32Array(0),
          indptr: Int32Array.of(0),
        })
      ).arrayBuffer(),
    );
    rid = (
      await setupE2eRun(
        variant((d) => {
          writeFileSync(join(d, "cnmf/A.npz"), empty);
          writeFileSync(join(d, "init/A.npz"), empty);
          writeFileSync(join(d, "cnmf/units.csv"), "unit_id,centroid_h,centroid_w,size_px,peak\n");
        }),
      )
    ).rid;
  });

  it("answers an empty unit list for both stages", async () => {
    expect(await getJSON(`/runs/${rid}/units`)).toEqual({ stage: "final", units: [] });
    expect(await getJSON(`/runs/${rid}/units?stage=init`)).toEqual({ stage: "init", units: [] });
  });
});

describe("a unit without pixels", () => {
  it("has no centroid and no outline, and does not stop the others", async () => {
    const a = await csrNpz({
      shape: [2, 64 * 64],
      data: Float32Array.of(1),
      indices: Int32Array.of(65),
      indptr: Int32Array.of(0, 0, 1),
    });
    const bytes = new Uint8Array(await a.arrayBuffer());
    const { rid } = await setupE2eRun(
      variant((d) => {
        writeFileSync(join(d, "cnmf/A.npz"), bytes);
        writeFileSync(join(d, "cnmf/units.csv"), "unit_id,peak\n0,0\n1,2.5\n");
      }),
    );
    const { units } = await getJSON<Units>(`/runs/${rid}/units`);
    expect(units).toEqual([
      { id: 0, centroid: [null, null], size_px: 0, peak: 0, polygons: [] },
      { id: 1, centroid: [1, 1], size_px: 1, peak: 2.5, polygons: [[[1, 1]]] },
    ]);
  });
});

describe("a units.csv written on Windows", () => {
  it("reads the peaks of CRLF lines (pandas' to_csv there)", async () => {
    const a = await csrNpz({
      shape: [2, 64 * 64],
      data: Float32Array.of(1, 1),
      indices: Int32Array.of(65, 130),
      indptr: Int32Array.of(0, 1, 2),
    });
    const bytes = new Uint8Array(await a.arrayBuffer());
    const { rid } = await setupE2eRun(
      variant((d) => {
        writeFileSync(join(d, "cnmf/A.npz"), bytes);
        writeFileSync(join(d, "cnmf/units.csv"), "unit_id,size_px,peak\r\n0,1,1.5\r\n1,1,2.5\r\n");
      }),
    );
    const { units } = await getJSON<Units>(`/runs/${rid}/units`);
    expect(units.map((u) => u.peak)).toEqual([1.5, 2.5]);
  });
});

/** The e2e run as a browser run writes it: `factors/meta.json` with its windows and `factors/w00/`
 * (constant factors, W = 1, V[t] = t, mean = 10; the Python run's block maxima) and the Python
 * run's tile files moved to `tiles/w00/`. */
const browserVariant = (): string =>
  variant((d) => {
    const meta = JSON.parse(readFileSync(join(d, "factors/meta.json"), "utf8"));
    writeFileSync(
      join(d, "factors/meta.json"),
      JSON.stringify({ ...meta, ranks: [1, 1, 1, 1], windows: [{ t0: 0, t1: 300 }] }),
    );
    const w0 = join(d, "factors", windowDir(0));
    mkdirSync(w0, { recursive: true });
    writeFileSync(join(w0, "W.npy"), writeNpy([4096], new Float32Array(4096).fill(1)));
    writeFileSync(
      join(w0, "V.npy"),
      writeNpy(
        [1200],
        Float32Array.from({ length: 1200 }, (_, i) => i % 300),
      ),
    );
    writeFileSync(join(w0, "mean.npy"), writeNpy([4096], new Float32Array(4096).fill(10)));
    writeFileSync(join(w0, "rank.npy"), writeNpy([4], BigInt64Array.of(1n, 1n, 1n, 1n)));
    cpSync(join(d, "factors/block_max.npy"), join(w0, "block_max.npy"));
    mkdirSync(join(d, "tiles", windowDir(0)));
    for (const f of readdirSync(join(d, "tiles")).filter((n) => n.startsWith("tile_")))
      renameSync(join(d, "tiles", f), join(d, "tiles", windowDir(0), f));
  });

describe("view routes on a browser run (the factors and tile store per window)", () => {
  let rid = "";
  beforeAll(async () => {
    rid = (await setupE2eRun(browserVariant())).rid;
  });

  it("answers recon frames from the window's factors: frame t is 10 + t, scaled by the contrast", async () => {
    await expect(getArray(`/runs/${rid}/frame/3?source=recon`)).rejects.toBeInstanceOf(Computing);
    const a = await settled(() => getArray(`/runs/${rid}/frame/3?source=recon`));
    expect([a.shape, a.dtype]).toEqual([[64, 64], "uint8"]);
    // 50 frames spread over the run set the contrast, each of 4096 equal pixels: 10 + t for each t
    const ts = linspaceInt(299, 50);
    const sorted = Float32Array.from(ts.flatMap((t) => Array<number>(64 * 64).fill(10 + t))).sort();
    const [lo, hi] = [percentile(sorted, 1), percentile(sorted, 99.9)];
    expect(new Set(a.data)).toEqual(new Set(toU8(Float32Array.of(13), lo, hi)));
    const doc = await (await state.cache(rid)).loadJSON<Record<string, number[]>>("contrast.json", {});
    expect(doc.recon).toEqual([lo, hi]);
    const b = await getArray(`/runs/${rid}/frame/200?source=recon`); // no Computing now
    expect(new Set(b.data)).toEqual(new Set(toU8(Float32Array.of(210), lo, hi)));
    expect(a.data[0]).toBeLessThan(b.data[0]);
  });

  it("answers the default source, recon, the same way", async () => {
    const a = await settled(() => getArray(`/runs/${rid}/frame/3`));
    expect([a.shape, a.dtype]).toEqual([[64, 64], "uint8"]);
  });

  it("answers pre frames from tiles/w00 byte for byte as main does from the flat store it was moved from", async () => {
    const got = await settled(() => getArray(`/runs/${rid}/frame/100?source=pre`));
    const want = py.find("GET /runs/{rid}/frame/100?source=pre").npy();
    expect(got.shape).toEqual(want.shape);
    expect(Array.from(got.data)).toEqual(Array.from(want.data));
  });

  it("computes the motion residual from the windows' block maxima on the first request, then answers it", async () => {
    await expect(getArray(`/runs/${rid}/motion/residual`)).rejects.toBeInstanceOf(Computing);
    const a = await settled(() => getArray(`/runs/${rid}/motion/residual`));
    expect([a.shape, a.dtype]).toEqual([[11, 3], "float32"]);
    expect(await (await state.cache(rid)).has("residual.npy")).toBe(true);
    const again = await getArray(`/runs/${rid}/motion/residual`); // cached: no Computing
    expect(Array.from(again.data)).toEqual(Array.from(a.data));
    const want = py.find("GET /runs/{rid}/motion/residual").npy();
    for (let i = 0; i < 33; i++) expect(Math.abs(a.data[i] - want.data[i])).toBeLessThan(1e-4);
  });
});
