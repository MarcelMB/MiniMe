import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Folders } from "../folders";
import { memoryKV } from "../kv";
import { writeNpy } from "../npy";
import { csrNpz, npzBlob } from "../npz";
import { Registry } from "../runs";
import { DATA } from "../testing/e2eRun";
import { FakeTuneWorker, out } from "../testing/fakeTuneWorker";
import { NodeDir } from "../testing/nodeDir";
import { autotune, type Ctx, RerunCancelled, rerun } from "./driver";

/** A root with `raw/0.avi` (any bytes: the fake stream worker never reads them) and a browser run
 * `processed/t` with one window's factors. */
async function fixture(hw = 2) {
  const top = mkdtempSync(join(tmpdir(), "drv-"));
  mkdirSync(join(top, "raw"));
  writeFileSync(join(top, "raw/0.avi"), new Uint8Array(16));
  const run = join(top, "processed/t");
  mkdirSync(join(run, "factors/w00"), { recursive: true });
  for (const d of ["cnmf", "motion", "seeds", "init"]) mkdirSync(join(run, d));
  for (const n of ["W", "V", "mean", "sigma", "energy", "rank", "block_max", "max_proj"])
    writeFileSync(join(run, `factors/w00/${n}.npy`), writeNpy([4], new Float32Array(4)));
  writeFileSync(
    join(run, "factors/meta.json"),
    JSON.stringify({ H: hw, W: hw, T: 6, tile: 2, ranks: [1], windows: [{ t0: 0, t1: 6 }] }),
  );
  writeFileSync(join(run, "factors/max_proj.npy"), writeNpy([2, 2], new Float32Array(4)));
  writeFileSync(join(run, "factors/sn.npy"), writeNpy([2, 2], new Float32Array(4)));
  writeFileSync(
    join(run, "motion/arrays.npz"),
    new Uint8Array(await (await npzBlob({ shifts: writeNpy([6, 2], new Float32Array(12)) })).arrayBuffer()),
  );
  writeFileSync(join(run, "motion/meta.json"), "{}");
  const folders = new Folders(memoryKV(), async () => null);
  const name = await folders.add(new NodeDir(top), "picked");
  const session = {
    pass1: { H: hw, W: hw, tile: 2, seeds_wnd: 3, seeds_stp: 1 },
    stitch: { cos_thres: 0.7 },
    stream: { H: hw, W: hw },
  };
  writeFileSync(
    join(run, "params.json"),
    JSON.stringify({ sparse_snr: 1.0, raw_dir: `${name}/raw`, fps: 20, source: "browser", ...session }),
  );
  writeFileSync(join(run, "report.json"), JSON.stringify({ stages: [], counts: {} }));
  const registry = new Registry(folders, memoryKV(), async () => new NodeDir(mkdtempSync(join(tmpdir(), "cache-"))));
  return { top, run, name, folders, registry };
}

const csr = (prefix: string) =>
  new Map<string, Uint8Array>([
    [`${prefix}_data.npy`, writeNpy([1], new Float32Array([1]))],
    [`${prefix}_indices.npy`, writeNpy([1], new Int32Array([0]))],
    [`${prefix}_indptr.npy`, writeNpy([2], new Int32Array([0, 1]))],
  ]);

function ctxFor(
  f: Awaited<ReturnType<typeof fixture>>,
  opts: { streamFails?: boolean } = {},
): Ctx & { posted: unknown[] } {
  const posted: unknown[] = [];
  return {
    registry: f.registry,
    folders: f.folders,
    maxWorkers: 2,
    scratch: "test",
    posted,
    makeTuneWorker: () => new FakeTuneWorker(() => out(5, 0)) as unknown as Worker,
    makeStreamWorker: () => {
      const w = {
        onmessage: null as null | ((e: MessageEvent) => void),
        onerror: null,
        terminate() {},
        postMessage(m: unknown) {
          posted.push(m);
          setTimeout(() => {
            if (opts.streamFails) return w.onmessage?.({ data: { type: "error", message: "boom" } } as MessageEvent);
            w.onmessage?.({ data: { type: "stage", stage: "stream", event: "end", t: 2, seconds: 1 } } as MessageEvent);
            const files = new Map<string, Uint8Array | Blob>([
              ...csr("Amerged"),
              ["b.npy", writeNpy([4], new Float32Array(4))],
              ["C.npy", new Blob([writeNpy([1, 6], new Float32Array(6))])],
              ["YrA.npy", new Blob([writeNpy([1, 6], new Float32Array(6))])],
              ["summary.json", new TextEncoder().encode('{"kept":1,"merged":1}')],
            ]);
            w.onmessage?.({ data: { type: "done", files, seconds: 1 } } as MessageEvent);
          }, 0);
        },
      };
      return w as unknown as Worker;
    },
    stitch: () => ({
      add_window: () => 1,
      add_mean_noise: () => undefined,
      run: () => 1,
      finish: () =>
        new Map([
          ...csr("A"),
          ["bright.npy", writeNpy([1], new Float32Array([1]))],
          ["sn.npy", writeNpy([4], new Float64Array(4))],
          ["ymean.npy", writeNpy([4], new Float64Array(4))],
          ["summary_stitch.json", new TextEncoder().encode('{"units_in":1,"units_out":1}')],
        ]),
      free: () => undefined,
    }),
  };
}

const progress = (cancelled = () => false) => {
  const events: string[] = [];
  return {
    events,
    p: {
      stage: (e: { stage: string; event: string }) => events.push(`${e.stage}:${e.event}`),
      live: () => undefined,
      log: () => undefined,
      cancelled,
    },
  };
};

describe("rerun", () => {
  it("runs the windows, the stitch and the streaming pass into reruns/<name>", async () => {
    const f = await fixture();
    const run = await f.registry.open(`${f.name}/processed/t`);
    const { events, p } = progress();
    const ctx = ctxFor(f);
    const out = await rerun(ctx, run, { sparse_snr: 0.5 }, "r1", p);
    expect((out as NodeDir).path).toBe(join(f.run, "reruns/r1"));
    expect(events).toEqual([
      "window2:start",
      "window2:end",
      "stitch:start",
      "stitch:end",
      "stream:end",
      "write_outputs:start",
      "write_outputs:end",
    ]);
    const msg = ctx.posted[0] as { inputs: File[]; avis: File[] };
    expect(msg.inputs.map((x) => x.name).sort()).toEqual([
      "A_data.npy",
      "A_indices.npy",
      "A_indptr.npy",
      "params.json",
      "shifts.npy",
      "sn.npy",
      "ymean.npy",
    ]);
    expect(msg.avis.map((x) => x.name)).toEqual(["0.avi"]);
    const rr = await f.registry.open(`${f.name}/processed/t/reruns/r1`);
    expect(rr.params).toMatchObject({ sparse_snr: 0.5, source_run: `${f.name}/processed/t` });
    expect(rr.report).toMatchObject({ source_run: `${f.name}/processed/t` });
  });

  it("refuses a factor parameter, the reserved name and a bad name before touching the folder", async () => {
    const f = await fixture();
    const run = await f.registry.open(`${f.name}/processed/t`);
    const ctx = ctxFor(f);
    await expect(rerun(ctx, run, { tile: 32 }, "r1", progress().p)).rejects.toThrow(
      "tile: not a window-2 parameter (needs a full run)",
    );
    await expect(rerun(ctx, run, {}, "auto", progress().p)).rejects.toThrow(
      "run name 'auto' is reserved for auto-tune; pick another name",
    );
    await expect(rerun(ctx, run, {}, "a/b", progress().p)).rejects.toThrow(
      "invalid run name 'a/b': use a single, plain directory name",
    );
    expect(ctx.posted).toEqual([]);
  });

  it("writes error.txt when the streaming pass fails or the raw folder is gone", async () => {
    const f = await fixture();
    const run = await f.registry.open(`${f.name}/processed/t`);
    await expect(rerun(ctxFor(f, { streamFails: true }), run, {}, "r2", progress().p)).rejects.toThrow("boom");
    expect(readFileSync(join(f.run, "reruns/r2/error.txt"), "utf8")).toContain("boom");
    rmSync(join(f.top, "raw"), { recursive: true });
    await expect(rerun(ctxFor(f), run, {}, "r3", progress().p)).rejects.toThrow("raw videos not found");
    expect(readFileSync(join(f.run, "reruns/r3/error.txt"), "utf8")).toContain("raw videos not found");
  });

  it("stops at a cancel and leaves error.txt saying so", async () => {
    const f = await fixture();
    const run = await f.registry.open(`${f.name}/processed/t`);
    await expect(rerun(ctxFor(f), run, {}, "r4", progress(() => true).p)).rejects.toBeInstanceOf(RerunCancelled);
    expect(readFileSync(join(f.run, "reruns/r4/error.txt"), "utf8")).toBe("cancelled");
  });
});

describe("autotune", () => {
  it("searches, writes reruns/auto with sweep.csv, unit_flags.csv and autotune.json", async () => {
    const f = await fixture(64);
    // the e2e run's A.npz / YrA.npy stand in for the fake run's final units
    cpSync(join(DATA, "processed/t/cnmf/A.npz"), join(f.run, "cnmf/A.npz"));
    cpSync(join(DATA, "processed/t/cnmf/YrA.npy"), join(f.run, "cnmf/YrA.npy"));
    const run = await f.registry.open(`${f.name}/processed/t`);
    const ctx = ctxFor(f);
    ctx.makeTuneWorker = () =>
      new FakeTuneWorker((_, s) =>
        out(10 - Math.round(10 * Math.abs(Math.log(Number(JSON.parse(s).cnmf.sparse_snr) / 0.5))), 1),
      ) as unknown as Worker;
    const { events, p } = progress();
    const dir = await autotune(ctx, run, { diameter: 7 }, p);
    expect((dir as NodeDir).path).toBe(join(f.run, "reruns/auto"));
    expect(events[0]).toBe("tune_run:start");
    expect(events).toContain("tune_sparse:start");
    expect(events).toContain("tune_rerun:start");
    expect(events.at(-1)).toBe("tune_rerun:end");
    expect(events.at(-2)).toBe("write_outputs:end");
    const doc = JSON.parse(readFileSync(join(f.run, "reruns/auto/autotune.json"), "utf8"));
    expect(doc).toMatchObject({
      strategy: "local",
      passes: 1,
      diameter: 7,
      diameter_source: "given",
      min_dist_frac: 0.4,
      tune_frames: 40000,
      segment: null,
      frames: 6,
      better_than_run: true,
    });
    expect(Object.keys(doc.proposed).sort()).toEqual([
      "init_merge_corr",
      "inita_corr",
      "loop_merge_corr",
      "merge_corr",
      "pnr_thres",
      "seeds_diff_thres",
      "sparse_snr",
    ]);
    expect(doc.proposed.sparse_snr).toBeCloseTo(0.51, 1);
    expect(doc.kept).toEqual([]); // the fake run left every size at its default
    expect(doc.pairs.run.n_near).toBeGreaterThanOrEqual(0);
    const csv = readFileSync(join(f.run, "reruns/auto/sweep.csv"), "utf8").trim().split("\n");
    expect(csv.length - 1).toBe(doc.n_trials);
    expect(readFileSync(join(f.run, "reruns/auto/unit_flags.csv"), "utf8")).toContain("unit_id,pnr");
    expect(readFileSync(join(f.run, "reruns/auto/params.json"), "utf8")).toContain('"source_run"');
  });

  it("reruns the run's own parameters when nothing beats the run", async () => {
    const f = await fixture(64);
    cpSync(join(DATA, "processed/t/cnmf/A.npz"), join(f.run, "cnmf/A.npz"));
    cpSync(join(DATA, "processed/t/cnmf/YrA.npy"), join(f.run, "cnmf/YrA.npy"));
    const run = await f.registry.open(`${f.name}/processed/t`);
    const ctx = ctxFor(f); // a constant score: no setting is better
    await autotune(ctx, run, { diameter: 7 }, progress().p);
    const doc = JSON.parse(readFileSync(join(f.run, "reruns/auto/autotune.json"), "utf8"));
    expect(doc.better_than_run).toBe(false);
    const params = JSON.parse(readFileSync(join(f.run, "reruns/auto/params.json"), "utf8"));
    expect(doc.derived.seeds_max_wnd).toBe(doc.run_values.seeds_max_wnd);
    expect(doc.derived.merge_min_dist).toBe(doc.run_values.merge_min_dist);
    expect(params.seeds_max_wnd ?? doc.run_values.seeds_max_wnd).toBe(doc.run_values.seeds_max_wnd);
    expect(params.merge_min_dist ?? doc.run_values.merge_min_dist).toBe(doc.run_values.merge_min_dist);
  });

  it("refuses a diameter the field of view cannot hold and estimates one when none is given", async () => {
    const f = await fixture();
    const run = await f.registry.open(`${f.name}/processed/t`);
    await expect(autotune(ctxFor(f), run, { diameter: 100 }, progress().p)).rejects.toThrow(
      "diameter must be a finite number of px of at most 2",
    );
    const one = {
      shape: [1, 4] as [number, number],
      data: new Float32Array([1]),
      indices: new Int32Array([0]),
      indptr: new Int32Array([0, 1]),
    };
    writeFileSync(join(f.run, "cnmf/A.npz"), new Uint8Array(await (await csrNpz(one)).arrayBuffer()));
    await expect(autotune(ctxFor(f), run, {}, progress().p)).rejects.toThrow("too few to estimate the cell diameter");
  });
});
