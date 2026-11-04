import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { RunCache } from "../cache";
import { readNpy, writeNpy } from "../npy";
import type { RunInfo } from "../runs";
import { loadContract } from "../testing/contract";
import { DATA } from "../testing/e2eRun";
import { NodeDir } from "../testing/nodeDir";
import { windowDir } from "../writer";
import { computeResidual } from "./residual";

const PYTHON_RUN = join(DATA, "processed/t");
const npy = (rel: string) => readNpy(new Uint8Array(readFileSync(join(PYTHON_RUN, rel))));

/** The Python e2e run's block maxima (11 blocks) as a browser run's windows: the first
 * `split[0]` blocks in window 0, the next in window 1, and so on; `factors/max_proj.npy` as the
 * session's. */
function browserRun(split: number[], params: Record<string, unknown>): RunInfo {
  const path = mkdtempSync(join(tmpdir(), "resid-"));
  const blocks = npy("factors/block_max.npy");
  const [H, W] = [64, 64];
  mkdirSync(join(path, "factors"));
  const mp = npy("factors/max_proj.npy").data;
  writeFileSync(join(path, "factors/max_proj.npy"), writeNpy([H, W], Float32Array.from(mp)));
  let at = 0;
  split.forEach((n, w) => {
    mkdirSync(join(path, "factors", windowDir(w)));
    const part = (blocks.data as Uint8Array).slice(at * H * W, (at + n) * H * W);
    writeFileSync(join(path, "factors", windowDir(w), "block_max.npy"), writeNpy([n, H, W], part));
    at += n;
  });
  return {
    id: `resid-${Math.random()}`,
    vpath: "x",
    dir: new NodeDir(path),
    params,
    report: {},
    meta: {
      H,
      W,
      T: 300,
      tile: 32,
      ranks: [1],
      windows: split.map((_, w) => ({ t0: w * 100, t1: (w + 1) * 100 })),
    },
    mtime: 0,
    stamp: "s",
  };
}

const cache = () => new RunCache(new NodeDir(mkdtempSync(join(tmpdir(), "resid-cache-"))));
const python = loadContract(join(DATA, "contract-python")).find("GET /runs/{rid}/motion/residual").npy();
const flat = JSON.parse(readFileSync(join(PYTHON_RUN, "params.json"), "utf8")) as Record<string, unknown>;

describe("computeResidual", () => {
  it("gives main's residual (dy, dx, correlation per block) for the same blocks and max projection", async () => {
    const c = cache();
    await computeResidual(browserRun([11], flat), c);
    const got = await c.loadNpy("residual.npy");
    expect(got.shape).toEqual([11, 3]);
    expect(got.data).toBeInstanceOf(Float32Array);
    for (let i = 0; i < 33; i++) expect(Math.abs(got.data[i] - python.data[i]), `value ${i}`).toBeLessThan(1e-4);
  });

  it("concatenates the windows' blocks in time order: the split changes nothing", async () => {
    const [one, split] = [cache(), cache()];
    await computeResidual(browserRun([11], flat), one);
    await computeResidual(browserRun([4, 1, 6], flat), split);
    expect(Array.from((await split.loadNpy("residual.npy")).data)).toEqual(
      Array.from((await one.loadNpy("residual.npy")).data),
    );
    expect((await split.loadNpy("residual.npy")).shape).toEqual([11, 3]);
  });

  it("takes the pass-1 parameters of a browser run's params.json from its pass1 object", async () => {
    const c = cache();
    await computeResidual(browserRun([11], { ...flat, motion_max_shift: 1, pass1: flat }), c);
    const a = (await c.loadNpy("residual.npy")).data;
    const d = cache();
    await computeResidual(browserRun([11], flat), d);
    expect(Array.from(a)).toEqual(Array.from((await d.loadNpy("residual.npy")).data)); // pass1 wins, not the flat 1 px
  });

  it("takes a run without window records as one window, w00", async () => {
    const r = browserRun([11], flat);
    r.meta.windows = undefined;
    const c = cache();
    await computeResidual(r, c);
    expect((await c.loadNpy("residual.npy")).shape).toEqual([11, 3]);
  });
});
