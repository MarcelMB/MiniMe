import { cpSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { getArray, getJSON } from "../../api";
import { writeNpy } from "../npy";
import { DATA, setupE2eRun } from "../testing/e2eRun";
import { settled } from "../testing/settled";

/** The e2e run with `cnmf/C.npy` replaced and/or `cnmf/YrA.npy` removed. */
const variant = (opts: { C?: Uint8Array; noYrA?: boolean }): string => {
  const run = join(mkdtempSync(join(tmpdir(), "traces-")), "t");
  cpSync(join(DATA, "processed/t"), run, { recursive: true });
  if (opts.C) writeFileSync(join(run, "cnmf/C.npy"), opts.C);
  if (opts.noYrA) rmSync(join(run, "cnmf/YrA.npy"));
  return run;
};

describe("trace routes on the e2e run", () => {
  let rid = "";
  beforeAll(async () => {
    rid = (await setupE2eRun(variant({ noYrA: true }))).rid;
  });

  it("answers the overview of a run's traces after computing it once", async () => {
    const a = await settled(() => getArray(`/runs/${rid}/traces/overview?bins=10`));
    expect(a.shape).toEqual([7, 2, 10]); // the e2e run's 7 units (15 before the Nov 2026 motion defaults)
    expect(a.data.length).toBe(7 * 2 * 10);
    // cached: the same request now answers at once
    expect((await getArray(`/runs/${rid}/traces/overview?bins=10`)).data).toEqual(a.data);
  });

  it("keeps a bin count above the frame count at one bin per frame", async () => {
    const a = await settled(() => getArray(`/runs/${rid}/traces/overview?bins=20000`));
    expect(a.shape).toEqual([7, 2, 300]);
  });

  it("checks kind, bins, unit and window like main", async () => {
    const bad = async (path: string, status: number, text: string) =>
      expect(getJSON(`/runs/${rid}${path}`)).rejects.toMatchObject({ status, message: text });
    await bad("/traces/overview?kind=X", 400, "kind must be C or YrA, got 'X'");
    await bad("/traces/0?kind=X", 400, "kind must be C or YrA, got 'X'");
    await bad("/traces/overview?bins=0", 400, "bins must be in 1..20000");
    await bad("/traces/overview?bins=20001", 400, "bins must be in 1..20000");
    await bad("/traces/overview?bins=abc", 422, "bins must be an integer");
    await bad("/traces/7", 400, "unit 7 out of range 0..6");
    await bad("/traces/-1", 400, "unit -1 out of range 0..6");
    await bad("/traces/x", 422, "unit must be an integer");
    await bad("/traces/0?start=200&stop=100", 400, "window 200:100 outside 0:300");
    await bad("/traces/0?stop=301", 400, "window 0:301 outside 0:300");
    await bad("/traces/0?start=1.5", 422, "start must be an integer");
    await bad("/traces/0?lowpass_hz=0", 400, "cutoff and frame rate must be > 0 (got 0.0 Hz, 30.0 Hz)");
    await bad("/traces/0?lowpass_hz=fast", 422, "lowpass_hz must be a number");
  });

  it("has no YrA traces without cnmf/YrA.npy", async () => {
    for (const path of ["/traces/overview?kind=YrA", "/traces/0?kind=YrA"])
      await expect(getJSON(`/runs/${rid}${path}`)).rejects.toMatchObject({
        status: 404,
        message: "YrA traces are not available for this run",
      });
  });

  it("windows a trace, with and without the low-pass", async () => {
    const plain = await getArray(`/runs/${rid}/traces/3`);
    expect(plain.shape).toEqual([300]);
    const part = await getArray(`/runs/${rid}/traces/3?start=100&stop=150`);
    expect(Array.from(part.data)).toEqual(Array.from(plain.data.slice(100, 150)));
    const smooth = await getArray(`/runs/${rid}/traces/3?start=100&stop=150&lowpass_hz=2`);
    expect(smooth.shape).toEqual([50]);
    expect(Array.from(smooth.data)).not.toEqual(Array.from(part.data));
  });

  it("answers an unknown run with 404", async () => {
    await expect(getJSON("/runs/000000000000/zooms")).rejects.toMatchObject({ status: 404 });
  });
});

describe("a run without units", () => {
  let rid = "";
  beforeAll(async () => {
    rid = (await setupE2eRun(variant({ C: writeNpy([0, 300], new Float32Array(0)) }))).rid;
  });

  it("has an empty overview and no zoom units, but still zoom windows", async () => {
    const o = await settled(() => getArray(`/runs/${rid}/traces/overview?bins=2000`));
    expect(o.shape).toEqual([0, 2, 300]);
    expect(await settled(() => getJSON(`/runs/${rid}/zooms`))).toEqual({
      units: [],
      windows: [{ start: 0, stop: 300 }],
    });
    await expect(getJSON(`/runs/${rid}/traces/0`)).rejects.toMatchObject({
      status: 400,
      message: "unit 0 out of range 0..-1",
    });
  });
});

describe("traces stored in Fortran order", () => {
  it("are refused rather than read transposed", async () => {
    // np.save of a transposed (K, T) array: the header says so, the rows are not contiguous
    const C = writeNpy([15, 300], new Float32Array(15 * 300));
    C.set(new TextEncoder().encode("True "), new TextDecoder("latin1").decode(C).indexOf("False"));
    const { rid } = await setupE2eRun(variant({ C }));
    for (const path of ["/traces/0", "/traces/overview?bins=10"])
      await expect(settled(() => getJSON(`/runs/${rid}${path}`))).rejects.toMatchObject({
        status: 500,
        message: "cnmf/C.npy is stored in Fortran order, which the browser app does not read",
      });
  });
});
