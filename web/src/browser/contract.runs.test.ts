import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { getJSON } from "../api";
import { expectClose, loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";

const py = loadContract(join(DATA, "contract-python"));
let rid = "";

beforeAll(async () => {
  rid = (await setupE2eRun()).rid;
});

describe("run requests match main's", () => {
  it("summary", async () => {
    const got = await getJSON(`/runs/${rid}/summary`);
    // paths differ by design (vpaths); everything else, fps (from ../meta.json) included, matches
    expectClose(got, py.find("GET /runs/{rid}/summary").json(), 1e-9, ["id", "path", "raw_dir", "mtime"]);
    expect((got as { available: { raw: boolean; pre: boolean } }).available).toMatchObject({ raw: true, pre: true });
  });
  it("reruns", async () => {
    expect(await getJSON(`/runs/${rid}/reruns`)).toEqual(py.find("GET /runs/{rid}/reruns").json());
  });
  it("params", async () => {
    expect(await getJSON("/params/defaults")).toEqual(py.find("GET /params/defaults").json());
    for (const d of ["7.5", "12"])
      expect(await getJSON(`/params/derived?diameter=${d}`)).toEqual(
        py.find(`GET /params/derived?diameter=${d}`).json(),
      );
  });
  it("tune requests answer 404", async () => {
    await expect(getJSON(`/runs/${rid}/autotune`)).rejects.toMatchObject({ status: 404 });
    await expect(getJSON(`/runs/${rid}/flags`)).rejects.toMatchObject({ status: 404 });
  });
});
