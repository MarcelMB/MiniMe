import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { getArray, getJSON } from "../api";
import { expectClose, loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { settled } from "./testing/settled";

const py = loadContract(join(DATA, "contract-python"));
let rid = "";
beforeAll(async () => {
  rid = (await setupE2eRun()).rid;
});

describe("tune requests match main's", () => {
  it("reruns", async () => {
    const got = await getJSON<{ reruns: { name: string }[] }>(`/runs/${rid}/reruns`);
    expectClose(got, py.find("GET /runs/{rid}/reruns").json(), 1e-6, ["wall_total"]);
    expect(got.reruns.map((r) => r.name)).toEqual(["r1", "auto"]);
  });
  it("autotune", async () => {
    expectClose(await getJSON(`/runs/${rid}/autotune`), py.find("GET /runs/{rid}/autotune").json(), 1e-6, ["wall_s"]);
  });
  it("flags and pairs of the run and its reruns", async () => {
    for (const q of ["", "?rerun=auto"])
      expectClose(
        await settled(() => getJSON(`/runs/${rid}/flags${q}`)),
        py.find(`GET /runs/{rid}/flags${q}`).json(),
        1e-6,
      );
    for (const q of ["", "?rerun=r1"])
      expectClose(
        await settled(() => getJSON(`/runs/${rid}/pairs${q}`)),
        py.find(`GET /runs/{rid}/pairs${q}`).json(),
        1e-6,
      );
    // one computation wrote both caches: the other route answers at once, with no second scoring
    await expect(getJSON(`/runs/${rid}/flags?rerun=r1`)).resolves.toBeDefined();
    await expect(getJSON(`/runs/${rid}/pairs?rerun=auto`)).resolves.toBeDefined();
    await expect(getJSON(`/runs/${rid}/flags?rerun=nope`)).rejects.toMatchObject({
      status: 404,
      message: `run ${rid} has no finished rerun 'nope'`,
    });
  });
  it("the data routes take ?rerun=", async () => {
    expectClose(await getJSON(`/runs/${rid}/units?rerun=r1`), py.find("GET /runs/{rid}/units?rerun=r1").json(), 1e-5, [
      "polygons",
    ]);
    expectClose(
      await getJSON(`/runs/${rid}/seeds?rerun=auto`),
      py.find("GET /runs/{rid}/seeds?rerun=auto").json(),
      1e-6,
    );
    expectClose(
      await settled(() => getJSON(`/runs/${rid}/zooms?rerun=r1`)),
      py.find("GET /runs/{rid}/zooms?rerun=r1").json(),
      1e-6,
    );
    const bg = await getArray(`/runs/${rid}/image/background?rerun=r1`);
    const want = py.find("GET /runs/{rid}/image/background?rerun=r1").npy();
    expect(bg.shape).toEqual(want.shape);
    const ov = await settled(() => getArray(`/runs/${rid}/traces/overview?kind=YrA&bins=37`));
    expect(ov.shape).toEqual(py.find("GET /runs/{rid}/traces/overview?kind=YrA&bins=37").npy().shape);
  });
});
