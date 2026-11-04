import { beforeAll, describe, expect, it } from "vitest";
import { getArray, getJSON } from "../api";
import { loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { settled } from "./testing/settled";

const py = loadContract(`${DATA}/contract-python`);
let rid = "";
beforeAll(async () => {
  rid = (await setupE2eRun()).rid;
});

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < b.length; i++)
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i])));
};

describe("trace requests match main's", () => {
  for (const req of py.entries.filter((e) => e.request.includes("/traces/") || e.request.endsWith("/zooms")))
    it(req.request, async () => {
      const path = req.request.slice(4).replace("{rid}", rid);
      const want = py.find(req.request);
      if (req.file.endsWith(".json")) expect(await settled(() => getJSON(path))).toEqual(want.json());
      else {
        const got = await settled(() => getArray(path));
        const w = want.npy();
        expect(got.shape).toEqual(w.shape);
        close(got.data, w.data, req.request.includes("lowpass") ? 1e-4 : 0);
      }
    });
});
