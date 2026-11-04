import { beforeAll, describe, expect, it } from "vitest";
import { getArray, getJSON } from "../api";
import { expectClose, loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { settled } from "./testing/settled";

const py = loadContract(`${DATA}/contract-python`);
let rid = "";
beforeAll(async () => {
  rid = (await setupE2eRun()).rid;
});

const MINE = ["/image/", "/motion", "/seeds", "/units", "/frame/"];
/** Answered differently on a Python run (its factors, and so the noise image and the
 * reconstruction, are not in the browser's format): the browser says 404 for these, which is all
 * that is checked. The preprocessed frames and the residual are not in this list: a Python run's
 * flat tile store and block maxima are read like main reads them, so `frame/100?source=pre` is
 * compared byte for byte and the residual within 1e-4. */
const BROWSER_ONLY = ["GET /runs/{rid}/image/noise", "GET /runs/{rid}/frame/299?source=recon"];

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < b.length; i++)
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i])));
};

describe("image, motion, seed, unit and frame requests match main's", () => {
  const entries = py.entries.filter(
    (e) => e.request.startsWith("GET /runs/{rid}/") && MINE.some((m) => e.request.includes(m)),
  );
  it("covers every kind of request", () => {
    for (const m of MINE)
      expect(
        entries.some((e) => e.request.includes(m)),
        m,
      ).toBe(true);
  });

  for (const req of entries)
    it(req.request, async () => {
      const path = req.request.slice(4).replace("{rid}", rid);
      const want = py.find(req.request);
      if (BROWSER_ONLY.includes(req.request)) {
        await expect(settled(() => getArray(path))).rejects.toMatchObject({
          status: 404,
          message: expect.stringContaining("not available for this run"),
        });
      } else if (req.status !== 200) {
        const detail = (want.json() as { detail: string }).detail;
        await expect(settled(() => getArray(path))).rejects.toMatchObject({ status: req.status, message: detail });
      } else if (req.file.endsWith(".json")) {
        expectClose(await settled(() => getJSON(path)), want.json(), 1e-5);
      } else {
        const got = await settled(() => getArray(path));
        const w = want.npy();
        expect(got.shape).toEqual(w.shape);
        expect(got.data.constructor).toBe(w.data.constructor);
        close(got.data, w.data, path.includes("/motion/residual") ? 1e-4 : path.includes("/motion") ? 1e-5 : 0);
      }
    });
});
