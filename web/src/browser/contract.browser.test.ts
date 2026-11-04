import { existsSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { getArray, getJSON } from "../api";
import { expectClose, loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { settled } from "./testing/settled";

// The run folder the browser pipeline wrote (`e2e/stream.spec.ts` saves it to
// e2e/.data/browser-run/run) and main's answers on it (`scripts/contract_dump.py`, with the
// reference Python routes): the browser router must answer every request main answers with 200
// the same. Both are produced by hand (see docs/browser-app.md), so the test skips without them.
const RUN = join(DATA, "browser-run/run");
const DUMP = join(DATA, "contract-browser");
const have = existsSync(join(RUN, "report.json")) && existsSync(join(DUMP, "manifest.json"));

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < b.length; i++)
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i])));
};

/** Keys that differ where the two sides see different folders: the test copies the run to
 * `processed/t` under a browser root that holds the raw folder and its `meta.json` (frame rate),
 * while main opens the exported folder, whose `raw_dir` is missing and which has no fps (so no
 * zoom windows). */
const SKIP: [string, string[]][] = [
  // `peak_wasm`: the browser's own key for the wasm heap of its run (main has no such field)
  ["/summary", ["id", "name", "path", "raw_dir", "mtime", "fps", "available", "params", "peak_wasm"]],
  ["/zooms", ["windows"]],
];

describe.skipIf(!have)("a run the browser wrote answers like main's routes on it", () => {
  const py = have ? loadContract(DUMP) : null;
  const entries = (py?.entries ?? []).filter((e) => e.status === 200 && e.request.startsWith("GET /runs/{rid}/"));
  let rid = "";
  beforeAll(async () => {
    rid = (await setupE2eRun(RUN)).rid;
  });

  it("is asked for every kind of request main answers", () => {
    for (const k of ["/summary", "/image/max_proj", "/image/background", "/image/rank", "/motion", "/seeds", "/units"])
      expect(
        entries.some((e) => e.request.includes(k)),
        k,
      ).toBe(true);
    expect(entries.some((e) => e.request.includes("/traces/overview"))).toBe(true);
    expect(entries.some((e) => /\/traces\/\d+/.test(e.request))).toBe(true);
  });

  it("zooms: the windows follow the frame rate of the raw folder", async () => {
    expect(await settled(() => getJSON(`/runs/${rid}/zooms`))).toMatchObject({ windows: [{ start: 0, stop: 300 }] });
  });

  for (const req of entries)
    it(req.request, async () => {
      const path = req.request.slice(4).replace("{rid}", rid);
      const want = py!.find(req.request);
      if (req.file.endsWith(".json")) {
        const skip = SKIP.find(([k]) => req.request.endsWith(k))?.[1] ?? [];
        expectClose(
          await settled(() => getJSON(path)),
          want.json(),
          req.request.includes("lowpass") ? 1e-4 : 1e-5,
          skip,
        );
      } else {
        const got = await settled(() => getArray(path));
        const w = want.npy();
        expect(got.shape).toEqual(w.shape);
        expect(got.data.constructor).toBe(w.data.constructor);
        close(got.data, w.data, req.request.includes("lowpass") ? 1e-4 : 1e-5);
      }
    });
});
