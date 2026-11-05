import { describe, expect, it } from "vitest";
import type { ToTuneWorker } from "../../stream/tuneProtocol";
import { FakeTuneWorker, out } from "../testing/fakeTuneWorker";
import { TuneWorkers } from "./windows";

const windows = (n: number) =>
  Array.from({ length: n }, (_, i) => ({ i, t0: i * 10, t1: i * 10 + 10, files: new Map() }));

describe("TuneWorkers", () => {
  it("caps the worker count by the budget and the window count", () => {
    expect(TuneWorkers.workerCount(12, 100e6, 8)).toBe(4); // 1.2 GB / (3 * 100 MB) = 4
    expect(TuneWorkers.workerCount(2, 1e6, 8)).toBe(2);
    expect(TuneWorkers.workerCount(12, 10e9, 8)).toBe(1);
  });

  it("sums the windows' trials into one session trial and keeps the median size", async () => {
    const made: FakeTuneWorker[] = [];
    const tw = new TuneWorkers(
      () => {
        const w = new FakeTuneWorker((i) => out(10 + i, 1));
        made.push(w);
        return w as unknown as Worker;
      },
      new Uint8Array(),
      2,
    );
    await tw.load(windows(3), 1e6);
    expect(made.map((w) => w.held)).toEqual([[0, 2], [1]]);
    const t = await tw.trial(new Uint8Array(), { diameter: 7 } as never, true);
    expect(t.score).toMatchObject({ good: 33, flagged: 3, lost: 0, score: 30 });
    expect(t.counts).toEqual({ seeds_pnr: 30, seeds_merged: 15, units_final: 36 });
    expect(t.size10Median).toBe(25);
    expect(t.error).toBeNull();
    expect(tw.heapPeakMb).toBe(20);
    const u = await tw.units();
    expect(u.map((w) => w.i)).toEqual([0, 1, 2]);
    expect(await tw.contrastHist(100, 50, 8, 16)).toEqual([3]);
    tw.close();
    expect(made.every((w) => w.terminated)).toBe(true);
  });

  it("fails the setting when one window fails, with that window's text", async () => {
    const tw = new TuneWorkers(
      () =>
        new FakeTuneWorker((i) =>
          i === 1 ? out(0, 0, 0, "ValueError: no seeds survived pnr_refine") : out(5, 0),
        ) as unknown as Worker,
      new Uint8Array(),
      2,
    );
    await tw.load(windows(2), 1e6);
    const t = await tw.trial(new Uint8Array(), { diameter: 7 } as never, false);
    expect(t.score).toBeNull();
    expect(t.error).toBe("ValueError: no seeds survived pnr_refine");
  });

  it("sends the reference trial's counts as the caps of the later ones", async () => {
    const seen: string[] = [];
    const tw = new TuneWorkers(
      () => {
        const w = new FakeTuneWorker(() => out(5, 0));
        const post = w.postMessage.bind(w);
        w.postMessage = (m: ToTuneWorker) => {
          if (m.type === "trial") seen.push(m.caps);
          post(m);
        };
        return w as unknown as Worker;
      },
      new Uint8Array(),
      1,
    );
    await tw.load(windows(1), 1e6);
    await tw.trial(new Uint8Array(), { diameter: 7 } as never, true);
    await tw.trial(new Uint8Array(), { diameter: 7 } as never, false);
    expect(seen).toEqual(["", JSON.stringify({ "0": { seeds_pnr: 10, seeds_merged: 5 } })]);
  });
});
