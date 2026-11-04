import { appendFileSync, mkdtempSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { type Indexer, listSegments, openRawVideo, RawVideo, type SegIndex } from "./avi";
import { NodeDir } from "./testing/nodeDir";

it("indexes the e2e AVIs in name order and decodes frames", async () => {
  const v = await RawVideo.open(new NodeDir(resolve(__dirname, "../../e2e/.data/raw")));
  expect([v.frames, v.height, v.width]).toEqual([300, 64, 64]);
  expect(v.names).toEqual(["0.avi", "1.avi", "2.avi"]);
  const f = await v.frame(150);
  expect(f.length).toBe(64 * 64);
  await expect(v.frame(300)).rejects.toMatchObject({ status: 400 });
});

describe("RawVideo on made-up files", () => {
  const folder = (files: Record<string, number[]>): NodeDir => {
    const d = mkdtempSync(join(tmpdir(), "avi-"));
    for (const [name, bytes] of Object.entries(files)) writeFileSync(join(d, name), Uint8Array.from(bytes));
    return new NodeDir(d);
  };
  /** Every file holds `n` frames of `size` bytes, back to back, of an `h` x `w` video. */
  const index =
    (n: number, size: number, h: number, w: number): Indexer =>
    async (files) =>
      files.map(
        (): SegIndex => ({
          n,
          height: h,
          width: w,
          offsets: Array.from({ length: n }, (_, i) => i * size),
          sizes: Array(n).fill(size),
        }),
      );

  it("orders the files by their number, lists the ones it ignores and leaves hidden files out", async () => {
    const bytes = [0, 0, 0, 0];
    const dir = folder({
      "10.avi": bytes,
      "2.avi": bytes,
      "0_session.avi": bytes,
      "behavior.avi": bytes,
      "._0.avi": bytes,
      "notes.txt": bytes,
    });
    const v = await RawVideo.open(dir, index(2, 2, 1, 2));
    expect(v.names).toEqual(["0_session.avi", "2.avi", "10.avi"]);
    expect(v.skipped).toEqual(["behavior.avi"]);
    expect([v.frames, v.bytes]).toEqual([6, 12]);
  });

  it("finds each frame in its file, across file borders", async () => {
    const dir = folder({ "0.avi": [10, 11, 12, 13], "1.avi": [20, 21, 22, 23] });
    const v = await RawVideo.open(dir, index(2, 2, 1, 2));
    expect(Array.from(await v.frame(1))).toEqual([12, 13]);
    expect(Array.from(await v.frame(2))).toEqual([20, 21]);
    expect((await v.clip(1, 3)).map((f) => Array.from(f))).toEqual([
      [12, 13],
      [20, 21],
      [22, 23],
    ]);
  });

  it("drops the padding of rows kept to 4 bytes", async () => {
    const dir = folder({ "0.avi": [1, 2, 3, 0, 4, 5, 6, 0] });
    const v = await RawVideo.open(dir, index(1, 8, 2, 3));
    expect(Array.from(await v.frame(0))).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("refuses frame numbers outside the videos", async () => {
    const v = await RawVideo.open(folder({ "0.avi": [0, 0] }), index(2, 1, 1, 1));
    for (const t of [-1, 2, 0.5, Number.NaN])
      await expect(v.frame(t)).rejects.toMatchObject({ status: 400, message: `frame ${t} out of range 0..1` });
  });

  it("explains the naming rule when no file qualifies", async () => {
    const folders: Record<string, number[]>[] = [{}, { "behavior.avi": [0], "readme.txt": [0] }];
    for (const files of folders)
      await expect(RawVideo.open(folder(files), index(1, 1, 1, 1))).rejects.toMatchObject({
        status: 404,
        message: expect.stringMatching(/^no videos in .*: each video's file name must start with a number/),
      });
  });

  it("refuses two files with the same number, as main does", async () => {
    const dir = folder({ "1.avi": [0], "01_b.avi": [0] });
    await expect(RawVideo.open(dir, index(1, 1, 1, 1))).rejects.toMatchObject({
      status: 400,
      message: expect.stringMatching(
        /^(1\.avi and 01_b|01_b\.avi and 1)\.avi in .* both start with 1; rename one: each video/,
      ),
    });
  });

  it("refuses files with frames of different sizes", async () => {
    const dir = folder({ "0.avi": [0], "1.avi": [0] });
    const mixed: Indexer = async () => [
      { n: 1, height: 2, width: 3, offsets: [0], sizes: [1] },
      { n: 1, height: 2, width: 4, offsets: [0], sizes: [1] },
    ];
    await expect(RawVideo.open(dir, mixed)).rejects.toMatchObject({
      status: 400,
      message: "1.avi: 2x4 frames, 0.avi has 2x3",
    });
  });
});

describe("openRawVideo (main caches io.probe per path)", () => {
  const folder = () => {
    const d = mkdtempSync(join(tmpdir(), "avi-"));
    for (const name of ["0.avi", "1.avi"]) writeFileSync(join(d, name), Uint8Array.of(1, 2, 3, 4));
    return { path: d, dir: new NodeDir(d) };
  };
  /** Two 1 x 2 frames per file; counts the folders it indexed. */
  const counting = () => {
    const calls: string[][] = [];
    const index: Indexer = async (files) => {
      calls.push(files.map((f) => f.name));
      await new Promise((r) => setTimeout(r, 5)); // the requests below come while it runs
      return files.map(() => ({ n: 2, height: 1, width: 2, offsets: [0, 2], sizes: [2, 2] }));
    };
    return { calls, index };
  };

  it("indexes a folder once for requests that come together", async () => {
    const { dir } = folder();
    const { calls, index } = counting();
    const all = await Promise.all(Array.from({ length: 8 }, () => openRawVideo("s/raw", dir, { index })));
    expect(calls).toHaveLength(1);
    expect(new Set(all).size).toBe(1);
    expect(all[0].frames).toBe(4);
    await openRawVideo("s/raw", dir, { index }); // later, unchanged
    expect(calls).toHaveLength(1);
  });

  it("indexes again when a video changed size or time, or a video came or went", async () => {
    const { path, dir } = folder();
    const { calls, index } = counting();
    await openRawVideo("s/raw", dir, { index });
    appendFileSync(join(path, "1.avi"), Uint8Array.of(5));
    await openRawVideo("s/raw", dir, { index });
    expect(calls).toHaveLength(2);
    utimesSync(join(path, "0.avi"), new Date(2020, 0, 1), new Date(2020, 0, 1));
    await openRawVideo("s/raw", dir, { index });
    expect(calls).toHaveLength(3);
    writeFileSync(join(path, "2.avi"), Uint8Array.of(1, 2, 3, 4));
    expect((await openRawVideo("s/raw", dir, { index })).names).toEqual(["0.avi", "1.avi", "2.avi"]);
    writeFileSync(join(path, "behavior.avi"), Uint8Array.of(0));
    expect((await openRawVideo("s/raw", dir, { index })).skipped).toEqual(["behavior.avi"]);
    expect(calls).toHaveLength(5);
  });

  it("keeps folders at different paths apart, takes a listing it is given, and does not remember a failure", async () => {
    const { dir } = folder();
    const { calls, index } = counting();
    await openRawVideo("a/raw", dir, { index });
    await openRawVideo("b/raw", dir, { index, listed: await listSegments(dir) });
    expect(calls).toHaveLength(2);
    let fail = true;
    const flaky: Indexer = async (files) => {
      if (fail) throw new Error("busy");
      return index(files);
    };
    await expect(openRawVideo("c/raw", dir, { index: flaky })).rejects.toThrow("busy");
    fail = false;
    await expect(openRawVideo("c/raw", dir, { index: flaky })).resolves.toMatchObject({ frames: 4 });
  });
});
