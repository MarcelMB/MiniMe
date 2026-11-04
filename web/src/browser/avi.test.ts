import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { type Indexer, RawVideo, type SegIndex } from "./avi";
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
