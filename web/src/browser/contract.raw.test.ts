import { cpSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { getJSON } from "../api";
import { Folders } from "./folders";
import { memoryKV } from "./kv";
import { configure } from "./state";
import { expectClose, loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { NodeDir } from "./testing/nodeDir";

const py = loadContract(join(DATA, "contract-python"));
let raw = "";

beforeAll(async () => {
  raw = (await setupE2eRun()).raw;
});

describe("/raw/info matches main's", () => {
  it("answers the e2e raw folder as main does, except its path", async () => {
    const got = await getJSON(`/raw/info?path=${encodeURIComponent(raw)}`);
    expectClose(got, py.find("GET /raw/info?path=e2e/.data/raw").json(), 1e-9, ["path"]);
    expect((got as { path: string }).path).toBe(raw);
  });

  it("lists the videos in number order, skips what has no number, and reads the frame rate from the files", async () => {
    const top = mkdtempSync(join(tmpdir(), "rawinfo-"));
    const dir = join(top, "raw");
    cpSync(join(DATA, "raw"), dir, { recursive: true });
    cpSync(join(dir, "2.avi"), join(dir, "10.avi"));
    cpSync(join(dir, "0.avi"), join(dir, "extra.avi"));
    writeFileSync(join(dir, "metaData.json"), JSON.stringify({ frameRate: "20FPS" }));
    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(new NodeDir(top), "picked");
    configure({ folders });
    const got = await getJSON<{ names: string[]; skipped: string[]; segments: number; frames: number; fps: number }>(
      `/raw/info?path=${encodeURIComponent(`${name}/raw`)}`,
    );
    expect(got).toMatchObject({ names: ["0.avi", "1.avi", "2.avi", "10.avi"], skipped: ["extra.avi"], segments: 4 });
    expect(got.frames).toBe(400);
    expect(got.fps).toBe(20);
  });

  it("answers 404 for a folder without videos and for one that is not known", async () => {
    const { name } = await setupE2eRun(); // the e2e folders again (the test above replaced them)
    await expect(getJSON(`/raw/info?path=${encodeURIComponent(name)}`)).rejects.toMatchObject({
      status: 404,
      message: expect.stringMatching(/^no videos in .*: each video's file name must start with a number/),
    });
    await expect(getJSON("/raw/info?path=nowhere/raw")).rejects.toMatchObject({ status: 404 });
  });
});
