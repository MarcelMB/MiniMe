import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { fpsFromFolder } from "./fps";
import { NodeDir } from "./testing/nodeDir";

/** A session folder `s/` holding `raw/`; `files` maps "meta.json" to s/ and the rest into s/raw/. */
function session(files: Record<string, string>) {
  const s = mkdtempSync(join(tmpdir(), "fps-"));
  mkdirSync(join(s, "raw"));
  for (const [name, text] of Object.entries(files))
    writeFileSync(name === "meta.json" ? join(s, name) : join(s, "raw", name), text);
  return { raw: new NodeDir(join(s, "raw")), parent: new NodeDir(s) };
}

// The expected values are what Python's io.fps_from_files answers on the same folders.
describe("fpsFromFolder", () => {
  it("reads fps from ../meta.json first", async () => {
    const { raw, parent } = session({ "meta.json": '{"fps": 30}', "metaData.json": '{"frameRate": "20FPS"}' });
    expect(await fpsFromFolder(raw, parent)).toBe(30);
  });

  it("reads the leading number of frameRate in metaData.json, then neural_metaData.json", async () => {
    const a = session({ "metaData.json": '{"frameRate": "29.97FPS"}' });
    expect(await fpsFromFolder(a.raw, a.parent)).toBe(29.97);
    const b = session({ "metaData.json": '{"frameRate": null}', "neural_metaData.json": '{"frameRate": 20}' });
    expect(await fpsFromFolder(b.raw, b.parent)).toBe(20);
    const c = session({ "metaData.json": '{"frameRate": "fast"}' });
    expect(await fpsFromFolder(c.raw, c.parent)).toBeNull();
  });

  it("answers null for a meta.json without fps or a metadata file that does not parse, like main", async () => {
    const a = session({ "meta.json": '{"frames": 3}', "metaData.json": '{"frameRate": 30}' });
    expect(await fpsFromFolder(a.raw, a.parent)).toBeNull();
    const b = session({ "metaData.json": "{oops", "neural_metaData.json": '{"frameRate": 30}' });
    expect(await fpsFromFolder(b.raw, b.parent)).toBeNull();
  });

  it("still reads ../meta.json when the raw folder itself is gone", async () => {
    const { parent } = session({ "meta.json": '{"fps": 25.5}' });
    expect(await fpsFromFolder(null, parent)).toBe(25.5);
    expect(await fpsFromFolder(null, null)).toBeNull();
  });
});
