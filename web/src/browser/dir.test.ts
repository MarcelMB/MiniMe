import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { exists, FileListDir, readJSON, walk, writeJSON } from "./dir";
import { NodeDir } from "./testing/nodeDir";

describe("Dir", () => {
  it("writes, walks and reads on disk", async () => {
    const root = new NodeDir(mkdtempSync(join(tmpdir(), "dir-")));
    await writeJSON(await walk(root, "a/b", { create: true }), "x.json", { v: 1 });
    expect(await readJSON(root, "a/b/x.json")).toEqual({ v: 1 });
    expect(await exists(root, "a/b/y.json")).toBe(false);
    await expect(walk(root, "nope")).rejects.toMatchObject({ status: 404 });
    expect((await root.entries()).map((e) => e.name)).toEqual(["a"]);
  });

  it("reads a webkitdirectory file list, read-only", async () => {
    const f = (path: string, text: string) => {
      const file = new File([text], path.split("/").pop()!);
      Object.defineProperty(file, "webkitRelativePath", { value: path });
      return file;
    };
    const d = FileListDir.fromFiles([f("run/params.json", "{}"), f("run/cnmf/units.csv", "u")]);
    expect(d.name).toBe("run");
    expect(d.writable).toBe(false);
    expect(await (await (await d.dir("cnmf")).file("units.csv")).text()).toBe("u");
    await expect(d.write("x", "y")).rejects.toMatchObject({ status: 403 });
  });
});
