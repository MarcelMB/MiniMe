import { mkdirSync, mkdtempSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import { runZip } from "./download";
import { NodeDir } from "./testing/nodeDir";
import { readZip } from "./zip";

/** A run folder as the browser pipeline writes it: nested folders, an empty file, binary traces. */
function runFolder(): { path: string; files: Map<string, Uint8Array> } {
  const path = mkdtempSync(join(tmpdir(), "dl-"));
  const files = new Map<string, Uint8Array>();
  const put = (rel: string, bytes: Uint8Array) => {
    const p = join(path, rel);
    mkdirSync(join(p, ".."), { recursive: true });
    writeFileSync(p, bytes);
    files.set(rel, bytes);
  };
  const noise = (n: number, seed: number) =>
    Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed * 7 + (i >> 8)) & 255);
  put("params.json", new TextEncoder().encode('{"source": "browser"}'));
  put("report.json", new TextEncoder().encode('{"stages": []}'));
  put("cnmf/C.npy", noise(300_000, 1));
  put("cnmf/units.csv", new TextEncoder().encode("unit_id,centroid_h\n0,1.5\n"));
  put("factors/w00/W.npy", noise(70_000, 2));
  put("factors/w01/W.npy", noise(1, 3));
  put("seeds/empty.npy", new Uint8Array(0));
  return { path, files };
}

const onDisk = (root: string, dir = root): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? onDisk(root, join(dir, e.name)) : [relative(root, join(dir, e.name))],
  );

describe("runZip", () => {
  it("holds every file of the run folder under its relative path, byte for byte", async () => {
    const { path, files } = runFolder();
    const zip = await readZip(await runZip({ dir: new NodeDir(path) }));
    expect([...zip.keys()].sort()).toEqual(onDisk(path).sort());
    expect([...zip.keys()].sort()).toEqual([...files.keys()].sort());
    for (const [name, bytes] of files) expect(new Uint8Array(await zip.get(name)!.arrayBuffer()), name).toEqual(bytes);
  });

  it("is the files plus headers: the data is in the archive once", async () => {
    const { path } = runFolder();
    const total = onDisk(path).reduce((s, f) => s + statSync(join(path, f)).size, 0);
    const blob = await runZip({ dir: new NodeDir(path) });
    const names = onDisk(path);
    const nameBytes = names.reduce((s, n) => s + n.length, 0);
    // a local and a central header per file (30 + 46 bytes and the name twice), then the end record
    const headers = names.length * (30 + 46) + 2 * nameBytes + 22;
    expect(blob.size).toBe(total + headers);
  });
});
