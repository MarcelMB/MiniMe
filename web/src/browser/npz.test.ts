import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { readNpy } from "./npy";
import { csrNpz, readCsr, readNpz } from "./npz";
import { readZip, zipBlob } from "./zip";

const fx = (n: string) => new Blob([readFileSync(join(__dirname, "__fixtures__", n))]);
const PY =
  process.env.MINIME_PYTHON ?? "/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/MiniMe/.venv/bin/python";

describe("npz", () => {
  for (const name of ["csr_compressed.npz", "csr_stored.npz"])
    it(`reads scipy's ${name}`, async () => {
      const a = await readCsr(fx(name));
      expect(a.shape).toEqual([3, 4]);
      expect(Array.from(a.indptr)).toEqual([0, 2, 2, 4]);
      expect(Array.from(a.indices)).toEqual([1, 3, 0, 2]);
      expect(Array.from(a.data)).toEqual([1.5, 2, 3, 0.25]);
    });

  it("reads numpy's savez", async () => {
    const m = await readNpz(fx("arrays.npz"));
    expect(readNpy(m.get("shifts")!).shape).toEqual([3, 2]);
    expect(Array.from(readNpy(m.get("q")!).data)).toEqual([0.5, 1]);
  });

  it("writes a CSR npz that scipy loads", async () => {
    const a = {
      shape: [2, 3] as [number, number],
      data: new Float32Array([1, 2]),
      indices: new Int32Array([0, 2]),
      indptr: new Int32Array([0, 1, 2]),
    };
    const dir = mkdtempSync(join(tmpdir(), "npz-"));
    const p = join(dir, "A.npz");
    writeFileSync(p, new Uint8Array(await (await csrNpz(a)).arrayBuffer()));
    const out = execFileSync(PY, [
      "-c",
      `import scipy.sparse as s; print(s.load_npz(${JSON.stringify(p)}).toarray().tolist())`,
    ]);
    expect(out.toString().trim()).toBe("[[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]");
    expect(await readCsr(new Blob([readFileSync(p)]))).toEqual(a);
  });

  it("writes zip64 records that readZip and Python's zipfile read", async () => {
    const z = await zipBlob(
      [
        { name: "a.txt", data: new Blob(["hello"]) },
        { name: "d/b.bin", data: new Blob([new Uint8Array([1, 2, 3])]) },
      ],
      { forceZip64: true },
    );
    const m = await readZip(z);
    expect(await m.get("a.txt")!.text()).toBe("hello");
    expect(new Uint8Array(await m.get("d/b.bin")!.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3]));
    const p = join(mkdtempSync(join(tmpdir(), "zip64-")), "z.zip");
    writeFileSync(p, new Uint8Array(await z.arrayBuffer()));
    // sizes and offsets live in the zip64 extra fields: 60 = 30 + 5 + 20 (header) + 5 (data) for the first entry
    const code = `import zipfile; z = zipfile.ZipFile(${JSON.stringify(p)}); print(z.testzip(), [(i.filename, i.file_size, i.header_offset) for i in z.infolist()])`;
    const out = execFileSync(PY, ["-c", code]);
    expect(out.toString().trim()).toBe("None [('a.txt', 5, 0), ('d/b.bin', 3, 60)]");
  });

  it("round-trips a stored zip", async () => {
    const z = await zipBlob([
      { name: "a/b.txt", data: new Blob(["hello"]) },
      { name: "c.bin", data: new Blob([new Uint8Array([1, 2, 3])]) },
    ]);
    const m = await readZip(z);
    expect(await m.get("a/b.txt")!.text()).toBe("hello");
    expect(new Uint8Array(await m.get("c.bin")!.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3]));
  });
});
