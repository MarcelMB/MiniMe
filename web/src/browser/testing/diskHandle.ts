// web/src/browser/testing/diskHandle.ts (tests only: a File System Access directory handle on a
// disk folder, enough for `HandleDir`; `path` places it in a made-up tree for `resolve`/`isSameEntry`)
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type Fake = FileSystemDirectoryHandle & { path: string[] };

/** A handle named after the last of `path`, on the folder `disk`. */
export function diskHandle(path: string[], disk: string): FileSystemDirectoryHandle {
  const handle = {
    kind: "directory",
    name: path[path.length - 1],
    path,
    async *entries() {
      for (const e of readdirSync(disk, { withFileTypes: true }))
        yield [
          e.name,
          e.isDirectory() ? diskHandle([...path, e.name], join(disk, e.name)) : fileHandle(join(disk, e.name)),
        ];
    },
    async getDirectoryHandle(name: string, o: { create?: boolean } = {}) {
      const p = join(disk, name);
      if (o.create) mkdirSync(p, { recursive: true });
      if (!existsSync(p) || !statSync(p).isDirectory()) throw new DOMException(`${name} not found`, "NotFoundError");
      return diskHandle([...path, name], p);
    },
    async getFileHandle(name: string, o: { create?: boolean } = {}) {
      const p = join(disk, name);
      if (o.create && !existsSync(p)) writeFileSync(p, "");
      if (!existsSync(p) || !statSync(p).isFile()) throw new DOMException(`${name} not found`, "NotFoundError");
      return fileHandle(p);
    },
    async removeEntry(name: string, o: { recursive?: boolean } = {}) {
      rmSync(join(disk, name), { recursive: !!o.recursive, force: true });
    },
    async isSameEntry(other: Fake) {
      return other.path.join("/") === path.join("/");
    },
    /** `[]` for the same folder, the names down to it for a folder inside, else null. */
    async resolve(other: Fake) {
      return path.every((p, i) => other.path[i] === p) ? other.path.slice(path.length) : null;
    },
  };
  return handle as unknown as FileSystemDirectoryHandle;
}

function fileHandle(p: string) {
  return {
    kind: "file",
    getFile: async () => new File([readFileSync(p)], p.split("/").pop() ?? "", { lastModified: statSync(p).mtimeMs }),
    createWritable: async () => ({
      write: async (data: Blob | Uint8Array | string) =>
        writeFileSync(
          p,
          typeof data === "string" || data instanceof Uint8Array ? data : new Uint8Array(await data.arrayBuffer()),
        ),
      close: async () => undefined,
    }),
  };
}
