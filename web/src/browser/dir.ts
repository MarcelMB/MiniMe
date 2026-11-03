import { HttpError } from "./router";

export type Entry = { name: string; kind: "file" | "directory" };

/** A folder: a picked or OPFS directory handle, a read-only file list, or (tests) a disk path. */
export interface Dir {
  readonly name: string;
  readonly writable: boolean;
  entries(): Promise<Entry[]>;
  dir(name: string, opts?: { create?: boolean }): Promise<Dir>;
  file(name: string): Promise<File>;
  write(name: string, data: Blob | Uint8Array | string): Promise<void>;
  remove(name: string, opts?: { recursive?: boolean }): Promise<void>;
  permission(mode: "read" | "readwrite", ask: boolean): Promise<boolean>;
}

const notFound = (name: string) => new HttpError(404, `${name} not found`);
const readOnly = () => new HttpError(403, "this folder was opened read-only");

export class HandleDir implements Dir {
  readonly writable = true;
  constructor(readonly handle: FileSystemDirectoryHandle) {}
  get name() {
    return this.handle.name;
  }
  async entries(): Promise<Entry[]> {
    const out: Entry[] = [];
    // biome-ignore lint/suspicious/noExplicitAny: entries() is missing from TS 5.6's DOM lib
    for await (const [name, h] of (this.handle as any).entries() as AsyncIterable<[string, FileSystemHandle]>)
      out.push({ name, kind: h.kind });
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }
  async dir(name: string, opts: { create?: boolean } = {}): Promise<Dir> {
    try {
      return new HandleDir(await this.handle.getDirectoryHandle(name, { create: !!opts.create }));
    } catch {
      throw notFound(name);
    }
  }
  async file(name: string): Promise<File> {
    try {
      return await (await this.handle.getFileHandle(name)).getFile();
    } catch {
      throw notFound(name);
    }
  }
  async write(name: string, data: Blob | Uint8Array | string): Promise<void> {
    const w = await (await this.handle.getFileHandle(name, { create: true })).createWritable();
    await w.write(data);
    await w.close();
  }
  async remove(name: string, opts: { recursive?: boolean } = {}): Promise<void> {
    await this.handle.removeEntry(name, opts).catch(() => undefined);
  }
  async permission(mode: "read" | "readwrite", ask: boolean): Promise<boolean> {
    // biome-ignore lint/suspicious/noExplicitAny: queryPermission/requestPermission are Chromium-only
    const h = this.handle as any;
    if (!h.queryPermission) return true; // OPFS and browsers without the API
    if ((await h.queryPermission({ mode })) === "granted") return true;
    return ask && (await h.requestPermission({ mode })) === "granted";
  }
}

/** A read-only folder from `<input webkitdirectory>` (any browser). */
export class FileListDir implements Dir {
  readonly writable = false;
  private constructor(
    readonly name: string,
    private files: Map<string, File>,
    private dirs: Map<string, FileListDir>,
  ) {}
  static fromFiles(files: File[]): FileListDir {
    const root = new FileListDir(files[0]?.webkitRelativePath.split("/")[0] ?? "", new Map(), new Map());
    for (const f of files) {
      const parts = f.webkitRelativePath.split("/").slice(1);
      let d = root;
      for (const p of parts.slice(0, -1)) {
        if (!d.dirs.has(p)) d.dirs.set(p, new FileListDir(p, new Map(), new Map()));
        d = d.dirs.get(p)!;
      }
      d.files.set(parts[parts.length - 1], f);
    }
    return root;
  }
  async entries(): Promise<Entry[]> {
    return [
      ...[...this.dirs.keys()].map((name) => ({ name, kind: "directory" as const })),
      ...[...this.files.keys()].map((name) => ({ name, kind: "file" as const })),
    ].sort((a, b) => a.name.localeCompare(b.name));
  }
  async dir(name: string): Promise<Dir> {
    const d = this.dirs.get(name);
    if (!d) throw notFound(name);
    return d;
  }
  async file(name: string): Promise<File> {
    const f = this.files.get(name);
    if (!f) throw notFound(name);
    return f;
  }
  async write(_name: string, _data: Blob | Uint8Array | string): Promise<void> {
    throw readOnly();
  }
  async remove(_name: string, _opts?: { recursive?: boolean }): Promise<void> {
    throw readOnly();
  }
  async permission(_mode: "read" | "readwrite", _ask: boolean): Promise<boolean> {
    return true;
  }
}

export async function walk(root: Dir, rel: string, opts: { create?: boolean } = {}): Promise<Dir> {
  let d = root;
  for (const p of rel.split("/").filter(Boolean)) d = await d.dir(p, opts);
  return d;
}

const splitLast = (rel: string): [string, string] => {
  const i = rel.lastIndexOf("/");
  return i < 0 ? ["", rel] : [rel.slice(0, i), rel.slice(i + 1)];
};

export async function fileAt(d: Dir, rel: string): Promise<File> {
  const [dir, name] = splitLast(rel);
  return (await walk(d, dir)).file(name);
}
export async function readBytes(d: Dir, rel: string): Promise<Uint8Array> {
  return new Uint8Array(await (await fileAt(d, rel)).arrayBuffer());
}
export async function readJSON<T>(d: Dir, rel: string): Promise<T> {
  return JSON.parse(await (await fileAt(d, rel)).text()) as T;
}
export async function writeJSON(d: Dir, rel: string, v: unknown): Promise<void> {
  const [dir, name] = splitLast(rel);
  await (await walk(d, dir, { create: true })).write(name, JSON.stringify(v, null, 1));
}
export async function exists(d: Dir, rel: string): Promise<boolean> {
  return fileAt(d, rel).then(
    () => true,
    () => false,
  );
}
