// web/src/browser/testing/nodeDir.ts (tests only: Node's fs behind the Dir interface)
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import type { Dir, Entry } from "../dir";
import { HttpError } from "../router";

export class NodeDir implements Dir {
  readonly writable = true;
  constructor(readonly path: string) {}
  get name() {
    return basename(this.path);
  }
  async entries(): Promise<Entry[]> {
    return readdirSync(this.path, { withFileTypes: true })
      .map((e) => ({ name: e.name, kind: e.isDirectory() ? ("directory" as const) : ("file" as const) }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }
  async dir(name: string, opts: { create?: boolean } = {}): Promise<Dir> {
    const p = join(this.path, name);
    if (opts.create) mkdirSync(p, { recursive: true });
    if (!existsSync(p) || !statSync(p).isDirectory()) throw new HttpError(404, `${name} not found`);
    return new NodeDir(p);
  }
  async file(name: string): Promise<File> {
    const p = join(this.path, name);
    if (!existsSync(p) || !statSync(p).isFile()) throw new HttpError(404, `${name} not found`);
    return new File([readFileSync(p)], name, { lastModified: statSync(p).mtimeMs });
  }
  async write(name: string, data: Blob | Uint8Array | string): Promise<void> {
    const bytes = data instanceof Blob ? new Uint8Array(await data.arrayBuffer()) : data;
    writeFileSync(join(this.path, name), bytes);
  }
  async remove(name: string, opts: { recursive?: boolean } = {}): Promise<void> {
    rmSync(join(this.path, name), { recursive: !!opts.recursive, force: true });
  }
  async permission(): Promise<boolean> {
    return true;
  }
}
