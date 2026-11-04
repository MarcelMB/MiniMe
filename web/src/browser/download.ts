import type { Dir } from "./dir";
import type { RunInfo } from "./runs";
import { zipBlob } from "./zip";

async function collect(d: Dir, prefix: string, out: { name: string; data: Blob }[]): Promise<void> {
  for (const e of await d.entries())
    if (e.kind === "file") out.push({ name: `${prefix}${e.name}`, data: await d.file(e.name) });
    else await collect(await d.dir(e.name), `${prefix}${e.name}/`, out);
}

/** The run folder as a zip, every file under its relative path (`cnmf/C.npy`, ...). The archive
 * is a Blob of headers and the files themselves, so nothing of a long run's traces is copied. */
export async function runZip(r: Pick<RunInfo, "dir">): Promise<Blob> {
  const entries: { name: string; data: Blob }[] = [];
  await collect(r.dir, "", entries);
  return zipBlob(entries);
}
