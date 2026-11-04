import type { Dir } from "./dir";

/** `io._leading_number`: the number a text starts with ("30", "29.97 fps"), no sign or exponent. */
function leadingNumber(v: unknown): number | null {
  const m = /^\s*([0-9]+(?:\.[0-9]+)?)/.exec(String(v));
  return m ? Number(m[1]) : null;
}

const fileOrNull = (d: Dir, name: string): Promise<File | null> =>
  d.file(name).then(
    (f) => f,
    () => null,
  );

/** The JSON object in `f`, or undefined when it does not parse. */
async function parsed(f: File): Promise<Record<string, unknown> | undefined> {
  try {
    const v = JSON.parse(await f.text());
    return v && typeof v === "object" ? v : undefined;
  } catch {
    return undefined;
  }
}

/** `io.fps_from_files`: `fps` of `../meta.json` (`parent` is the raw folder's parent), else
 * `frameRate` of a Miniscope metadata file in `raw`. As in main, a file that is there but does
 * not parse (`meta.json` also: has no `fps`) answers null instead of trying the next one. */
export async function fpsFromFolder(raw: Dir | null, parent: Dir | null): Promise<number | null> {
  const meta = parent ? await fileOrNull(parent, "meta.json") : null;
  if (meta) {
    const v = Number((await parsed(meta))?.fps);
    return Number.isFinite(v) ? v : null;
  }
  if (!raw) return null;
  for (const name of ["metaData.json", "neural_metaData.json"]) {
    const f = await fileOrNull(raw, name);
    if (!f) continue;
    const doc = await parsed(f);
    if (!doc) return null;
    const v = leadingNumber(doc.frameRate);
    if (v) return v;
  }
  return null;
}
