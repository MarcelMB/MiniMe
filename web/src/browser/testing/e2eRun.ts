import { cpSync, existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Folders } from "../folders";
import { memoryKV } from "../kv";
import { Registry } from "../runs";
import { configure } from "../state";
import { NodeDir } from "./nodeDir";

export const DATA = resolve(__dirname, "../../../e2e/.data");

/** The e2e run as a browser root: `<name>/raw`, `<name>/meta.json`, `<name>/processed/t`. */
export async function setupE2eRun(
  run = join(DATA, "processed/t"),
): Promise<{ rid: string; name: string; raw: string }> {
  const top = mkdtempSync(join(tmpdir(), "e2e-"));
  cpSync(join(DATA, "raw"), join(top, "raw"), { recursive: true });
  cpSync(join(DATA, "meta.json"), join(top, "meta.json"));
  cpSync(run, join(top, "processed/t"), { recursive: true, preserveTimestamps: true }); // the reruns are listed by age;
  const folders = new Folders(memoryKV(), async () => null);
  const name = await folders.add(new NodeDir(top), "picked");
  const reruns = join(top, "processed/t/reruns");
  const rerunDirs = existsSync(reruns) ? readdirSync(reruns).map((d) => join(reruns, d)) : [];
  for (const dir of [join(top, "processed/t"), ...rerunDirs]) {
    const pj = join(dir, "params.json");
    if (existsSync(pj))
      writeFileSync(pj, JSON.stringify({ ...JSON.parse(readFileSync(pj, "utf8")), raw_dir: `${name}/raw` }));
  }
  const cacheRoot = new NodeDir(mkdtempSync(join(tmpdir(), "cache-")));
  const registry = new Registry(folders, memoryKV(), async () => cacheRoot);
  configure({ folders, registry, cacheRoot });
  return { rid: (await registry.open(`${name}/processed/t`)).id, name, raw: `${name}/raw` };
}
