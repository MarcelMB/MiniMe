import { cpSync, mkdirSync, mkdtempSync, readFileSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { getJSON, postJSON } from "../api";
import { RunCache } from "./cache";
import type { Dir } from "./dir";
import { Folders } from "./folders";
import { memoryKV } from "./kv";
import { Registry } from "./runs";
import { configure } from "./state";
import { NodeDir } from "./testing/nodeDir";

const E2E = resolve(__dirname, "../../e2e/.data/processed/t");

async function setup() {
  const top = mkdtempSync(join(tmpdir(), "runs-"));
  cpSync(E2E, join(top, "s1/processed/t"), { recursive: true });
  cpSync(E2E, join(top, "s2/processed/u"), { recursive: true });
  mkdirSync(join(top, ".hidden/processed/x"), { recursive: true });
  writeFileSync(join(top, ".hidden/processed/x/report.json"), "{}");
  mkdirSync(join(top, "a/b/c/d/e/f"), { recursive: true }); // deeper than 4: not searched
  const folders = new Folders(memoryKV(), async () => null);
  const name = await folders.add(new NodeDir(top), "picked");
  return { name, top, folders, reg: new Registry(folders, memoryKV()) };
}

describe("Registry", () => {
  it("opens a rerun folder with its source run's factors", async () => {
    const { name, top, reg } = await setup();
    const run = join(top, "s1/processed/t");
    const rerun = join(run, "reruns/r1");
    mkdirSync(join(rerun, "cnmf"), { recursive: true });
    mkdirSync(join(rerun, "factors"), { recursive: true });
    cpSync(join(run, "factors/meta.json"), join(rerun, "factors/meta.json"));
    const params = JSON.parse(readFileSync(join(run, "params.json"), "utf8"));
    const src = `${name}/s1/processed/t`;
    writeFileSync(join(rerun, "params.json"), JSON.stringify({ ...params, source_run: src }));
    writeFileSync(join(rerun, "report.json"), JSON.stringify({ stages: [], counts: {} }));
    const r = await reg.open(`${src}/reruns/r1`);
    expect(r.sourceRun).toBe(src);
    expect((r.factors as NodeDir).path).toBe(run);
    expect((r.dir as NodeDir).path).toBe(rerun);
    const plain = await reg.open(src);
    expect(plain.sourceRun).toBeNull();
    expect((plain.factors as NodeDir).path).toBe(run);
    // find never lists a rerun
    expect((await reg.find(name)).sort()).toEqual([src, `${name}/s2/processed/u`]);
  });

  it("finds runs below a root, never in hidden folders", async () => {
    const { name, reg } = await setup();
    const found = await reg.find(name);
    expect(found.sort()).toEqual([`${name}/s1/processed/t`, `${name}/s2/processed/u`]);
  });

  it("gives each run a stable id", async () => {
    const { name, reg } = await setup();
    const a = await reg.open(`${name}/s1/processed/t`);
    const b = await reg.open(`${name}/s1/processed/t`);
    expect(a.id).toMatch(/^[0-9a-f]{12}$/);
    expect(b.id).toBe(a.id);
    expect((await reg.get(a.id)).vpath).toBe(`${name}/s1/processed/t`);
    await expect(reg.get("000000000000")).rejects.toMatchObject({ status: 404 });
  });
});

/** Records what is removed from it, to count the drops of a run's cache. */
class CountingDir extends NodeDir {
  removed: string[] = [];
  async remove(name: string, opts?: { recursive?: boolean }): Promise<void> {
    this.removed.push(name);
    return super.remove(name, opts);
  }
}

describe("stale run cache", () => {
  const cached = async (cache: Dir, id: string) => (await cache.entries()).some((e) => e.name === id);

  it("drops a run's cache when report.json is rewritten, and only then", async () => {
    const { name, top, folders } = await setup();
    const cache = new CountingDir(mkdtempSync(join(tmpdir(), "cache-")));
    const reg = new Registry(folders, memoryKV(), async () => cache);
    const vpath = `${name}/s1/processed/t`;
    const report = join(top, "s1/processed/t/report.json");

    const first = await reg.open(vpath);
    expect(first.stamp).toMatch(/^[\d.]+-\d+$/);
    const views = new RunCache(await cache.dir(first.id, { create: true }));
    await views.saveJSON("units.json", { units: 15 });
    await reg.setFps(first, 25);
    expect(cache.removed).toEqual([]); // first sight only stores the stamp

    // nothing changed: the same stamp, the cache stays
    expect((await reg.open(vpath)).stamp).toBe(first.stamp);
    expect((await reg.get(first.id)).stamp).toBe(first.stamp);
    expect(await views.has("units.json")).toBe(true);

    // a rewrite that changes the size: the cache goes (three requests at once drop it once), the id and
    // the frame rate set in the viewer stay
    writeFileSync(report, `${readFileSync(report, "utf8")} `);
    const [again] = await Promise.all([reg.open(vpath), reg.get(first.id), reg.open(vpath)]);
    expect(again.stamp).not.toBe(first.stamp);
    expect(again.id).toBe(first.id);
    expect(await cached(cache, first.id)).toBe(false);
    expect(cache.removed).toEqual([first.id]);
    expect(await reg.fps(again)).toBe(25);

    // new views are kept again until the next rewrite
    const fresh = new RunCache(await cache.dir(first.id, { create: true }));
    await fresh.saveJSON("units.json", { units: 15 });
    expect((await reg.open(vpath)).stamp).toBe(again.stamp);
    expect(await fresh.has("units.json")).toBe(true);

    // same size, newer modification time: also a rewrite
    const t = new Date(Date.now() + 60_000);
    utimesSync(report, t, t);
    const touched = await reg.open(vpath);
    expect(touched.stamp).not.toBe(again.stamp);
    expect(await cached(cache, first.id)).toBe(false);
    expect(cache.removed).toEqual([first.id, first.id]);
  });
});

describe("run routes", () => {
  it("lists folders and runs, a broken run with its error", async () => {
    const { name, top, folders, reg } = await setup();
    configure({ folders, registry: reg });
    mkdirSync(join(top, "s3/processed/bad"), { recursive: true });
    for (const f of ["params.json", "report.json"]) writeFileSync(join(top, "s3/processed/bad", f), "{}");

    expect(await getJSON("/folders")).toMatchObject({ folders: [{ path: name, kind: "run" }] });
    await expect(getJSON("/fs")).rejects.toMatchObject({ status: 404 });
    type Listed = { id?: string; name: string; path: string; error?: string };
    const { root, runs } = await getJSON<{ root: string; runs: Listed[] }>(`/runs?root=${encodeURIComponent(name)}`);
    expect(root).toBe(name);
    expect(runs.map((r) => r.name).sort()).toEqual(["bad", "t", "u"]);
    expect(runs.find((r) => r.name === "bad")).toMatchObject({
      path: `${name}/s3/processed/bad`,
      error: expect.any(String),
    });
    expect(runs.find((r) => r.name === "t")).toMatchObject({
      id: expect.stringMatching(/^[0-9a-f]{12}$/),
      units: 15,
      T: 300,
    });
    // no root given: the first one, the most recently used
    expect((await getJSON<{ root: string }>("/runs")).root).toBe(name);
  });

  it("sets the frame rate the viewer asks for, with main's messages", async () => {
    const { name, folders, reg } = await setup();
    configure({ folders, registry: reg });
    const rid = (await reg.open(`${name}/s1/processed/t`)).id;
    // raw_dir is a disk path the browser cannot open, so there is no frame rate until one is set
    expect((await getJSON<{ fps: number | null }>(`/runs/${rid}/summary`)).fps).toBeNull();
    await expect(postJSON(`/runs/${rid}/fps`, { fps: 0 })).rejects.toMatchObject({
      status: 400,
      message: "frame rate must be in (0, 1000] Hz, got 0.0",
    });
    await expect(postJSON(`/runs/${rid}/fps`, { fps: 2000 })).rejects.toMatchObject({
      message: "frame rate must be in (0, 1000] Hz, got 2000.0",
    });
    await expect(postJSON(`/runs/${rid}/fps`, { fps: "fast" })).rejects.toMatchObject({ status: 422 });
    await expect(postJSON("/runs/000000000000/fps", { fps: 25 })).rejects.toMatchObject({ status: 404 });
    expect(await postJSON(`/runs/${rid}/fps`, { fps: 25 })).toEqual({ fps: 25 });
    expect((await getJSON<{ fps: number }>(`/runs/${rid}/summary`)).fps).toBe(25);
  });
});
