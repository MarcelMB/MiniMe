import { cpSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { getJSON, postJSON } from "../api";
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
