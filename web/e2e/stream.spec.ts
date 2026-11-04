import { appendFileSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { expect, type Page, test } from "@playwright/test";

// Runs the browser pipeline on raw AVIs and saves its outputs for the
// experiments/streaming/check_*.py scripts:
//   STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm run e2e:stream
// STREAM_KEEP_STORE=1 keeps each window's preprocessed tile store in the run folder (`tiles/wNN/`,
// the frames' size again, and exported with it): the viewer's preprocessed frames need it.
// Without STREAM_INPUTS the AVIs alone are picked (the whole pipeline, default parameters), and
// the MiniMe run folder the page wrote to browser storage (`runs/<name>`) is saved to
// <STREAM_OUT>/run, which `minime score` and `scripts/contract_dump.py` open.
// The worker is started through the e2e hook `window.__minimeStream` (`src/browser/e2eStream.ts`,
// the browser build opened with `?e2e=1`); the AVIs reach the page as disk-backed `File`s of file
// inputs (`setInputFiles`), never as bytes in a message: sessions are several GB.
const raw = process.env.STREAM_RAW;
const inputs = process.env.STREAM_INPUTS;
const out = process.env.STREAM_OUT;
const CHUNK = 8 * 2 ** 20; // bytes per CDP message when a file is copied out of the page

/** Copy the files of OPFS `runs/<name>/` (the one run the page wrote) to `dest`. */
async function exportRun(page: Page, dest: string): Promise<string[]> {
  const files = await page.evaluate(async () => {
    const found: { path: string; size: number }[] = [];
    // biome-ignore lint/suspicious/noExplicitAny: entries() is missing from the DOM lib
    const walk = async (d: any, prefix: string): Promise<void> => {
      for await (const [name, h] of d.entries()) {
        if (h.kind === "directory") await walk(h, `${prefix}${name}/`);
        else found.push({ path: `${prefix}${name}`, size: (await h.getFile()).size });
      }
    };
    await walk(await (await navigator.storage.getDirectory()).getDirectoryHandle("runs"), "");
    return found;
  });
  const names = new Set(files.map((f) => f.path.split("/")[0]));
  expect([...names], "one run in browser storage").toHaveLength(1);
  for (const f of files) {
    const target = join(dest, ...f.path.split("/").slice(1));
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, "");
    for (let off = 0; off < f.size; off += CHUNK) {
      const b64 = await page.evaluate(
        async ({ path, start, len }) => {
          let d = await (await navigator.storage.getDirectory()).getDirectoryHandle("runs");
          const parts = path.split("/");
          for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p);
          const file = await (await d.getFileHandle(parts[parts.length - 1])).getFile();
          return await new Promise<string>((res) => {
            const r = new FileReader();
            r.onload = () => res(String(r.result).split(",")[1] ?? "");
            r.readAsDataURL(file.slice(start, start + len));
          });
        },
        { path: f.path, start: off, len: CHUNK },
      );
      appendFileSync(target, Buffer.from(b64, "base64"));
    }
  }
  return files.map((f) => f.path.split("/").slice(1).join("/"));
}

/** Add a file input to the page (`webkitdirectory` for a folder) and give it `paths`. */
async function inputFiles(page: Page, id: string, paths: string[], folder = false): Promise<void> {
  await page.evaluate(
    ({ id, folder }) => {
      const input = document.createElement("input");
      input.type = "file";
      input.id = id;
      input.multiple = true;
      if (folder) input.setAttribute("webkitdirectory", "");
      document.body.append(input);
    },
    { id, folder },
  );
  await page.locator(`#${id}`).setInputFiles(paths);
}

/** Copy the outputs the page kept (`__minimeStream`'s `done` files) to `dest`. */
async function saveOutputs(page: Page, files: { name: string; size: number }[], dest: string): Promise<void> {
  mkdirSync(dest, { recursive: true });
  for (const f of files) {
    const target = join(dest, f.name);
    writeFileSync(target, "");
    for (let off = 0; off < f.size; off += CHUNK) {
      const b64 = await page.evaluate(({ name, start, len }) => window.__minimeStreamRead!(name, start, len), {
        name: f.name,
        start: off,
        len: CHUNK,
      });
      appendFileSync(target, Buffer.from(b64, "base64"));
    }
  }
}

test("streams raw AVIs in the browser and saves the outputs", async ({ page }) => {
  test.skip(!raw || !out, "set STREAM_RAW and STREAM_OUT (and STREAM_INPUTS for a reference folder)");
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("worker", (w) => console.log(`worker started: ${w.url()}`));
  page.on("console", (m) => m.text().startsWith("[worker]") && console.log(m.text()));
  page.on("console", (m) => m.text().startsWith("[stream]") && console.log(m.text()));
  await page.goto("/?e2e=1#/");
  await page.waitForFunction(() => "__minimeStream" in window);
  const avis = readdirSync(raw!)
    .filter((f) => /^\d+.*\.avi$/i.test(f))
    .sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10)) // STREAM_FILES takes the first n
    .map((f) => join(raw!, f));
  const nFiles = Number(process.env.STREAM_FILES ?? avis.length);
  await inputFiles(page, "avis", avis.slice(0, nFiles));
  if (inputs) await inputFiles(page, "inputs", [inputs], true);
  const keepStore = Boolean(process.env.STREAM_KEEP_STORE);
  // resolves when the run is done (hours on a long session); a failed run rejects with its message
  const done = await page.evaluate(
    ({ keepStore }) => {
      const files = (id: string) => [...((document.getElementById(id) as HTMLInputElement | null)?.files ?? [])];
      return window.__minimeStream!(files("avis"), files("inputs"), { keepStore });
    },
    { keepStore },
  );
  await saveOutputs(page, done.files, out!);
  console.log(`done in ${done.seconds.toFixed(0)} s: ${done.files.length} output files -> ${out}`);
  if (!inputs) {
    const written = await exportRun(page, join(out!, "run"));
    console.log(`run folder: ${written.length} files -> ${join(out!, "run")}`);
    expect(written).toEqual(expect.arrayContaining(["params.json", "report.json", "cnmf/A.npz", "cnmf/C.npy"]));
  }
  expect(errors).toEqual([]);
});
