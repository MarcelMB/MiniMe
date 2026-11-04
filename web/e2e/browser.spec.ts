import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";

// The browser build on the e2e raw folder (300 frames): a run is started through the in-page
// API (the New run form is main's; the folder dialog is the part Playwright cannot drive), the
// job page follows it, and the finished run opens in the viewer. Same small-session parameters
// as `e2e/make_run.py`.
const RAW = "e2e/.data/raw";
/** Python for `minime score` (as `playwright.config.ts` finds it), run from this branch's `src`. */
const PY = process.env.MINIME_PYTHON ?? resolve(process.cwd(), "../.venv/bin/python");
const PARAMS = {
  tile: 32,
  seeds_wnd: 100,
  seeds_stp: 50,
  seeds_max_wnd: 8,
  template_frames: 50,
  inita_wnd: 8,
  dl_wnd: 5,
  size_thres: [9, null],
  svd_k0: 16,
  motion_before_tophat: false,
};
const PAGES: [string, (page: Page) => Locator][] = [
  ["Overview", (p) => p.getByText("From seeds to units")],
  ["Frames", (p) => p.getByRole("region", { name: "Reconstruction" })],
  ["Motion", (p) => p.locator(".uplot").first()],
  ["Compression", (p) => p.getByText("Total rank")],
  ["Seeds", (p) => p.getByText("Seeds per step")],
  ["Units", (p) => p.locator("table.data tbody tr").first()], // the run found units
  ["Traces", (p) => p.getByText("Add unit", { exact: true })],
];

/** Open the browser build with the e2e hooks and copy the raw folder into browser storage;
 * returns its root name (a vpath starts with it). With `n` > 3 the folder holds `n` files, the
 * three of the e2e raw folder over and over (`0.avi` ... `<n-1>.avi`, 100 frames each). */
async function openApp(page: Page, n = 3): Promise<string> {
  await page.goto("/?e2e=1#/");
  await page.waitForFunction(() => "__minimeAddFolder" in window && "__minimeApi" in window);
  const files = readdirSync(RAW).map((name) => readFileSync(join(RAW, name)).toString("base64"));
  return page.evaluate(
    async ({ files, n }) => {
      const list = Array.from({ length: n }, (_, i) => {
        const b = files[i % files.length];
        return new File([Uint8Array.from(atob(b), (c) => c.charCodeAt(0))], `${i}.avi`);
      });
      // biome-ignore lint/suspicious/noExplicitAny: test hook
      return (await (window as any).__minimeAddFolder("synth", list)) as string;
    },
    { files, n },
  );
}

/** Start the job `name` on the raw folder (the tile store kept) and show its page. */
async function startJob(
  page: Page,
  root: string,
  name: string,
  params: Record<string, unknown>,
): Promise<{ id: string; out_dir: string }> {
  const job = await page.evaluate(
    async ({ raw, name, params }) => {
      // biome-ignore lint/suspicious/noExplicitAny: test hook
      return (await (window as any).__minimeApi.postJSON("/jobs", {
        raw_dir: raw,
        name,
        params,
        keep_store: true,
        overwrite: true,
        tune: false,
        tune_args: {},
        fps: 30,
      })) as { id: string; out_dir: string };
    },
    { raw: `${root}/raw`, name, params },
  );
  expect(job.out_dir).toBe(`${root}/processed/${name}`);
  await page.evaluate((id) => {
    window.location.hash = `#/job/${id}`;
  }, job.id);
  return job;
}

test("New run → job page → Open run → every page", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  const root = await openApp(page);

  // the raw folder's videos are indexed in a worker (avi.worker.ts), which only the built page runs
  const info = await page.evaluate(
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    (raw) => (window as any).__minimeApi.getJSON(`/raw/info?path=${encodeURIComponent(raw)}`),
    `${root}/raw`,
  );
  expect(info).toMatchObject({
    path: `${root}/raw`,
    segments: 3,
    frames: 300,
    height: 64,
    width: 64,
    names: ["0.avi", "1.avi", "2.avi"],
  });

  const job = await startJob(page, root, "e2e", PARAMS);

  const openRun = page.getByRole("button", { name: "Open run" });
  const failed = page.getByRole("alert");
  await expect(openRun.or(failed)).toBeVisible({ timeout: 300_000 });
  await expect(failed).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Stages" })).toContainText("Streaming pass");

  // the run folder the worker wrote carries the raw folder and the frame rate
  const params = await page.evaluate(async (path) => {
    let d = await navigator.storage.getDirectory();
    for (const p of path.split("/")) d = await d.getDirectoryHandle(p);
    return JSON.parse(await (await (await d.getFileHandle("params.json")).getFile()).text());
  }, `e2e/${root}/processed/e2e`);
  expect(params).toMatchObject({ raw_dir: `${root}/raw`, fps: 30, tile: 32 });

  await openRun.click();
  await expectEveryPage(page);

  // after a reload the finished job is still there, and so is its run
  await page.reload();
  await page.evaluate((id) => {
    window.location.hash = `#/job/${id}`;
  }, job.id);
  await expect(page.getByText("Run · done")).toBeVisible();
  // its stage rows come back from the run folder's progress.jsonl, as main replays them
  await expect(page.getByRole("region", { name: "Stages" })).toContainText("Streaming pass");
  await expect(page.getByText("Starting…")).toHaveCount(0);
  await page.getByRole("button", { name: "Open run" }).click();
  await expect(page.getByRole("heading", { name: "e2e", exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

// 4500 frames (the e2e raw folder's three files, fifteen times over) make two windows, 3000 and
// 1500 frames: the tile store and the factors of the second window, not only the first, reach the
// run folder and answer the viewer at that window's own frame numbers.
test("a run of two windows keeps each window's tile store and factors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  const root = await openApp(page, 45);
  await startJob(page, root, "e2e2w", PARAMS);
  const openRun = page.getByRole("button", { name: "Open run" });
  const failed = page.getByRole("alert");
  await expect(openRun.or(failed)).toBeVisible({ timeout: 540_000 });
  await expect(failed).toHaveCount(0);

  // browser storage: tiles/wNN/ holds a file per tile (32 x 32 px, one byte per frame and pixel)
  const run = `e2e/${root}/processed/e2e2w`;
  const sizes = (kind: string) =>
    page.evaluate(
      async ({ run, kind }) => {
        let d = await navigator.storage.getDirectory();
        for (const p of `${run}/${kind}`.split("/")) d = await d.getDirectoryHandle(p);
        const out: Record<string, Record<string, number>> = {};
        // biome-ignore lint/suspicious/noExplicitAny: entries() is missing from the DOM lib
        for await (const [name, h] of (d as any).entries() as AsyncIterable<[string, FileSystemHandle]>) {
          if (h.kind !== "directory") continue;
          out[name] = {};
          // biome-ignore lint/suspicious/noExplicitAny: entries() is missing from the DOM lib
          for await (const [f, fh] of (h as any).entries() as AsyncIterable<[string, FileSystemHandle]>)
            if (fh.kind === "file") out[name][f] = (await (fh as FileSystemFileHandle).getFile()).size;
        }
        return out;
      },
      { run, kind },
    );
  const tiles = await sizes("tiles");
  const store = (frames: number) =>
    Object.fromEntries(Array.from({ length: 4 }, (_, k) => [`tile_00${k}.part00`, frames * 32 * 32]));
  expect(tiles).toEqual({ w00: store(3000), w01: store(1500) });
  const factors = await sizes("factors");
  expect(Object.keys(factors).sort()).toEqual(["w00", "w01"]);
  for (const w of ["w00", "w01"])
    expect(Object.keys(factors[w]).sort()).toEqual(
      expect.arrayContaining(["V.npy", "W.npy", "block_max.npy", "mean.npy", "rank.npy"]),
    );

  // the views at frames of each window: the reconstruction follows the preprocessed frame, and
  // the residual has the blocks of both windows with no leftover shift
  const api = (path: string) =>
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    page.evaluate((p) => (window as any).__minimeApi.getArray(p), path) as Promise<{
      shape: number[];
      dtype: string;
      data: number[];
    }>;
  const { runs } = (await page.evaluate(
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    (r) => (window as any).__minimeApi.getJSON(`/runs?root=${encodeURIComponent(r)}`),
    `${root}/processed/e2e2w`,
  )) as { runs: { id: string }[] };
  const rid = runs[0].id;
  for (const t of [100, 2999, 3000, 4400]) {
    const recon = await api(`/runs/${rid}/frame/${t}?source=recon`);
    const pre = await api(`/runs/${rid}/frame/${t}?source=pre`);
    expect([recon.shape, recon.dtype, pre.shape, pre.dtype]).toEqual([[64, 64], "uint8", [64, 64], "uint8"]);
    const r = pearson(recon.data, pre.data);
    console.log(`frame ${t}: recon vs pre Pearson ${r.toFixed(4)}`);
    expect(r, `frame ${t}`).toBeGreaterThan(0.9);
  }
  const resid = await api(`/runs/${rid}/motion/residual`);
  expect(resid.shape[1]).toBe(3);
  expect(resid.shape[0]).toBeGreaterThan(2);
  // the footage has no drift: the typical block sits where the session's max projection does. A
  // block with little in common with it (a few dozen frames) can lock on a wrong peak, as in main's
  // own residual of this footage (block 5 of the 300 frames: 16.5 px at a correlation of 0.29)
  const shifts = Array.from({ length: resid.shape[0] }, (_, b) =>
    Math.max(Math.abs(resid.data[3 * b]), Math.abs(resid.data[3 * b + 1])),
  ).sort((a, b) => a - b);
  const median = shifts[Math.floor(shifts.length / 2)];
  console.log(
    `residual: ${shifts.length} blocks, median shift ${median.toFixed(3)} px, ${shifts.filter((s) => s >= 1).length} blocks shifted by 1 px or more, largest ${shifts.at(-1)?.toFixed(3)} px`,
  );
  expect(median).toBeLessThan(1);
  expect(errors).toEqual([]);
});

async function expectEveryPage(page: Page, name = "e2e"): Promise<void> {
  await expect(page.getByRole("heading", { name, exact: true })).toBeVisible();
  const stage = page.getByRole("group", { name: "Stage" });
  for (const [p, marker] of PAGES) {
    const button = stage.getByRole("button", { name: p });
    await button.click();
    await expect(button).toHaveAttribute("aria-pressed", "true");
    await expect(marker(page).first()).toBeVisible({ timeout: 60_000 });
    if (p === "Frames") await expectFrames(page);
    if (p === "Motion") await expectResidual(page);
    await page.waitForLoadState("networkidle");
    expect(await page.getByRole("alert").allTextContents(), `alerts on ${p}`).toEqual([]);
  }
  await expect(stage.getByRole("button", { name: "Tune" })).toHaveCount(0);
}

/** The gray values (the red channel) of a canvas. */
const grays = (canvas: Locator): Promise<number[]> =>
  canvas.evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    return Array.from({ length: d.length / 4 }, (_, i) => d[4 * i]);
  });

const pearson = (a: number[], b: number[]): number => {
  const [ma, mb] = [a.reduce((s, v) => s + v, 0) / a.length, b.reduce((s, v) => s + v, 0) / b.length];
  let [ab, aa, bb] = [0, 0, 0];
  a.forEach((x, i) => {
    ab += (x - ma) * (b[i] - mb);
    aa += (x - ma) ** 2;
    bb += (b[i] - mb) ** 2;
  });
  return ab / Math.sqrt(aa * bb);
};

/** The Frames page draws the preprocessed frame (the tile store the job kept) and the
 * reconstruction (the window factors) as pictures with no note under them, and the second follows
 * the first (the reconstruction approximates the preprocessed frame). */
async function expectFrames(page: Page): Promise<void> {
  const pictures: number[][] = [];
  for (const [section, image] of [
    ["Preprocessed", "Preprocessed frame"],
    ["Reconstruction", "Reconstruction frame"],
  ]) {
    const canvas = page.getByRole("img", { name: image });
    await expect(canvas, `${section} frame`).toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("region", { name: section }).locator(".muted")).toHaveCount(0);
    pictures.push(await grays(canvas));
    expect(new Set(pictures.at(-1)).size, `${section} frame has a picture`).toBeGreaterThan(10);
  }
  const r = pearson(pictures[0], pictures[1]);
  console.log(`Frames page: reconstruction vs preprocessed frame, Pearson ${r.toFixed(4)}`);
  expect(r, "reconstruction vs preprocessed frame").toBeGreaterThan(0.9);
}

/** The Motion page's residual (computed from the windows' block maxima on the first request)
 * is drawn as its two charts. */
async function expectResidual(page: Page): Promise<void> {
  const residual = page.getByRole("region", { name: "Residual shift per block" });
  await expect(residual.locator(".uplot")).toHaveCount(2, { timeout: 60_000 });
}

/** Copy the e2e raw videos into Browser storage itself (OPFS `runs/<folder>/raw/`): a run on
 * them is written beside them, into Browser storage. */
async function putRawInBrowserStorage(page: Page, folder: string): Promise<void> {
  const files = readdirSync(RAW).map((name) => [name, readFileSync(join(RAW, name)).toString("base64")]);
  await page.evaluate(
    async ({ files, folder }) => {
      let d = await navigator.storage.getDirectory();
      for (const p of ["runs", folder, "raw"]) d = await d.getDirectoryHandle(p, { create: true });
      for (const [name, b64] of files) {
        const w = await (await d.getFileHandle(name, { create: true })).createWritable();
        await w.write(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
        await w.close();
      }
    },
    { files, folder },
  );
}

// A run in Browser storage (the raw videos are in it) has a "Download run (.zip)" button. The
// zip unzips to the run folder: main's `minime score` reads it, and, picked as a folder upload in
// a browser that has no copy of the run and no folder dialog (a read-only folder), it opens in the
// viewer with every page.
test("a run in Browser storage downloads as a zip that minime score reads and that reopens read-only", async ({
  page,
  browser,
  baseURL,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("/?e2e=1#/");
  await page.waitForFunction(() => "__minimeApi" in window);
  await putRawInBrowserStorage(page, "dl");
  const job = await startJob(page, "Browser storage/dl", "dl", PARAMS);
  const openRun = page.getByRole("button", { name: "Open run" });
  const failed = page.getByRole("alert");
  await expect(openRun.or(failed)).toBeVisible({ timeout: 300_000 });
  await expect(failed).toHaveCount(0);
  expect(job.out_dir).toBe("Browser storage/dl/processed/dl");
  await openRun.click();
  await expect(page.getByRole("heading", { name: "dl", exact: true })).toBeVisible();

  // the zip is put together before the download starts: the button says so and takes no second click
  await page.evaluate(() => {
    const seen: string[] = [];
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    (window as any).__downloadButton = seen;
    new MutationObserver(() => {
      for (const b of document.querySelectorAll("button"))
        if (b.textContent === "Preparing…") seen.push(`${b.textContent}${b.disabled ? " disabled" : ""}`);
    }).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true });
  });
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download run (.zip)" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("dl.zip");
  await expect(page.getByRole("button", { name: "Download run (.zip)" })).toBeEnabled();
  // biome-ignore lint/suspicious/noExplicitAny: test hook
  expect(await page.evaluate(() => (window as any).__downloadButton)).toContain("Preparing… disabled");
  const unzipped = "e2e/.data/download";
  rmSync(unzipped, { recursive: true, force: true });
  mkdirSync(unzipped, { recursive: true });
  await download.saveAs("e2e/.data/download.zip");
  execFileSync("unzip", ["-q", "e2e/.data/download.zip", "-d", unzipped]);
  const files = readdirSync(unzipped);
  expect(files).toEqual(expect.arrayContaining(["params.json", "report.json", "cnmf", "factors", "motion"]));
  execFileSync(PY, ["-m", "minime", "score", unzipped, "--raw", RAW], {
    env: { ...process.env, PYTHONPATH: "../src" },
    stdio: "pipe",
  });

  // another browser (no OPFS copy of the run, no folder dialog): the unzipped folder is an upload
  const other = await browser.newContext({ baseURL });
  await other.addInitScript(() => {
    // biome-ignore lint/suspicious/noExplicitAny: Chromium-only API, removed to look like Firefox or Safari
    delete (window as any).showDirectoryPicker;
  });
  const page2 = await other.newPage();
  page2.on("pageerror", (e) => errors.push(e.message));
  page2.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page2.goto("/#/");
  await page2.getByRole("button", { name: "Change folder" }).click();
  await expect(page2.getByText("This browser opens folders read-only")).toBeVisible();
  await page2.locator('input[type="file"]').setInputFiles(unzipped);
  await page2.getByRole("link", { name: "Open run download" }).click();
  await expect(page2.getByRole("group", { name: "Stage" })).toBeVisible(); // the run page, not Home's card of the same name
  await expectEveryPage(page2, "download");
  await expect(page2.getByRole("button", { name: "Download run (.zip)" })).toHaveCount(0); // not in Browser storage
  await other.close();
  expect(errors).toEqual([]);
});

// Playwright cannot drive the system folder dialog, so `showDirectoryPicker` is replaced by a
// function that hands out OPFS folders. A raw folder picked on its own (the videos are in the
// folder itself) has nothing beside it: Start asks for a second folder, the run goes there and
// the raw folder is not written to; picking a folder again gives the root it already was.
test("a raw folder picked on its own asks for an output folder and the run goes there", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("/?e2e=1#/");
  await page.waitForFunction(() => "__minimeApi" in window);
  const files = readdirSync(RAW).map((name) => [name, readFileSync(join(RAW, name)).toString("base64")]);
  await page.evaluate(
    async ({ files, hand }) => {
      const e2e = await (await navigator.storage.getDirectory()).getDirectoryHandle("e2e", { create: true });
      const raw = await e2e.getDirectoryHandle("rawonly", { create: true });
      await e2e.getDirectoryHandle("outonly", { create: true });
      for (const [name, b64] of files) {
        const w = await (await raw.getFileHandle(name, { create: true })).createWritable();
        await w.write(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
        await w.close();
      }
      const asked: unknown[] = [];
      const queue = [...hand];
      // biome-ignore lint/suspicious/noExplicitAny: test hooks on window
      const w = window as any;
      w.__dialogs = asked;
      w.__hand = queue;
      w.showDirectoryPicker = async (opts: unknown) => {
        asked.push(opts);
        let d = e2e;
        for (const p of (queue.shift() ?? "none").split("/")) d = await d.getDirectoryHandle(p);
        return d;
      };
    },
    { files, hand: ["rawonly", "outonly", "outonly"] },
  );
  const dialogs = () => page.evaluate(() => (window as unknown as { __dialogs: unknown[] }).__dialogs);
  const pickFolder = async () => {
    await page.getByRole("button", { name: "Change folder" }).click();
    await page.getByRole("button", { name: "Show runs here" }).click();
  };

  // Runs → Change folder → the dialog: the raw folder becomes a root, writable (asked for at once)
  await pickFolder();
  await expect(page.locator("h1.mono")).toHaveText("rawonly");
  expect(await dialogs()).toEqual([{ id: "minime", mode: "readwrite" }]);

  // Start: nothing is beside the videos, so the output folder is asked for
  const job = await page.evaluate(
    async ({ params }) =>
      // biome-ignore lint/suspicious/noExplicitAny: test hook
      (await (window as any).__minimeApi.postJSON("/jobs", {
        raw_dir: "rawonly",
        name: "oo",
        params,
        keep_store: false,
        overwrite: false,
        tune: false,
        tune_args: {},
        fps: 30,
      })) as { id: string; out_dir: string },
    { params: PARAMS },
  );
  expect(job.out_dir).toBe("outonly/oo");
  expect(await dialogs()).toEqual([
    { id: "minime", mode: "readwrite" },
    { id: "minime-out", mode: "readwrite" },
  ]);
  await page.evaluate((id) => {
    window.location.hash = `#/job/${id}`;
  }, job.id);
  const openRun = page.getByRole("button", { name: "Open run" });
  const failed = page.getByRole("alert");
  await expect(openRun.or(failed)).toBeVisible({ timeout: 300_000 });
  await expect(failed).toHaveCount(0);

  // the run is in the output folder; the raw folder holds its videos and nothing else
  const entries = (name: string) =>
    page.evaluate(async (name) => {
      let d = await navigator.storage.getDirectory();
      for (const p of ["e2e", name]) d = await d.getDirectoryHandle(p);
      const out: string[] = [];
      // biome-ignore lint/suspicious/noExplicitAny: entries() is missing from the DOM lib
      for await (const [n] of (d as any).entries() as AsyncIterable<[string, FileSystemHandle]>) out.push(n);
      return out.sort();
    }, name);
  expect(await entries("rawonly")).toEqual(files.map(([name]) => name));
  expect(await entries("outonly")).toEqual(["oo"]);

  // the folder the run went to is a root now; picking it again is the same root, with its run listed
  await page.evaluate(() => {
    window.location.hash = "#/";
  });
  await pickFolder();
  await expect(page.locator("h1.mono")).toHaveText("outonly");
  await expect(page.getByRole("link", { name: "Open run oo" })).toBeVisible();
  const roots = await page.evaluate(
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    async () => ((await (window as any).__minimeApi.getJSON("/folders")) as { folders: { path: string }[] }).folders,
  );
  expect(roots.map((r) => r.path).sort()).toEqual(["Browser storage", "outonly", "rawonly"]);

  // an output folder that is, or was made inside, the raw folder is refused (the roots' names say
  // nothing about that; the handles do) and is not remembered
  for (const inside of ["rawonly", "rawonly/new folder"]) {
    const refused = await page.evaluate(async (inside) => {
      const e2e = await (await navigator.storage.getDirectory()).getDirectoryHandle("e2e");
      let d = e2e;
      for (const p of inside.split("/")) d = await d.getDirectoryHandle(p, { create: true });
      // biome-ignore lint/suspicious/noExplicitAny: test hooks on window
      const w = window as any;
      w.__hand.push(inside);
      try {
        await w.__minimeApi.postJSON("/jobs", {
          raw_dir: "rawonly",
          name: "bad",
          params: {},
          keep_store: false,
          overwrite: false,
          tune: false,
          tune_args: {},
          fps: 30,
        });
        return "";
      } catch (e) {
        return (e as Error).message;
      }
    }, inside);
    expect(refused).toBe(`output dir ${inside.split("/").pop()}/bad lies inside the raw dir rawonly; choose another`);
  }
  const after = await page.evaluate(
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    async () => ((await (window as any).__minimeApi.getJSON("/folders")) as { folders: { path: string }[] }).folders,
  );
  expect(after.map((r) => r.path).sort()).toEqual(["Browser storage", "outonly", "rawonly"]);
  expect(errors).toEqual([]);
});

test("a browser without OPFS gets a page that says so instead of the app", async ({ page }) => {
  await page.addInitScript(() => {
    // biome-ignore lint/suspicious/noExplicitAny: removed to look like a browser without OPFS
    delete (StorageManager.prototype as any).getDirectory;
  });
  await page.goto("/#/");
  await expect(page.getByRole("heading", { name: "This browser cannot run MiniMe" })).toBeVisible();
  await expect(page.getByText("recent Chrome, Edge, Firefox or Safari")).toBeVisible();
  await expect(page.getByRole("link", { name: "New run" })).toHaveCount(0); // the app is gone
});
