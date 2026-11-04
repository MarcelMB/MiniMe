import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";

// The browser build on the e2e raw folder (300 frames): a run is started through the in-page
// API (the New run form is main's; the folder dialog is the part Playwright cannot drive), the
// job page follows it, and the finished run opens in the viewer. Same small-session parameters
// as `e2e/make_run.py`.
const RAW = "e2e/.data/raw";
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

async function expectEveryPage(page: Page): Promise<void> {
  await expect(page.getByRole("heading", { name: "e2e", exact: true })).toBeVisible();
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
