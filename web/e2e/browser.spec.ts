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
// Not built yet: the Motion page's residual and the Frames page's preprocessed frames come with
// the derived views; until then the page says so in an alert (drop this once they exist).
const NOT_YET = ["the motion residual is not available for this run"];
const PAGES: [string, (page: Page) => Locator][] = [
  ["Overview", (p) => p.getByText("From seeds to units")],
  ["Frames", (p) => p.getByRole("region", { name: "Reconstruction" })],
  ["Motion", (p) => p.locator(".uplot").first()],
  ["Compression", (p) => p.getByText("Total rank")],
  ["Seeds", (p) => p.getByText("Seeds per step")],
  ["Units", (p) => p.locator("table.data tbody tr").first()], // the run found units
  ["Traces", (p) => p.getByText("Add unit", { exact: true })],
];

test("New run → job page → Open run → every page", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("/?e2e=1#/");
  await page.waitForFunction(() => "__minimeAddFolder" in window && "__minimeApi" in window);
  const files = readdirSync(RAW).map((n) => ({ n, b: readFileSync(join(RAW, n)).toString("base64") }));
  const root = await page.evaluate(async (fs) => {
    const list = fs.map(({ n, b }) => new File([Uint8Array.from(atob(b), (c) => c.charCodeAt(0))], n));
    // biome-ignore lint/suspicious/noExplicitAny: test hook
    return (await (window as any).__minimeAddFolder("synth", list)) as string;
  }, files);

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

  const job = await page.evaluate(
    async ({ raw, params }) => {
      // biome-ignore lint/suspicious/noExplicitAny: test hook
      return (await (window as any).__minimeApi.postJSON("/jobs", {
        raw_dir: raw,
        name: "e2e",
        params,
        keep_store: true,
        overwrite: true,
        tune: false,
        tune_args: {},
        fps: 30,
      })) as { id: string; out_dir: string };
    },
    { raw: `${root}/raw`, params: PARAMS },
  );
  expect(job.out_dir).toBe(`${root}/processed/e2e`);
  await page.evaluate((id) => {
    window.location.hash = `#/job/${id}`;
  }, job.id);

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

async function expectEveryPage(page: Page): Promise<void> {
  await expect(page.getByRole("heading", { name: "e2e", exact: true })).toBeVisible();
  const stage = page.getByRole("group", { name: "Stage" });
  for (const [p, marker] of PAGES) {
    const button = stage.getByRole("button", { name: p });
    await button.click();
    await expect(button).toHaveAttribute("aria-pressed", "true");
    await expect(marker(page).first()).toBeVisible({ timeout: 60_000 });
    await page.waitForLoadState("networkidle");
    const alerts = (await page.getByRole("alert").allTextContents()).filter((t) => !NOT_YET.includes(t));
    expect(alerts, `alerts on ${p}`).toEqual([]);
  }
  await expect(stage.getByRole("button", { name: "Tune" })).toHaveCount(0);
}
