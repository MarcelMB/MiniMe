import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";

// Auto-tune and reruns in the browser build on the e2e raw folder (4500 frames: two windows).
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

/** Open the browser build with the e2e hooks and copy `n` AVIs (100 frames each) into storage. */
async function openApp(page: Page, n = 45): Promise<string> {
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

const api = (page: Page, method: "getJSON" | "postJSON", path: string, body?: unknown) =>
  // biome-ignore lint/suspicious/noExplicitAny: test hook
  page.evaluate(({ method, path, body }) => (window as any).__minimeApi[method](path, body), { method, path, body });

test("New run + auto-tune → Tune page → manual rerun → rerun switcher", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  const root = await openApp(page);
  const job = (await api(page, "postJSON", "/jobs", {
    raw_dir: `${root}/raw`,
    name: "tuned",
    params: PARAMS,
    keep_store: false,
    overwrite: true,
    tune: true,
    tune_args: { diameter: 7 },
    fps: 30,
  })) as { id: string };
  await page.evaluate((id) => {
    window.location.hash = `#/job/${id}`;
  }, job.id);
  await expect(page.getByText("Run + auto-tune")).toBeVisible();
  const openRun = page.getByRole("button", { name: "Open run" });
  const failed = page.getByRole("alert");
  await expect(openRun.or(failed)).toBeVisible({ timeout: 540_000 });
  await expect(failed).toHaveCount(0);
  const stages = page.getByRole("region", { name: "Stages" });
  await expect(stages).toContainText("Streaming pass");
  await expect(stages).toContainText("auto-tune · the run's values");
  await expect(stages).toContainText("auto-tune · final rerun");

  await openRun.click();
  // the job opens the Tune page
  const proposal = page.getByRole("region", { name: "Proposal" });
  await expect(proposal).toBeVisible();
  await expect(proposal).toContainText("good units");
  await expect(page.getByRole("region", { name: "Sweep" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Thresholds" })).toContainText("pairs within D", {
    timeout: 60_000,
  });
  await expect(page.getByRole("region", { name: "Flagged units" })).toBeVisible();
  const reruns = page.getByRole("region", { name: "Reruns" });
  await expect(reruns).toContainText("auto");
  await expect(reruns.locator("tbody tr")).toHaveCount(2); // the run and auto

  // a manual rerun from the window-2 form
  await page.getByLabel("sparse_snr").fill("0.7");
  await page.getByLabel("Rerun name").fill("r1");
  await page.getByRole("button", { name: "Rerun window 2" }).click();
  await expect(page.getByText("Window-2 rerun · done").or(failed)).toBeVisible({ timeout: 300_000 });
  await expect(failed).toHaveCount(0);
  await page.getByRole("button", { name: "Open run" }).click();
  await expect(reruns.locator("tbody tr")).toHaveCount(3);
  await expect(reruns).toContainText("sparse_snr 1→0.7");

  // the rerun switcher drives the other pages
  await page.getByRole("group", { name: "Stage" }).getByRole("button", { name: "Units" }).click();
  const runId = page.url().split("#/run/")[1].split("/")[0];
  await page.getByRole("group", { name: "Rerun shown" }).getByRole("button", { name: "r1" }).click();
  await expect(page.locator("table.data tbody tr").first()).toBeVisible();
  const units = (await api(page, "getJSON", `/runs/${runId}/units?rerun=r1`)) as { units: unknown[] };
  expect(units.units.length).toBeGreaterThan(0);

  expect(errors).toEqual([]);
});
