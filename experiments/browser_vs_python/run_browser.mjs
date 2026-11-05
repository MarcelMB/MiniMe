// Runs one session through the built browser app in headless Chromium, the way a user of a
// browser without the folder dialog does: New run → Choose folder (folder upload of the raw
// AVIs, read from disk as they are) → Run name, frame rate → Start run (default parameters) →
// job page → Open run → "Download run (.zip)", unzipped into DEST.
//
//   node run_browser.mjs --web <worktree>/web --base http://127.0.0.1:8796 --raw RAW \
//     --name browser_20261103 --fps 20 --dest <session>/processed/browser_20261103 \
//     --profile <empty dir> --log job.json
//
// The app must already be served (`npx vite preview --mode browser --outDir dist-browser`).
// The persistent profile keeps browser storage on disk (an incognito context keeps it in
// memory); the run is written there, then downloaded.
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values: a } = parseArgs({
  options: {
    web: { type: "string" },
    base: { type: "string" },
    raw: { type: "string" },
    name: { type: "string" },
    fps: { type: "string" },
    dest: { type: "string" },
    profile: { type: "string" },
    log: { type: "string" },
    tune: { type: "boolean" },
  },
});
const { chromium } = createRequire(join(a.web, "package.json"))("@playwright/test");
const HOURS = 3_600_000;
const stamp = () => new Date().toISOString().slice(11, 19);

const ctx = await chromium.launchPersistentContext(a.profile, { headless: true, acceptDownloads: true });
// no File System Access: the raw folder is uploaded read-only and the run goes to browser storage
await ctx.addInitScript(() => {
  delete Window.prototype.showDirectoryPicker;
  delete window.showDirectoryPicker;
});
const page = ctx.pages()[0] ?? (await ctx.newPage());
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

await page.goto(`${a.base}/?e2e=1#/new`);
await page.getByRole("button", { name: "Choose folder" }).click();
await page.locator("input[type=file]").setInputFiles(a.raw);
await page.getByText("Read in this order").waitFor({ timeout: 600_000 });
await page.getByLabel("Frame rate of the recording (Hz)").fill(a.fps);
await page.getByLabel("Run name").fill(a.name);
if (a.tune) await page.getByLabel(/^Auto-tune the seed, init and CNMF parameters/).check();
const t0 = Date.now();
await page.getByRole("button", { name: "Start run" }).click();
await page.waitForURL(/#\/job\//, { timeout: 120_000 });
const jobId = page.url().split("#/job/")[1];
console.log(`${stamp()} job ${jobId} started`);

const openRun = page.getByRole("button", { name: "Open run" });
const failed = page.getByRole("alert");
const progress = setInterval(async () => {
  const t = await page.getByRole("region", { name: "Stages" }).innerText().catch(() => "");
  const last = t.split("\n").filter(Boolean).slice(-2).join(" | ");
  console.log(`${stamp()} ${Math.round((Date.now() - t0) / 1000)} s  ${last}`);
}, 60_000);
await openRun.or(failed).first().waitFor({ timeout: 4 * HOURS });
clearInterval(progress);
const wall = (Date.now() - t0) / 1000;
const job = await page.evaluate((id) => window.__minimeApi.getJSON(`/jobs/${id}`), jobId);
writeFileSync(a.log, JSON.stringify({ job, wall_click_to_done_s: wall, errors }, null, 1));
if (job.status !== "done") {
  console.error(`run ${job.status}: ${job.error ?? ""}`);
  await ctx.close();
  process.exit(1);
}
console.log(`${stamp()} done in ${wall.toFixed(0)} s`);

await openRun.click();
const button = page.getByRole("button", { name: "Download run (.zip)" });
await button.waitFor({ timeout: 120_000 });
const [download] = await Promise.all([page.waitForEvent("download", { timeout: HOURS }), button.click()]);
const zip = join(a.profile, "..", `${a.name}.zip`);
await download.saveAs(zip);
await ctx.close();
mkdirSync(a.dest, { recursive: true });
execFileSync("unzip", ["-q", zip, "-d", a.dest]);
rmSync(zip);
console.log(`${stamp()} unzipped into ${a.dest}; console errors: ${errors.length}`);
