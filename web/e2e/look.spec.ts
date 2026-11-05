import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Locator, type Page, test } from "@playwright/test";

// Same look: main's `minime serve` and the browser build show one run folder (the e2e Python run
// `processed/t`, copied with its raw videos), and every page is screenshotted in both. main's
// screenshot is written as the reference of the browser build's (a snapshot made in the same run,
// nothing is committed), and the browser build's must match it within `maxDiffPixelRatio`. Run it
// with `npx playwright test -c playwright.look.config.ts`; the diff images land in test-results/.
const DATA = resolve(fileURLToPath(new URL(".", import.meta.url)), ".data");
// outside e2e/.data: main's smoke test lists that folder's runs, and this copy must not be one of them
const LOOK = join(tmpdir(), "minime-look");
// the ports playwright.look.config.ts serves on
const PY = `http://127.0.0.1:${process.env.LOOK_PY_PORT ?? 8799}`;
const BR = `http://127.0.0.1:${process.env.LOOK_BR_PORT ?? 8797}`;
const ROOT = "look"; // the folder's root name in the browser build

const PAGES: [string, (page: Page) => Locator][] = [
  ["Overview", (p) => p.getByText("From seeds to units")],
  ["Frames", (p) => p.getByRole("region", { name: "Reconstruction" })],
  ["Motion", (p) => p.locator(".uplot").first()],
  ["Compression", (p) => p.getByText("Total rank")],
  ["Seeds", (p) => p.getByText("Seeds per step")],
  ["Units", (p) => p.locator("table.data tbody tr").first()],
  ["Traces", (p) => p.getByText("Add unit", { exact: true })],
];

const SHOT = { fullPage: true, animations: "disabled", caret: "hide" } as const;

/** Where main shows an absolute disk path and the browser build a virtual path: both get the same
 * short text before a screenshot (a path's length decides how the line around it wraps). */
const PATH_TEXTS = [
  "h1.mono", // Home: the folder runs are listed from
  "a.card .mono", // Home: a run's path
  'section[aria-label="1. Raw videos"] .row .mono', // New run: the raw folder
  "span.muted.mono", // New run: the output folder
];

/** What else differs by design: shown by main, left out (or answered differently) by the browser
 * build. Painting over it would not do, as a missing picture's note or a missing button changes
 * the size of the page around it, so these are taken out of both builds before a screenshot. */
const BY_DESIGN: { selector: string; text?: string }[] = [
  { selector: '[aria-label="Known folders"]' }, // Home: recent folders (the browser build adds Browser storage)
  { selector: 'section[aria-label="Your data stays here"]' }, // Home: the browser build's privacy banner
  { selector: '[role="group"][aria-label="Stage"] button', text: "Tune" }, // no Tune in the browser build
  { selector: "label.switch", text: "Auto-tune" }, // New run: the same
  { selector: 'section[aria-label="Reconstruction"]' }, // a Python run's factors are not in the browser's format
  { selector: 'section[aria-label="Noise per pixel"]' }, // (the noise image: the same)
  { selector: 'section[aria-label="Speed"]' }, // New run: the browser build's GPU / CPU card
];

/** Files under `dir` as `[relative path, base64]`. */
function readTree(dir: string): [string, string][] {
  const out: [string, string][] = [];
  const walk = (d: string) => {
    for (const name of readdirSync(d)) {
      const p = join(d, name);
      if (statSync(p).isDirectory()) walk(p);
      else out.push([relative(dir, p).split("\\").join("/"), readFileSync(p).toString("base64")]);
    }
  };
  walk(dir);
  return out;
}

/** The look folder main serves: the e2e Python run and its raw videos, side by side, the run's
 * `raw_dir` pointing at the copy (and, as a virtual path, at the browser build's copy). */
function prepare(): { entries: [string, string][] } {
  rmSync(LOOK, { recursive: true, force: true });
  mkdirSync(join(LOOK, "processed"), { recursive: true });
  cpSync(join(DATA, "raw"), join(LOOK, "raw"), { recursive: true });
  cpSync(join(DATA, "processed", "t"), join(LOOK, "processed", "t"), { recursive: true });
  if (existsSync(join(DATA, "meta.json"))) cpSync(join(DATA, "meta.json"), join(LOOK, "meta.json"));
  const paramsFile = join(LOOK, "processed", "t", "params.json");
  const params = JSON.parse(readFileSync(paramsFile, "utf8"));
  const entries = readTree(LOOK).map(([path, b64]): [string, string] =>
    path === "processed/t/params.json"
      ? [path, Buffer.from(JSON.stringify({ ...params, raw_dir: `${ROOT}/raw` })).toString("base64")]
      : [path, b64],
  );
  writeFileSync(paramsFile, JSON.stringify({ ...params, raw_dir: join(LOOK, "raw") }, null, 1));
  // main's shortcut list (`/folders`): this folder, with the videos New run starts from
  const cache = join(DATA, "look-cache");
  mkdirSync(cache, { recursive: true });
  writeFileSync(
    join(cache, "folders.json"),
    JSON.stringify([{ path: LOOK, kind: "run", time: Date.now() / 1e3, raw: join(LOOK, "raw") }]),
  );
  return { entries };
}
test.afterAll(() => rmSync(LOOK, { recursive: true, force: true }));

/** The browser build with the look folder added to it (as a picked folder, raw videos remembered). */
async function openBrowserBuild(page: Page, entries: [string, string][]): Promise<void> {
  await page.goto(`${BR}/?e2e=1#/`);
  await page.waitForFunction(() => "__minimeAddTree" in window);
  await page.evaluate(
    ({ name, entries }) => {
      const files = entries.map(
        ([path, b64]) =>
          [path, new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], path.split("/").pop()!)] as [
            string,
            File,
          ],
      );
      return window.__minimeAddTree!(name, files, "raw");
    },
    { name: ROOT, entries },
  );
  await page.goto(`${BR}/?e2e=1&root=${ROOT}#/`); // reload: Home lists the folder just added
}

/** Wait until the page has drawn what it will draw: requests done, no computing note, fonts in. */
async function settle(page: Page): Promise<void> {
  await page.waitForLoadState("networkidle");
  await expect(page.getByText(/Computing|Loading/)).toHaveCount(0, { timeout: 60_000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(300);
}

test("every page looks like main's", async ({ browser }, testInfo) => {
  const { entries } = prepare();
  const errors: string[] = [];
  const open = async (): Promise<Page> => {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    return page;
  };
  const [py, br] = [await open(), await open()];
  await py.goto(`${PY}/?token=e2e&root=${encodeURIComponent(LOOK)}`);
  await openBrowserBuild(br, entries);

  /** Screenshot `name` in both builds; the browser build's must match main's. */
  const compare = async (name: string): Promise<void> => {
    await Promise.all([settle(py), settle(br)]);
    for (const p of [py, br])
      await p.evaluate(
        ({ paths, hide }) => {
          for (const el of document.querySelectorAll(paths.join(","))) el.textContent = "path";
          for (const { selector, text } of hide)
            for (const el of document.querySelectorAll<HTMLElement>(selector))
              if (!text || el.textContent?.includes(text)) el.style.display = "none";
        },
        { paths: PATH_TEXTS, hide: BY_DESIGN },
      );
    const reference = testInfo.snapshotPath(`${name}.png`);
    mkdirSync(dirname(reference), { recursive: true });
    writeFileSync(reference, await py.screenshot(SHOT));
    expect.soft(await br.screenshot(SHOT), name).toMatchSnapshot(`${name}.png`, { maxDiffPixelRatio: 0.02 });
  };

  // Home, then the run's pages
  for (const p of [py, br]) await expect(p.getByRole("link", { name: "Open run t" })).toBeVisible({ timeout: 60_000 });
  await compare("Home");
  for (const p of [py, br]) await p.getByRole("link", { name: "Open run t" }).click();
  for (const p of [py, br]) await expect(p.getByRole("heading", { name: "t", exact: true })).toBeVisible();
  for (const [name, marker] of PAGES) {
    for (const p of [py, br]) {
      const button = p.getByRole("group", { name: "Stage" }).getByRole("button", { name });
      await button.click();
      await expect(button).toHaveAttribute("aria-pressed", "true");
      await expect(marker(p).first()).toBeVisible({ timeout: 60_000 });
    }
    await compare(name);
  }

  // New run, with the e2e raw videos chosen (the page starts from the last run's)
  for (const p of [py, br]) {
    await p.evaluate(() => {
      window.location.hash = "#/new";
    });
    await expect(p.getByText("Read in this order")).toBeVisible({ timeout: 60_000 });
  }
  await compare("New run");
  expect(errors).toEqual([]);
});
