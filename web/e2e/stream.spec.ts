import { expect, test } from "@playwright/test";
import { mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Runs one streaming pass in the browser and saves its outputs for
// experiments/streaming/check_rust.py:
//   STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm run e2e:stream
const raw = process.env.STREAM_RAW;
const inputs = process.env.STREAM_INPUTS;
const out = process.env.STREAM_OUT;

test("streams raw AVIs in the browser and saves the outputs", async ({ page }) => {
  test.skip(!raw || !inputs || !out, "set STREAM_RAW, STREAM_INPUTS and STREAM_OUT");
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("worker", (w) => console.log(`worker started: ${w.url()}`));
  page.on("console", (m) => m.text().startsWith("[worker]") && console.log(m.text()));
  page.on("console", (m) => m.text().startsWith("[stream]") && console.log(m.text()));
  await page.goto("/#/stream");
  const avis = readdirSync(raw!).filter((f) => /^\d+.*\.avi$/i.test(f)).map((f) => join(raw!, f));
  const nFiles = Number(process.env.STREAM_FILES ?? avis.length);
  await page.getByLabel("Raw AVIs").setInputFiles(avis.slice(0, nFiles));
  await page.getByLabel("Inputs folder").setInputFiles(inputs!);
  await page.getByRole("button", { name: "Run" }).click();
  const progress = setInterval(async () => {
    const t = await page.getByRole("region", { name: "Progress" }).textContent().catch(() => null);
    if (t) console.log(t);
  }, 30_000);
  const results = page.getByRole("region", { name: "Results" });
  const failed = page.getByRole("alert");
  await expect(results.or(failed)).toBeVisible({ timeout: 3 * 3600_000 });
  clearInterval(progress);
  if (await failed.count()) console.log(await failed.textContent());
  await expect(failed).toHaveCount(0);
  mkdirSync(out!, { recursive: true });
  // read each output's blob in the page (clicking many download links trips Chrome's
  // multiple-downloads prompt)
  for (const link of await results.getByRole("link").all()) {
    const [name, url] = [await link.getAttribute("download"), await link.getAttribute("href")];
    const b64 = await page.evaluate(async (u) => {
      const blob = await (await fetch(u!)).blob();
      return await new Promise<string>((res) => {
        const r = new FileReader();
        r.onload = () => res(String(r.result).split(",")[1] ?? "");
        r.readAsDataURL(blob);
      });
    }, url);
    writeFileSync(join(out!, name!), Buffer.from(b64, "base64"));
  }
  console.log(await results.locator("p").first().textContent());
  expect(errors).toEqual([]);
});
