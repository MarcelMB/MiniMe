import { readdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

// The New run page's GPU self-test and speed test on raw AVIs (the e2e hook `__minimeSpeed`), for the
// stream config's browser: STREAM_RAW=<raw dir> [STREAM_FILES=n] [STREAM_WORKERS=n] [STREAM_GPU_TEST=..]
// npx playwright test -c <config with this spec> speed. Prints what it measured; passes when the
// self-test passed or a reason says why the GPU is not used.
const raw = process.env.STREAM_RAW;

test("measures the GPU and the CPU on raw AVIs", async ({ page }) => {
  test.skip(!raw, "set STREAM_RAW");
  page.on("console", (m) => m.text().startsWith("[worker]") && console.log(m.text()));
  await page.goto("/?e2e=1#/runs");
  await page.waitForFunction(() => "__minimeSpeed" in window);
  const avis = readdirSync(raw!)
    .filter((f) => /^\d+.*\.avi$/i.test(f))
    .sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10))
    .map((f) => join(raw!, f))
    .slice(0, Number(process.env.STREAM_FILES ?? 1));
  await page.evaluate(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.id = "avis";
    input.multiple = true;
    document.body.append(input);
  });
  await page.locator("#avis").setInputFiles(avis);
  const workers = Number(process.env.STREAM_WORKERS ?? 0) || undefined;
  const answer = await page.evaluate(
    ({ workers }) => {
      const files = [...((document.getElementById("avis") as HTMLInputElement).files ?? [])];
      return window.__minimeSpeed!(files, workers ? { workers } : {}, true);
    },
    { workers },
  );
  console.log(
    `speed test: ${JSON.stringify({ ...answer, test: answer.test && { fps: answer.test.fps, bothGpu: answer.test.bothGpu, seconds: answer.test.seconds } })}`,
  );
  expect(answer.selfTest === "pass" || answer.reason.length > 0).toBe(true);
});
