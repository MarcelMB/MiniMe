// node run.mjs <suite> [key=value ...]: run a suite in headless Chromium (new headless = real Metal GPU)
import { chromium, webkit } from "../../web/node_modules/@playwright/test/index.mjs";
import { serve } from "./serve.mjs";

const port = Number(process.env.GPU_PORT ?? 8814);
const suite = process.argv[2] ?? "check";
const extra = process.argv.slice(3).join("&");
const server = await serve(port);
const channel = process.env.GPU_CHANNEL ?? "chromium";
// GPU_CHANNEL: chromium (default, new headless = Metal), chrome, shell (old headless shell = SwiftShader), webkit
const browser =
  channel === "webkit"
    ? await webkit.launch()
    : await chromium.launch({
        ...(channel === "shell" ? {} : { channel }),
        headless: !process.env.GPU_HEADED,
        args: ["--enable-unsafe-webgpu", ...(process.env.GPU_ARGS ? process.env.GPU_ARGS.split(" ") : [])],
      });
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log(m.text()));
  page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  await page.goto(`http://127.0.0.1:${port}/bench.html?suite=${suite}&${extra}`);
  await page.waitForFunction(() => window.__done, null, {
    timeout: Number(process.env.GPU_TIMEOUT ?? 280000),
    polling: 500,
  });
  console.log("done", JSON.stringify(await page.evaluate(() => window.__done)));
} finally {
  await browser.close();
  server.close();
}
