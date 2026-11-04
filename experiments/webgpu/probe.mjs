// node probe.mjs: does each browser / launch mode expose WebGPU on the page and in a dedicated Worker?
import { chromium, firefox, webkit } from "../../web/node_modules/@playwright/test/index.mjs";
import { serve } from "./serve.mjs";

const port = Number(process.env.GPU_PORT ?? 8814);
const server = await serve(port);
const variants = [
  ["chromium headless-shell, no flags", () => chromium.launch()],
  ["chromium headless-shell, --enable-unsafe-webgpu", () => chromium.launch({ args: ["--enable-unsafe-webgpu"] })],
  [
    "chromium headless-shell, --enable-unsafe-webgpu --enable-features=Vulkan",
    () => chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] }),
  ],
  ["chromium new-headless (channel chromium), no flags", () => chromium.launch({ channel: "chromium" })],
  [
    "chromium new-headless, --enable-unsafe-webgpu",
    () => chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] }),
  ],
  ["chromium headed, no flags", () => chromium.launch({ headless: false })],
  [
    "chromium headed, --enable-unsafe-webgpu",
    () => chromium.launch({ headless: false, args: ["--enable-unsafe-webgpu"] }),
  ],
  ["google chrome headless, no flags", () => chromium.launch({ channel: "chrome" })],
  [
    "google chrome headless, --enable-unsafe-webgpu",
    () => chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu"] }),
  ],
  ["google chrome headed, no flags", () => chromium.launch({ channel: "chrome", headless: false })],
  ["firefox", () => firefox.launch()],
  ["webkit", () => webkit.launch()],
];
const only = process.argv[2];
for (const [name, launch] of variants) {
  if (only && !name.includes(only)) continue;
  let b;
  try {
    b = await launch();
    const page = await b.newPage();
    await page.goto(`http://127.0.0.1:${port}/probe.html`);
    await page.waitForFunction(() => window.__result, null, { timeout: 30000 });
    const r = await page.evaluate(() => window.__result);
    const m = r.main;
    const w = r.worker;
    console.log(`\n== ${name} (${b.version()})`);
    console.log(
      `  main  : gpu=${m.hasGpu} adapter=${m.adapter} ${m.info ? `${m.info.vendor}/${m.info.architecture}/${m.info.description}` : ""} roundtrip=${JSON.stringify(m.roundTrip)} deviceMs=${m.deviceMs?.toFixed(0)} ${m.error ?? ""}`,
    );
    console.log(
      `  worker: gpu=${w.hasGpu} adapter=${w.adapter} roundtrip=${JSON.stringify(w.roundTrip)} deviceMs=${w.deviceMs?.toFixed(0)} ${w.error ?? ""}`,
    );
    if (process.env.VERBOSE && m.limits) console.log("  ", JSON.stringify(m.limits), m.features.join(","));
  } catch (e) {
    console.log(`\n== ${name}: FAILED ${String(e).split("\n")[0]}`);
  } finally {
    await b?.close().catch(() => {});
  }
}
server.close();
