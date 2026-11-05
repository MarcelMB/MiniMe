// node run.mjs <pkg dir> <frames.u8> <port> [kernels...]: old vs new kernels in headless Chrome
// (BROWSER=firefox or webkit for Playwright's builds of those: the outputs must be identical there too).
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox, webkit } from "../../../../web/node_modules/@playwright/test/index.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const [pkg, framesFile, port, ...rest] = process.argv.slice(2);
const kernels = rest.length ? rest : ["median", "tophat", "gauss_u8", "shift", "shift_f32", "gauss_f32", "resize_up2"];
const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };
const server = createServer((req, res) => {
  const name = (req.url ?? "/").split("?")[0];
  const file =
    name === "/" ? join(here, "bench.html") : name === "/frames.u8" ? framesFile : join(pkg, name.replace(/^\/pkg\//, ""));
  let body;
  try {
    body = readFileSync(file);
  } catch {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "content-type": types[file.slice(file.lastIndexOf("."))] ?? "application/octet-stream" });
  res.end(body);
}).listen(Number(port), "127.0.0.1");

const frames = new Uint8Array(readFileSync(framesFile));
const kind = process.env.BROWSER ?? "chrome";
const browser =
  kind === "firefox" ? await firefox.launch() : kind === "webkit" ? await webkit.launch() : await chromium.launch({ channel: "chrome" });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`);
await page.waitForFunction(() => window.ready);
console.log(`${kind} ${browser.version()}, ${frames.length / 360000} frames`);
const res = await page.evaluate((k) => window.run(k, 7, 30, 600, 600), kernels);
for (const r of res)
  console.log(`${r.kernel.padEnd(14)} old ${r.old.toFixed(3).padStart(8)} ms  new ${r.new.toFixed(3).padStart(8)} ms  ${(r.old / r.new).toFixed(2)}x  identical on ${r.identical_on_frames} frames  hash ${r.hash}`);
await browser.close();
server.close();
