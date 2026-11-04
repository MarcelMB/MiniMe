// Pass 2 on single tiles in Chrome, old code against new (see `src/wasm_bench.rs`):
//   node run.mjs --pkg PKG_DIR --tiles TILE_DIR --inputs INPUTS_DIR --port 8812 \
//        [--n 100] [--ids 0,1,2] [--reps 1] [--only ref|new] [--headed]
// PKG_DIR is a `wasm-pack build --target web -- --features profile,reference` output.
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => {
  const i = process.argv.indexOf(n);
  return i >= 0 ? process.argv[i + 1] : d;
};
const pkg = resolve(arg("--pkg"));
const tiles = resolve(arg("--tiles"));
const inputs = resolve(arg("--inputs"));
const port = Number(arg("--port", "8812"));
const reps = Number(arg("--reps", "1"));
const only = arg("--only");
const n = Number(arg("--n", "100"));
const ids = arg("--ids") ? arg("--ids").split(",").map(Number) : [...Array(n).keys()];

const require = createRequire(join(here, "../../../../web/package.json"));
const { chromium } = require("@playwright/test");

const types = { ".js": "text/javascript", ".wasm": "application/wasm", ".html": "text/html", ".json": "application/json" };
const roots = { "/pkg/": pkg, "/tiles/": tiles, "/inputs/": inputs };
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  let file = url === "/" ? join(here, "bench.html") : null;
  for (const [pre, dir] of Object.entries(roots)) if (url.startsWith(pre)) file = join(dir, url.slice(pre.length));
  if (!file || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(port, "127.0.0.1", r));

const browser = await chromium.launch({ channel: "chrome", headless: !process.argv.includes("--headed") });
const page = await (await browser.newContext()).newPage();
page.on("pageerror", (e) => console.error("pageerror", e.message));
await page.goto(`http://127.0.0.1:${port}/`);
await page.waitForFunction(() => window.benchReady);
console.log(`chrome ${browser.version()}`);
const grid = null;
const r = await page.evaluate((cfg) => window.runBench(cfg), { ids, reps, only, grid });
await browser.close();
server.close();

const f = (v) => v.toFixed(1).padStart(10);
console.log(`tiles ${ids.length} reps ${reps} (ms per pass over the tiles)`);
console.log("step".padEnd(16) + "ref".padStart(10) + "new".padStart(10) + "x".padStart(8));
const names = [...new Set([...Object.keys(r.sums.ref.steps), ...Object.keys(r.sums.new.steps)])].sort();
for (const s of names) {
  const a = (r.sums.ref.steps[s] ?? 0) / reps;
  const b = (r.sums.new.steps[s] ?? 0) / reps;
  console.log(s.padEnd(16) + f(a) + f(b) + (b > 0 && a > 0 ? (a / b).toFixed(2) : "-").padStart(8));
}
const a = r.sums.ref.ms / reps;
const b = r.sums.new.ms / reps;
console.log("TOTAL".padEnd(16) + f(a) + f(b) + (a > 0 && b > 0 ? (a / b).toFixed(2) : "-").padStart(8));
if (!only) console.log(JSON.stringify(r.worst));
if (process.argv.includes("--per-tile")) console.log(JSON.stringify(r.perTile));
