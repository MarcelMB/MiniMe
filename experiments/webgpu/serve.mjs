// Tiny static server for the WebGPU harness: this folder at /, GPU_DATA (frames, references) at /data/.
// node serve.mjs [port]   (default 8814)
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = resolve(process.env.GPU_DATA ?? join(here, "../../web/e2e/.data/gpu"));
const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wgsl": "text/plain",
  ".wasm": "application/wasm",
  ".json": "application/json",
};

export function serve(port = 8814) {
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://x");
    const rel = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
    const file = rel.startsWith("/data/") ? join(dataDir, rel.slice(6)) : join(here, rel === "/" ? "index.html" : rel);
    if (!existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, {
      "content-type": MIME[extname(file)] ?? "application/octet-stream",
      "content-length": statSync(file).size,
      "cache-control": "no-store",
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    });
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(port, "127.0.0.1", () => ok(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.argv[2] ?? 8814);
  await serve(port);
  console.log(`http://127.0.0.1:${port}/  data: ${dataDir}`);
}
