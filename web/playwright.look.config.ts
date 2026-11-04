import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

// Same look: main's `minime serve` (port 8799) and the browser build (port 8797) show the same run
// folder, and e2e/look.spec.ts compares their pages screenshot by screenshot. Both servers are
// started here and never reused. Needs the e2e run `playwright.config.ts` makes; set
// `MINIME_PYTHON` if the venv is not `../.venv`.
const here = fileURLToPath(new URL(".", import.meta.url));
const py = process.env.MINIME_PYTHON ?? resolve(here, "../.venv/bin/python");
const data = resolve(here, "e2e/.data");

export default defineConfig({
  testDir: "e2e",
  testMatch: /look\.spec\.ts/,
  timeout: 300_000,
  // main's screenshot of each page is written here by the spec, and the browser build's is compared with it
  snapshotPathTemplate: "{testDir}/.data/look-shots/{arg}{ext}",
  use: { browserName: "chromium", viewport: { width: 1280, height: 900 } },
  webServer: [
    {
      command: `"${py}" e2e/make_run.py "${data}" && MINIME_CACHE="${data}/look-cache" "${py}" -m minime serve "${data}" --no-browser --port 8799 --token e2e`,
      url: "http://127.0.0.1:8799/",
      timeout: 300_000,
      reuseExistingServer: false,
    },
    {
      command:
        "npx vite build --mode browser && npx vite preview --mode browser --outDir dist-browser --port 8797 --strictPort",
      url: "http://127.0.0.1:8797/",
      timeout: 300_000,
      reuseExistingServer: false,
    },
  ],
});
