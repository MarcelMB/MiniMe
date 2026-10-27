import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const py = process.env.MINIME_PYTHON ?? resolve(here, "../.venv/bin/python");
const data = resolve(here, "e2e/.data");

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  use: { baseURL: "http://127.0.0.1:8799" },
  webServer: {
    command: `"${py}" e2e/make_run.py "${data}" && MINIME_CACHE="${data}/cache" "${py}" -m minime serve "${data}" --no-browser --port 8799 --token e2e`,
    url: "http://127.0.0.1:8799/",
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
