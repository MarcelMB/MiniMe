import { defineConfig } from "@playwright/test";

// Runs e2e/tune.spec.ts alone, on port 8798 (the browser config holds 8797).
// It builds and serves its own copy and never reuses a server that is already there.
export default defineConfig({
  testDir: "e2e",
  testMatch: /tune\.spec\.ts/,
  timeout: 600_000,
  use: { baseURL: "http://127.0.0.1:8798", browserName: "chromium" },
  webServer: {
    command:
      "VITE_TUNE=1 npx vite build --mode browser && npx vite preview --mode browser --outDir dist-browser --port 8798 --host 127.0.0.1 --strictPort",
    url: "http://127.0.0.1:8798/",
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
