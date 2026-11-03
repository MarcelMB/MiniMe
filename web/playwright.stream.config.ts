import { defineConfig } from "@playwright/test";

// Streaming pass in headless Chromium on real files (no Python server): see e2e/stream.spec.ts.
export default defineConfig({
  testDir: "e2e",
  testMatch: "stream.spec.ts",
  timeout: 3 * 3600_000,
  // the installed Google Chrome by default (STREAM_CHANNEL="" for Playwright's Chromium)
  use: { baseURL: "http://127.0.0.1:8798", channel: process.env.STREAM_CHANNEL ?? "chrome" },
  webServer: {
    command: "npx vite --port 8798 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:8798/",
    reuseExistingServer: true,
  },
});
