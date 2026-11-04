import { defineConfig } from "@playwright/test";

// Streaming pass in headless Chromium on real files (no Python server): see e2e/stream.spec.ts.
// It builds its own copy of the browser build (own folder, port 8796) and serves that, so a run
// of hours is not hurt by edits to web/src or by another config's build, and it never reuses a
// server that is already there.
export default defineConfig({
  testDir: "e2e",
  testMatch: "stream.spec.ts",
  timeout: 3 * 3600_000,
  // the installed Google Chrome by default (STREAM_CHANNEL="" for Playwright's Chromium)
  use: { baseURL: "http://127.0.0.1:8796", channel: process.env.STREAM_CHANNEL ?? "chrome" },
  webServer: {
    command:
      "npx vite build --mode browser --outDir dist-stream && npx vite preview --mode browser --outDir dist-stream --port 8796 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:8796/",
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
