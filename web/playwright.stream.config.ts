import { defineConfig } from "@playwright/test";

// Streaming pass in headless Chromium on real files (no Python server): see e2e/stream.spec.ts.
// It builds its own copy of the browser build (own folder, port 8796 or STREAM_PORT) and serves that,
// so a run of hours is not hurt by edits to web/src or by another config's build, and it never reuses
// a server that is already there. Two runs at once need their own STREAM_PORT and STREAM_DIST.
const port = Number(process.env.STREAM_PORT ?? 8796);
const dist = process.env.STREAM_DIST ?? "dist-stream";
export default defineConfig({
  testDir: "e2e",
  testMatch: "stream.spec.ts",
  timeout: 3 * 3600_000,
  // the installed Google Chrome by default (STREAM_CHANNEL="" for Playwright's Chromium)
  use: { baseURL: `http://127.0.0.1:${port}`, channel: process.env.STREAM_CHANNEL ?? "chrome" },
  webServer: {
    command: `npx vite build --mode browser --outDir ${dist} && npx vite preview --mode browser --outDir ${dist} --port ${port} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${port}/`,
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
