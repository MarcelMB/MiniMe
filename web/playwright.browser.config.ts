import { defineConfig } from "@playwright/test";

// The browser build (no Python server): New run → job → viewer, see e2e/browser.spec.ts. It
// builds and serves its own copy on port 8797 (or BROWSER_PORT) and never reuses a server that is
// already there.
const port = Number(process.env.BROWSER_PORT ?? 8797);
export default defineConfig({
  testDir: "e2e",
  testMatch: /browser\.spec\.ts/,
  timeout: 600_000,
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: "chromium" },
  webServer: {
    command: `npx vite build --mode browser && npx vite preview --mode browser --outDir dist-browser --port ${port} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${port}/`,
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
