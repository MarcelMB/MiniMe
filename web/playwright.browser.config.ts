import { defineConfig } from "@playwright/test";

// The browser build (no Python server): New run → job → viewer, see e2e/browser.spec.ts. It
// builds and serves its own copy on port 8797 and never reuses a server that is already there.
export default defineConfig({
  testDir: "e2e",
  testMatch: /(browser|tune)\.spec\.ts/,
  timeout: 600_000,
  use: { baseURL: "http://127.0.0.1:8797", browserName: "chromium" },
  webServer: {
    command:
      "npx vite build --mode browser && npx vite preview --mode browser --outDir dist-browser --port 8797 --strictPort",
    url: "http://127.0.0.1:8797/",
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
