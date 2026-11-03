import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/browser/**/*.test.ts"],
    setupFiles: ["src/browser/testing/setup.ts"],
    testTimeout: 60_000,
  },
});
