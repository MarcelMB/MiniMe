import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../src/minime/serve/static", emptyOutDir: true },
  server: { proxy: { "/api": process.env.MINIME_API ?? "http://127.0.0.1:8765" } },
  test: { environment: "jsdom", include: ["src/**/*.test.ts"] },
});
