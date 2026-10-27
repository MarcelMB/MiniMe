/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../src/caanfast/serve/static", emptyOutDir: true },
  server: { proxy: { "/api": process.env.CAANFAST_API ?? "http://127.0.0.1:8765" } },
  test: { environment: "jsdom", include: ["src/**/*.test.ts"] },
});
