import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

const src = resolve(import.meta.dirname, "src");
/** The browser build: these modules are replaced wherever they are imported. */
const SWAP: Record<string, string> = {
  [resolve(src, "transport.ts")]: resolve(src, "browser/transport.ts"),
  [resolve(src, "features.ts")]: resolve(src, "browser/features.ts"),
};

function swap(map: Record<string, string>): Plugin {
  return {
    name: "browser-swap",
    enforce: "pre",
    async resolveId(source, importer, opts) {
      const r = await this.resolve(source, importer, { ...opts, skipSelf: true });
      return r && map[r.id] ? map[r.id] : r;
    },
  };
}

export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === "browser" ? [swap(SWAP)] : [])],
  base: "./",
  build:
    mode === "browser"
      ? { outDir: "dist-browser", emptyOutDir: true }
      : { outDir: "../src/minime/serve/static", emptyOutDir: true },
  server: { proxy: { "/api": process.env.MINIME_API ?? "http://127.0.0.1:8765" } },
  test: { environment: "jsdom", include: ["src/**/*.test.ts"], exclude: ["src/browser/**", "node_modules/**"] },
}));
