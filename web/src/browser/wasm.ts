import init, * as core from "../stream/pkg/minime_core.js";

let ready: Promise<typeof core> | null = null;

/** The wasm core in the page (frames, previews); the pipeline runs it in its own worker. */
export function wasm(): Promise<typeof core> {
  ready ??= load().then(() => core);
  return ready;
}

async function load(): Promise<void> {
  if (import.meta.env.MODE === "test") {
    // Node cannot fetch the module by URL: hand `init` the bytes
    const { readFileSync } = await import("node:fs");
    const bytes = readFileSync(new URL("../stream/pkg/minime_core_bg.wasm", import.meta.url));
    await init({ module_or_path: bytes });
  } else await init();
}
