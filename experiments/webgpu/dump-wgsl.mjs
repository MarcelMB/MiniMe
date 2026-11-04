// Writes the WGSL of kernels.js to spike/src/kernels/*.wgsl, so the Rust (wgpu) crate runs the very same shaders.
import { writeFileSync } from "node:fs";
import * as K from "./kernels.js";

const out = {
  unpack: K.unpack,
  pack: K.pack,
  tophat_final: K.tophatFinal,
  median: K.median(7),
  shift: K.shift,
  erode_rows: K.morphRows(15, true),
  erode_cols: K.morphCols(15, true),
  dilate_rows: K.morphRows(15, false),
  dilate_cols: K.morphCols(15, false),
  gemm_t: K.gemm(false),
  gemm_n: K.gemm(true),
};
for (const [name, code] of Object.entries(out))
  writeFileSync(new URL(`spike/src/kernels/${name}.wgsl`, import.meta.url), `${code.trim()}\n`);
console.log("wrote", Object.keys(out).length, "shaders");
