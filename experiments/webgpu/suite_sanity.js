// Is the exactness check able to fail? Does the shift need WGSL fma, or would a*b+c do?
import * as K from "./kernels.js";
import { Prep, initDevice } from "./gpuprep.js";

export async function run({ log, get, compare, fmt }) {
  const { device } = await initDevice();
  const meta = JSON.parse(new TextDecoder().decode(await get("full_meta.json")));
  const { h, w } = meta;
  const n = 64;
  const hw = h * w;
  const frames = (await get("full_frames.u8")).subarray(0, n * hw);
  const shifts = new Float32Array((await get("full_shifts.f32")).buffer).subarray(0, 2 * n);
  const want = (await get("full_shift_raw.u8")).subarray(0, n * hw);
  const median = (await get("full_median.u8")).subarray(0, n * hw);
  log(`self-test: median reference vs raw frames -> ${fmt(compare(median, frames))}`);

  const variants = {
    "fma()": K.shift,
    "a*b+c (unfused in source)": K.shift.replaceAll(/fma\(([^,]+), ([^,]+), ([^)]+)\)/g, "(($1) * ($2) + ($3))"),
    "mul and add forced apart (via select)": K.shift.replaceAll(
      /fma\(([^,]+), ([^,]+), ([^)]+)\)/g,
      "(select(0.0, ($1) * ($2), ax >= 0.0) + ($3))",
    ),
  };
  for (const [name, code] of Object.entries(variants)) {
    const prep = new Prep(device, { h, w, maxB: n });
    const mod = device.createShaderModule({ code });
    prep.pipes = {};
    await prep.compile();
    prep.pipes.shift = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: mod, entryPoint: "main" },
    });
    const out = await prep.runBatch(frames, shifts, n, ["shift"]);
    log(`shift ${name}: ${fmt(compare(out, want))}`);
    prep.destroy();
  }
}
