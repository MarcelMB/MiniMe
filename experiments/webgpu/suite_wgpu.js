// The Rust `wgpu` crate compiled to wasm32 (WebGPU backend), run inside this worker: init, exactness,
// speed next to the same Rust CPU code (wasm simd128, one worker) and next to the TypeScript driver.
import init, { GpuPrep, cpu_chain, cpu_median, cpu_shift, cpu_tophat } from "./pkg_gpu/gpu_spike.js";
import { Prep, initDevice } from "./gpuprep.js";

const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

export async function run({ log, get, compare, fmt, q }) {
  const meta = JSON.parse(new TextDecoder().decode(await get("full_meta.json")));
  const { n: N, h, w } = meta;
  const hw = h * w;
  const frames = await get("full_frames.u8");
  const shifts = new Float32Array((await get("full_shifts.f32")).buffer);
  const want = await get("full_final.u8");
  const B = 64;

  let t0 = performance.now();
  const wasm = await init();
  log(
    `wasm instantiate ${(performance.now() - t0).toFixed(0)} ms (gpu_spike_bg.wasm, memory ${(wasm.memory.buffer.byteLength / 1e6).toFixed(0)} MB)`,
  );
  t0 = performance.now();
  const gpu = await GpuPrep.create(h, w, B);
  log(
    `GpuPrep.create (adapter + device + 8 pipelines, async, in worker) ${(performance.now() - t0).toFixed(0)} ms; adapter: ${gpu.info()}`,
  );

  // exactness against the CPU reference (the result stays in wasm memory)
  const got = new Uint8Array(N * hw);
  for (let f0 = 0; f0 < N; f0 += B) {
    const len = await gpu.run(frames.subarray(f0 * hw, (f0 + B) * hw), shifts.subarray(2 * f0, 2 * (f0 + B)), B);
    got.set(new Uint8Array(wasm.memory.buffer, gpu.out_ptr(), len), f0 * hw);
  }
  log(`wgpu-in-wasm chain vs CPU reference: ${fmt(compare(got, want))}`);

  // speed
  const reps = Number(q.get("reps") ?? 5);
  let ts = [];
  for (let k = 0; k < reps + 2; k++) {
    const f0 = (k % 4) * B;
    const t = performance.now();
    await gpu.run(frames.subarray(f0 * hw, (f0 + B) * hw), shifts.subarray(2 * f0, 2 * (f0 + B)), B);
    if (k >= 2) ts.push((performance.now() - t) / B);
  }
  log(
    `wgpu-in-wasm sequential B=${B}: ${median(ts).toFixed(3)} ms/frame (upload incl. JS->wasm copy of the frames, compute, readback into wasm memory)`,
  );
  // phases: submit (frames copied into wasm by the binding + write_buffer + encode) vs finish (wait + map + copy)
  const tu = [];
  const tf = [];
  for (let k = 0; k < reps + 2; k++) {
    const f0 = (k % 4) * B;
    const a = performance.now();
    gpu.submit(frames.subarray(f0 * hw, (f0 + B) * hw), shifts.subarray(2 * f0, 2 * (f0 + B)), B);
    const b = performance.now();
    await gpu.finish(B);
    const c = performance.now();
    if (k >= 2) {
      tu.push((b - a) / B);
      tf.push((c - b) / B);
    }
  }
  log(
    `  phases per frame: submit (JS->wasm copy + write_buffer + encode) ${median(tu).toFixed(3)} ms, finish (GPU wait + map + copy to wasm Vec + copy to out) ${median(tf).toFixed(3)} ms`,
  );

  // the same chain driven from TypeScript in this worker
  const { device } = await initDevice();
  const prep = new Prep(device, { h, w, maxB: B, slots: 1 });
  await prep.compile();
  const sink = new Uint8Array(B * hw);
  ts = [];
  for (let k = 0; k < reps + 2; k++) {
    const f0 = (k % 4) * B;
    const t = performance.now();
    await prep.runBatch(
      frames.subarray(f0 * hw, (f0 + B) * hw),
      shifts.subarray(2 * f0, 2 * (f0 + B)),
      B,
      ["median", "shift", "tophat"],
      { into: sink },
    );
    if (k >= 2) ts.push((performance.now() - t) / B);
  }
  log(`hand-written WGSL driven from TypeScript, sequential B=${B}: ${median(ts).toFixed(3)} ms/frame`);

  // CPU, one worker, wasm simd128, same frames
  const nCpu = 16;
  for (const [name, fn] of [
    ["median 7x7", (i) => cpu_median(frames.subarray(i * hw, (i + 1) * hw), h, w)],
    ["shift", (i) => cpu_shift(frames.subarray(i * hw, (i + 1) * hw), h, w, shifts[2 * i], shifts[2 * i + 1])],
    ["tophat 15", (i) => cpu_tophat(frames.subarray(i * hw, (i + 1) * hw), h, w)],
  ]) {
    fn(0);
    const t = performance.now();
    for (let i = 0; i < nCpu; i++) fn(i * 7);
    log(`wasm CPU ${name}: ${((performance.now() - t) / nCpu).toFixed(2)} ms/frame`);
  }
  ts = [];
  for (let k = 0; k < 4; k++) {
    const f0 = k * 16;
    const t = performance.now();
    cpu_chain(frames.subarray(f0 * hw, (f0 + 16) * hw), shifts.subarray(2 * f0, 2 * (f0 + 16)), 16, h, w);
    ts.push((performance.now() - t) / 16);
  }
  log(`wasm CPU median+shift+tophat chain, 1 worker: ${median(ts).toFixed(2)} ms/frame`);
}
