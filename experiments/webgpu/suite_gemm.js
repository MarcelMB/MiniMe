// Pass 2 on real 64x64 tiles (T = 3000, P = 4096): WGSL tiled GEMM (hand-written, driven from TS and through
// wgpu in wasm) against the app's CPU GEMM (matrixmultiply, wasm simd128, one worker).
import init, {
  GpuPass2,
  p2_cpu_compress,
  p2_cpu_matmul,
  p2_cpu_new,
  p2_cpu_noise,
  p2_cpu_products,
  p2_cpu_t_matmul,
  p2_mean,
} from "./pkg_gpu/gpu_spike.js";
import * as K from "./kernels.js";
import { initDevice } from "./gpuprep.js";

const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const T = 3000;
const P = 4096;

function relErr(got, want) {
  let num = 0;
  let den = 0;
  let maxAbs = 0;
  for (let i = 0; i < want.length; i++) {
    const d = got[i] - want[i];
    num += d * d;
    den += want[i] * want[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  let maxWant = 0;
  for (let i = 0; i < want.length; i++) maxWant = Math.max(maxWant, Math.abs(want[i]));
  return { rel: Math.sqrt(num / den), maxAbs, maxWant };
}

export async function run({ log, get, q }) {
  await init();
  const reps = Number(q.get("reps") ?? 5);
  const tileId = q.get("tile") ?? "11";
  const tile = await get(`tiles/tile_${tileId}.u8`);
  log(`tile ${tileId}: ${T} x ${P} uint8 (${(tile.length / 1e6).toFixed(1)} MB)`);

  // ---- CPU in this worker (wasm simd128, 1 thread) ----
  const timeIt = (name, fn, n = 3) => {
    fn();
    const ts = [];
    for (let i = 0; i < n; i++) {
      const t = performance.now();
      fn();
      ts.push(performance.now() - t);
    }
    log(`wasm CPU ${name}: ${median(ts).toFixed(1)} ms`);
    return median(ts);
  };
  timeIt("TileMatrix::new (mean/energy pass)", () => p2_cpu_new(tile, T, P));
  const tNoise = timeIt("noise (rfft of every pixel, noise_frames=0)", () => p2_cpu_noise(tile, T, P), 2);
  const tProd = timeIt("4 products ell=50 (one rSVD at k=40, 1 power iteration)", () =>
    p2_cpu_products(tile, T, P, 50, 1),
  );
  const tAll = await (async () => {
    await p2_cpu_compress(tile, T, P);
    const ts = [];
    for (let i = 0; i < 3; i++) {
      const t = performance.now();
      await p2_cpu_compress(tile, T, P);
      ts.push(performance.now() - t);
    }
    log(`wasm CPU whole rSVD + rank cut (no noise): ${median(ts).toFixed(1)} ms`);
    return median(ts);
  })();
  log(
    `  => per tile in wasm: noise ${tNoise.toFixed(0)} + rSVD ${tAll.toFixed(0)} (of which 4 products ${tProd.toFixed(0)}) ms; products are ${((100 * tProd) / (tNoise + tAll)).toFixed(0)} % of the tile at k=40`,
  );

  // ---- GPU from TypeScript ----
  const { device } = await initDevice();
  const mean = p2_mean(tile, T, P);
  const mk = (size, usage) => device.createBuffer({ size, usage });
  const S = GPUBufferUsage.STORAGE;
  const D = GPUBufferUsage.COPY_DST;
  const tt0 = performance.now();
  const rawBuf = mk(tile.length, S | D);
  device.queue.writeBuffer(rawBuf, 0, tile);
  const meanBuf = mk(P * 4, S | D);
  device.queue.writeBuffer(meanBuf, 0, mean);
  await device.queue.onSubmittedWorkDone();
  const tUpload = performance.now() - tt0;
  log(`GPU tile upload (12.3 MB): ${tUpload.toFixed(1)} ms`);
  const pipes = {
    t: await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: K.gemm(false) }), entryPoint: "main" },
    }),
    n: await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: K.gemm(true) }), entryPoint: "main" },
    }),
  };
  // one product; `resident` keeps B and C on the GPU and only times the kernel
  async function product(kind, b, ell, { timeKernel = 0 } = {}) {
    const m = kind === "t" ? T : P;
    const k = kind === "t" ? P : T;
    const g = mk(16, GPUBufferUsage.UNIFORM | D);
    device.queue.writeBuffer(g, 0, new Uint32Array([m, ell, k, P]));
    const bb = mk(b.byteLength, S | D);
    device.queue.writeBuffer(bb, 0, b);
    const out = mk(m * ell * 4, S | GPUBufferUsage.COPY_SRC);
    const rd = mk(m * ell * 4, GPUBufferUsage.MAP_READ | D);
    const bg = device.createBindGroup({
      layout: pipes[kind].getBindGroupLayout(0),
      entries: [g, rawBuf, meanBuf, bb, out].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const enc = device.createCommandEncoder();
    const reps = Math.max(1, timeKernel);
    for (let r = 0; r < reps; r++) {
      const p = enc.beginComputePass();
      p.setPipeline(pipes[kind]);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(Math.ceil(ell / 64), Math.ceil(m / 64));
      p.end();
    }
    enc.copyBufferToBuffer(out, 0, rd, 0, m * ell * 4);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(rd.getMappedRange().slice(0));
    rd.unmap();
    const ms = performance.now() - t0;
    for (const x of [g, bb, out, rd]) x.destroy();
    return { res, ms };
  }
  const ell = 50;
  const rnd = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
  const qm = rnd(P * ell);
  const zm = rnd(T * ell);
  // accuracy against the app's CPU result
  const wantT = p2_cpu_t_matmul(tile, T, P, qm, ell);
  const wantN = p2_cpu_matmul(tile, T, P, zm, ell);
  const gotT = (await product("t", qm, ell)).res;
  const gotN = (await product("n", zm, ell)).res;
  const eT = relErr(gotT, wantT);
  const eN = relErr(gotN, wantN);
  log(
    `GPU vs CPU sgemm, Y^T Q (T x ell): rel Frobenius err ${eT.rel.toExponential(2)}, max abs ${eT.maxAbs.toExponential(2)} (values up to ~${eT.maxWant.toExponential(1)})`,
  );
  log(
    `GPU vs CPU sgemm, Y Z (P x ell): rel Frobenius err ${eN.rel.toExponential(2)}, max abs ${eN.maxAbs.toExponential(2)}`,
  );
  // speed: kernel only (10 repeats in one submit), then a full product with upload + readback
  const flop = () => 2 * T * P * ell;
  for (const kind of ["t", "n"]) {
    const b = kind === "t" ? qm : zm;
    const ts = [];
    for (let i = 0; i < reps; i++) ts.push((await product(kind, b, ell, { timeKernel: 20 })).ms);
    const full = [];
    for (let i = 0; i < reps; i++) full.push((await product(kind, b, ell)).ms);
    const kernel = (median(ts) - median(full)) / 19; // (20 kernels + transfer) - (1 kernel + transfer)
    log(
      `GPU ${kind === "t" ? "Y^T Q" : "Y Z  "} ell=${ell}: kernel ~${kernel.toFixed(2)} ms (${(flop() / (kernel / 1000) / 1e9).toFixed(0)} GFLOP/s), complete product with upload+readback of the small matrices ${median(full).toFixed(2)} ms`,
    );
  }
  // 4 products in a rSVD sequence, each awaited as the CPU needs the result for the QR
  const seq = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await product("n", zm, ell);
    await product("t", qm, ell);
    await product("n", zm, ell);
    await product("t", qm, ell);
    seq.push(performance.now() - t0);
  }
  log(
    `GPU 4 products ell=50 in sequence (TS-driven, round trip each): ${median(seq).toFixed(2)} ms  (wasm CPU: ${tProd.toFixed(0)} ms)`,
  );

  // ---- the same through the wgpu crate in wasm, and the whole rSVD ----
  const g2 = await GpuPass2.create();
  let t0 = performance.now();
  g2.set_tile(tile, T, P);
  log(`wgpu-in-wasm set_tile (mean pass + upload + 2 pipelines) ${(performance.now() - t0).toFixed(1)} ms`);
  const sq = [];
  for (let i = 0; i < reps + 1; i++) {
    t0 = performance.now();
    await g2.matmul(zm, ell);
    await g2.t_matmul(qm, ell);
    await g2.matmul(zm, ell);
    await g2.t_matmul(qm, ell);
    if (i > 0) sq.push(performance.now() - t0);
  }
  log(`wgpu-in-wasm 4 products ell=50 in sequence: ${median(sq).toFixed(2)} ms`);
  log(`wgpu-in-wasm CPU-vs-GPU rSVD + rank cut on tile ${tileId}: ${await g2.compare(tile, T, P)}`);
  const whole = [];
  await g2.compress(tile, T, P);
  for (let i = 0; i < 3; i++) {
    t0 = performance.now();
    await g2.compress(tile, T, P);
    whole.push(performance.now() - t0);
  }
  log(`wgpu-in-wasm whole rSVD + rank cut: ${median(whole).toFixed(1)} ms (wasm CPU: ${tAll.toFixed(1)} ms)`);
  const tileWasmCpu = tNoise + tAll;
  const tileWasmGpu = tNoise + median(whole);
  log(
    `=> tile with the noise FFT still on the CPU: ${tileWasmCpu.toFixed(0)} ms (CPU) vs ${tileWasmGpu.toFixed(0)} ms (GPU products) = ${(tileWasmCpu / tileWasmGpu).toFixed(2)}x`,
  );
}
