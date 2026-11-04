// The registration image (motion.bandpass) on the GPU against rust/minime-core's CPU result, bit for bit (float32).
import * as K from "./kernels.js";
import { initDevice } from "./gpuprep.js";

const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

export async function run({ log, get }) {
  const { device } = await initDevice();
  const meta = JSON.parse(new TextDecoder().decode(await get("full_meta.json")));
  const { h, w } = meta;
  const B = 64;
  const hw = h * w;
  const hh = h / 2;
  const ww = w / 2;
  const den = (await get("full_median.u8")).subarray(0, B * hw);
  const want = new Uint32Array((await get("full_bandpass.f32")).buffer);
  const taps = new Uint8Array((await get("bp_taps.f32")).slice(0).buffer);
  const S = GPUBufferUsage.STORAGE;
  const D = GPUBufferUsage.COPY_DST;
  const mk = (size, usage = S) => device.createBuffer({ size, usage });
  const u8taps = [1, 14, 62, 102, 62, 14, 1];
  const mod = (code) =>
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
  const pipes = {
    unpack: await mod(K.unpack),
    gu8r: await mod(K.gaussU8Rows(u8taps)),
    gu8c: await mod(K.gaussU8Cols(u8taps)),
    down: await mod(K.areaDown2),
    gf32r: await mod(K.gaussF32Rows(25)),
    gf32c: await mod(K.gaussF32Cols(25)),
    upsub: await mod(K.upSub),
  };
  const params = mk(16, GPUBufferUsage.UNIFORM | D);
  device.queue.writeBuffer(params, 0, new Uint32Array([h, w, B, B * hw]));
  const packed = mk(B * hw, S | D);
  device.queue.writeBuffer(packed, 0, den);
  const tapBuf = mk(100, S | D);
  device.queue.writeBuffer(tapBuf, 0, taps);
  const denB = mk(B * hw * 4);
  const rowsU = mk(B * hw * 4);
  const narrow = mk(B * hw * 4);
  const half = mk(B * hh * ww * 4);
  const rowsH = mk(B * hh * ww * 4);
  const wideLow = mk(B * hh * ww * 4);
  const out = mk(B * hw * 4, S | GPUBufferUsage.COPY_SRC);
  const bind = (pipe, bufs) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [params, ...bufs].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  const c = (n, d) => Math.ceil(n / d);
  const g1 = Math.ceil((B * hw) / 4 / 256);
  const passes = [
    ["unpack", pipes.unpack, [packed, denB], [Math.min(g1, 65535), Math.ceil(g1 / 65535), 1]],
    ["gaussU8 rows", pipes.gu8r, [denB, rowsU], [c(w, 16), c(h, 16), B]],
    ["gaussU8 cols", pipes.gu8c, [rowsU, narrow], [c(w, 16), c(h, 16), B]],
    ["area down 2", pipes.down, [denB, half], [c(ww, 16), c(hh, 16), B]],
    ["gaussF32 rows", pipes.gf32r, [half, rowsH, tapBuf], [c(ww, 16), c(hh, 16), B]],
    ["gaussF32 cols", pipes.gf32c, [rowsH, wideLow, tapBuf], [c(ww, 16), c(hh, 16), B]],
    ["resize up 2 + subtract", pipes.upsub, [wideLow, narrow, out], [c(w, 16), c(h, 16), B]],
  ].map(([name, pipe, bufs, grid]) => ({ name, pipe, bg: bind(pipe, bufs), grid }));
  const enc = device.createCommandEncoder();
  for (const p of passes) {
    const cp = enc.beginComputePass();
    cp.setPipeline(p.pipe);
    cp.setBindGroup(0, p.bg);
    cp.dispatchWorkgroups(...p.grid);
    cp.end();
  }
  const rd = mk(B * hw * 4, GPUBufferUsage.MAP_READ | D);
  enc.copyBufferToBuffer(out, 0, rd, 0, B * hw * 4);
  device.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  const got = new Uint32Array(rd.getMappedRange().slice(0));
  rd.unmap();
  let bad = 0;
  let maxUlp = 0;
  let first = null;
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  for (let i = 0; i < B * hw; i++) {
    if (got[i] !== want[i]) {
      bad++;
      u[0] = got[i];
      const a = f[0];
      u[0] = want[i];
      const b = f[0];
      maxUlp = Math.max(maxUlp, Math.abs(a - b));
      first ??= [i, a, b];
    }
  }
  log(
    `bandpass registration image (${B} frames ${h}x${w}) vs CPU: ${bad === 0 ? `EXACT, float32 bit for bit (${B * hw} values)` : `MISMATCH ${bad}/${B * hw}, max abs diff ${maxUlp}, first ${first}`}`,
  );
  // float32 shift of the registration image (what Pass1::push does for the quality), from the CPU's bandpass result
  {
    const shifts = new Float32Array((await get("full_shifts.f32")).buffer).subarray(0, 2 * B);
    const srcB = mk(B * hw * 4, S | D);
    device.queue.writeBuffer(srcB, 0, new Uint8Array((await get("full_bandpass.f32")).buffer));
    const shB = mk(B * 8, S | D);
    device.queue.writeBuffer(shB, 0, shifts);
    const dstB = mk(B * hw * 4, S | GPUBufferUsage.COPY_SRC);
    const pipe = await mod(K.shiftF32);
    const e = device.createCommandEncoder();
    const cp = e.beginComputePass();
    cp.setPipeline(pipe);
    cp.setBindGroup(0, bind(pipe, [srcB, dstB, shB]));
    cp.dispatchWorkgroups(c(w, 16), c(h, 16), B);
    cp.end();
    const rd2 = mk(B * hw * 4, GPUBufferUsage.MAP_READ | D);
    e.copyBufferToBuffer(dstB, 0, rd2, 0, B * hw * 4);
    device.queue.submit([e.finish()]);
    await rd2.mapAsync(GPUMapMode.READ);
    const g2 = new Uint32Array(rd2.getMappedRange().slice(0));
    rd2.unmap();
    const w2 = new Uint32Array((await get("full_shift_f32.f32")).buffer);
    let bad2 = 0;
    for (let i = 0; i < g2.length; i++) if (g2[i] !== w2[i]) bad2++;
    log(
      `shift_f32 of the registration image: ${bad2 === 0 ? `EXACT, float32 bit for bit (${g2.length} values)` : `MISMATCH ${bad2}/${g2.length}`}`,
    );
    const tsRead = [];
    for (let k = 0; k < 5; k++) {
      const e2 = device.createCommandEncoder();
      e2.copyBufferToBuffer(dstB, 0, rd2, 0, B * hw * 4);
      device.queue.submit([e2.finish()]);
      const t0 = performance.now();
      await rd2.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(rd2.getMappedRange().slice(0));
      rd2.unmap();
      tsRead.push((performance.now() - t0) / B + copy[0] * 0);
    }
    log(`  readback of the float32 image (1.44 MB/frame): ${median(tsRead).toFixed(3)} ms/frame`);
  }
  // speed, kernel by kernel (resident)
  const R = 10;
  let total = 0;
  for (const p of passes) {
    const ts = [];
    for (let k = 0; k < 5; k++) {
      const e = device.createCommandEncoder();
      for (let r = 0; r < R; r++) {
        const cp = e.beginComputePass();
        cp.setPipeline(p.pipe);
        cp.setBindGroup(0, p.bg);
        cp.dispatchWorkgroups(...p.grid);
        cp.end();
      }
      const t0 = performance.now();
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
      ts.push((performance.now() - t0) / (R * B));
    }
    const m = median(ts);
    if (p.name !== "unpack") total += m;
    log(`  ${p.name.padEnd(24)} ${m.toFixed(4)} ms/frame`);
  }
  log(`  registration image total (without unpack) ${total.toFixed(4)} ms/frame`);
}
