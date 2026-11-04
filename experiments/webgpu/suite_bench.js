// Speed of the GPU preprocessing chain on 600x600 real frames: per kernel, per batch size,
// sequential vs pipelined, and the upload / readback share.
import { Prep, initDevice } from "./gpuprep.js";

const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

export async function run({ log, get, q }) {
  const { adapter, device, initMs } = await initDevice();
  log(
    `adapter ${adapter.info?.vendor}/${adapter.info?.architecture}, init ${initMs.toFixed(0)} ms, hardwareConcurrency ${navigator.hardwareConcurrency}`,
  );
  const meta = JSON.parse(new TextDecoder().decode(await get("full_meta.json")));
  const { n: N, h, w } = meta;
  const hw = h * w;
  const frames = await get("full_frames.u8");
  const shiftsAll = new Float32Array((await get("full_shifts.f32")).buffer);
  const chain = ["median", "shift", "tophat"];
  const reps = Number(q.get("reps") ?? 5);
  const sizes = (q.get("sizes") ?? "16,32,64,128,256").split(",").map(Number);

  // ---- per kernel, data resident (B = 64) ----
  {
    const B = 64;
    const prep = new Prep(device, { h, w, maxB: B });
    log(`shader compile ${(await prep.compile()).toFixed(0)} ms (10 pipelines, cold cache)`);
    prep.upload(0, frames.subarray(0, B * hw), shiftsAll.subarray(0, 2 * B), B);
    const warm = device.createCommandEncoder();
    prep.encode(warm, 0, B, chain);
    device.queue.submit([warm.finish()]);
    await device.queue.onSubmittedWorkDone();
    const R = 10;
    const plan = prep.plan(0, B, chain);
    const names = [...new Set(plan.map((p) => p.name))];
    const per = {};
    for (const name of names) {
      const ts = [];
      for (let k = 0; k < reps; k++) {
        const enc = device.createCommandEncoder();
        for (const st of plan.filter((p) => p.name === name)) {
          for (let r = 0; r < R; r++) {
            const p = enc.beginComputePass();
            p.setPipeline(st.pipe);
            p.setBindGroup(0, prep.bind(st.pipe, st.buffers));
            p.dispatchWorkgroups(st.x, st.y, st.z);
            p.end();
          }
        }
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        ts.push((performance.now() - t0) / (R * B));
      }
      per[name] = median(ts);
    }
    for (const [k, v] of Object.entries(per))
      log(`  kernel ${k.padEnd(12)} ${v.toFixed(4)} ms/frame (GPU, resident, B=${B})`);
    log(
      `  sum of kernels ${Object.values(per)
        .reduce((a, b) => a + b, 0)
        .toFixed(4)} ms/frame`,
    );
    prep.destroy();
  }

  // ---- whole chain per batch size ----
  for (const B of sizes) {
    const prep = new Prep(device, { h, w, maxB: B, slots: 3 });
    await prep.compile();
    const batches = Math.floor(N / B) || 1;
    const nb = Math.min(B, N);
    const fr = (i) => frames.subarray(((i * nb) % N) * hw, (((i * nb) % N) + nb) * hw);
    const sh = (i) => shiftsAll.subarray(2 * ((i * nb) % N), 2 * (((i * nb) % N) + nb));
    // GPU compute only (resident)
    prep.upload(0, fr(0), sh(0), nb);
    let ts = [];
    for (let k = 0; k < reps + 1; k++) {
      const enc = device.createCommandEncoder();
      prep.encode(enc, 0, nb, chain);
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (k > 0) ts.push((performance.now() - t0) / nb);
    }
    const compute = median(ts);
    // upload only
    ts = [];
    for (let k = 0; k < reps + 1; k++) {
      const t0 = performance.now();
      prep.upload(0, fr(k), sh(k), nb);
      await device.queue.onSubmittedWorkDone();
      if (k > 0) ts.push((performance.now() - t0) / nb);
    }
    const up = median(ts);
    // readback only
    ts = [];
    const sink = new Uint8Array(nb * hw);
    for (let k = 0; k < reps + 1; k++) {
      const enc = device.createCommandEncoder();
      prep.copyOut(enc, 0, nb);
      device.queue.submit([enc.finish()]);
      const t0 = performance.now();
      await prep.readback(0, nb, sink);
      if (k > 0) ts.push((performance.now() - t0) / nb);
    }
    const down = median(ts);
    // sequential: upload, compute, readback, one batch at a time
    ts = [];
    for (let k = 0; k < reps + 1; k++) {
      const t0 = performance.now();
      await prep.runBatch(fr(k), sh(k), nb, chain, { into: sink });
      if (k > 0) ts.push((performance.now() - t0) / nb);
    }
    const seq = median(ts);
    // pipelined: 3 slots, the next batch is uploaded and queued before the previous one is read
    const total = Math.max(batches, 8);
    const slotsN = 3;
    const t0 = performance.now();
    const pending = [];
    for (let i = 0; i < total; i++) {
      const s = i % slotsN;
      if (pending[s]) await pending[s];
      prep.upload(s, fr(i), sh(i), nb);
      const enc = device.createCommandEncoder();
      prep.encode(enc, s, nb, chain);
      prep.copyOut(enc, s, nb);
      device.queue.submit([enc.finish()]);
      pending[s] = prep.readback(s, nb, sink); // slot buffers are free again once read
    }
    await Promise.all(pending);
    const pipe = (performance.now() - t0) / (total * nb);
    log(
      `B=${String(B).padStart(3)}: compute ${compute.toFixed(3)} + upload ${up.toFixed(3)} + readback ${down.toFixed(3)} ms/frame | sequential ${seq.toFixed(3)} ms/frame (${(1000 / seq).toFixed(0)} fps) | pipelined ${pipe.toFixed(3)} ms/frame (${(1000 / pipe).toFixed(0)} fps, ${total * nb} frames)`,
    );
    prep.destroy();
  }
}
