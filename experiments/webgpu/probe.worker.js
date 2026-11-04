const probe = async () => {
  const out = { hasGpu: "gpu" in navigator, isSecure: self.isSecureContext };
  if (!out.hasGpu) return out;
  try {
    const t0 = performance.now();
    const ad = await navigator.gpu.requestAdapter();
    out.adapter = ad !== null;
    if (ad) {
      const dev = await ad.requestDevice();
      out.deviceMs = performance.now() - t0;
      const buf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const rb = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const mod = dev.createShaderModule({
        code: "@group(0) @binding(0) var<storage, read_write> o: array<u32>; @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) g: vec3u) { o[g.x] = g.x * 7u + 1u; }",
      });
      const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
      const bg = dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: buf } }],
      });
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1);
      pass.end();
      enc.copyBufferToBuffer(buf, 0, rb, 0, 16);
      dev.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      out.roundTrip = [...new Uint32Array(rb.getMappedRange())];
    }
  } catch (e) {
    out.error = String(e);
  }
  return out;
};
postMessage(await probe());
