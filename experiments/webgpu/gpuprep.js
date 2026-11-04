// Driver for the per-frame preprocessing kernels (works on the page and inside a Worker).
import * as K from "./kernels.js";

const STORAGE = GPUBufferUsage.STORAGE;
const DST = GPUBufferUsage.COPY_DST;
const SRC = GPUBufferUsage.COPY_SRC;

export async function initDevice({ wantTimestamps = false } = {}) {
  const t0 = performance.now();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const requiredLimits = {
    maxBufferSize: Math.min(adapter.limits.maxBufferSize, 2 ** 31 - 4),
    maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 2 ** 31 - 4),
  };
  const requiredFeatures = wantTimestamps && adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  device.lost.then((i) => console.error("device lost", i.message));
  device.addEventListener("uncapturederror", (e) => console.error("gpu error", e.error.message));
  return { adapter, device, initMs: performance.now() - t0 };
}

const ceil = (a, b) => Math.ceil(a / b);

/** One slot of upload / result buffers; several let a batch upload while the previous one reads back. */
class Slot {
  constructor(dev, maxPx, maxB) {
    const words = ceil(maxPx, 4);
    this.inPacked = dev.createBuffer({ size: words * 4, usage: STORAGE | DST });
    this.outPacked = dev.createBuffer({ size: words * 4, usage: STORAGE | SRC });
    this.shifts = dev.createBuffer({ size: maxB * 8, usage: STORAGE | DST });
    this.read = dev.createBuffer({ size: words * 4, usage: GPUBufferUsage.MAP_READ | DST });
  }
}

export class Prep {
  /**
   * @param dev GPUDevice
   * @param o {h, w, maxB, median, tophat, slots, blocks}
   */
  constructor(dev, { h, w, maxB, median = 7, tophat = 15, slots = 1, edges = [0, 1 << 30] }) {
    this.dev = dev;
    this.h = h;
    this.w = w;
    this.hw = h * w;
    this.maxB = maxB;
    this.median = median;
    this.tophat = tophat;
    this.edges = edges;
    const maxPx = maxB * this.hw;
    const big = (usage = STORAGE) => dev.createBuffer({ size: ceil(maxPx, 4) * 16, usage });
    this.scratch = [big(), big(), big()];
    this.slots = Array.from({ length: slots }, () => new Slot(dev, maxPx, maxB));
    this.maxp = dev.createBuffer({ size: this.hw * 4, usage: STORAGE | SRC | DST });
    this.blockmax = dev.createBuffer({ size: (edges.length - 1) * this.hw * 4, usage: STORAGE | SRC | DST });
    this.params = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | DST });
    this.pipes = {};
  }

  async compile() {
    const t0 = performance.now();
    const { dev, median, tophat } = this;
    const mk = (code) =>
      dev.createComputePipelineAsync({
        layout: "auto",
        compute: { module: dev.createShaderModule({ code }), entryPoint: "main" },
      });
    const defs = {
      unpack: K.unpack,
      pack: K.pack,
      tophatFinal: K.tophatFinal,
      median: K.median(median),
      shift: K.shift,
      erodeRows: K.morphRows(tophat, true),
      erodeCols: K.morphCols(tophat, true),
      dilateRows: K.morphRows(tophat, false),
      dilateCols: K.morphCols(tophat, false),
      maxProj: K.maxProj,
    };
    const names = Object.keys(defs);
    const pipes = await Promise.all(names.map((n) => mk(defs[n])));
    names.forEach((n, i) => {
      this.pipes[n] = pipes[i];
    });
    return performance.now() - t0;
  }

  bind(pipe, entries) {
    return this.dev.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }

  /** Words of work for the 1-D kernels as a 2-D grid. */
  grid1d(items) {
    const groups = ceil(items, 256);
    return [Math.min(groups, 65535), ceil(groups, 65535)];
  }

  /**
   * The passes of the chain for `n` frames resident in slot `s`: [{name, pipe, buffers, x, y, z}].
   * chain: steps in ['median', 'shift', 'tophat'], tophat last. Result: slot.outPacked.
   */
  plan(s, n, chain) {
    const { pipes, h, w, hw } = this;
    const slot = this.slots[s];
    const total = n * hw;
    this.dev.queue.writeBuffer(this.params, 0, new Uint32Array([h, w, n, total]));
    const S = this.scratch;
    let cur = 0;
    const out = [];
    const add = (name, buffers, x, y, z) =>
      out.push({ name, pipe: pipes[name], buffers: [this.params, ...buffers], x, y, z });
    const [gx, gy] = this.grid1d(ceil(total, 4));
    add("unpack", [slot.inPacked, S[cur]], gx, gy, 1);
    let tophatLast = false;
    for (const step of chain) {
      const nxt = (cur + 1) % 3;
      if (step === "median") {
        add("median", [S[cur], S[nxt]], ceil(w, 16), ceil(h, 16), n);
        cur = nxt;
      } else if (step === "shift") {
        add("shift", [S[cur], S[nxt], slot.shifts], ceil(w, 16), ceil(h, 16), n);
        cur = nxt;
      } else if (step === "tophat") {
        const t1 = nxt;
        const t2 = (cur + 2) % 3;
        const gxw = ceil(w, 32);
        const gyh = ceil(h, 8);
        add("erodeRows", [S[cur], S[t1]], gxw, gyh, n);
        add("erodeCols", [S[t1], S[t2]], gxw, gyh, n);
        add("dilateRows", [S[t2], S[t1]], gxw, gyh, n);
        add("dilateCols", [S[t1], S[t2]], gxw, gyh, n);
        add("tophatFinal", [S[cur], S[t2], slot.outPacked], gx, gy, 1);
        tophatLast = true;
      } else throw new Error(`unknown step ${step}`);
    }
    if (!tophatLast) add("pack", [S[cur], slot.outPacked], gx, gy, 1);
    return out;
  }

  /** Record the passes of the chain into `enc`; `reps` repeats every pass (for timing a kernel on its own). */
  encode(enc, s, n, chain, firstFrame = 0, { fold = false, reps = 1 } = {}) {
    const slot = this.slots[s];
    for (const st of this.plan(s, n, chain)) {
      for (let r = 0; r < reps; r++) {
        const p = enc.beginComputePass();
        p.setPipeline(st.pipe);
        p.setBindGroup(0, this.bind(st.pipe, st.buffers));
        p.dispatchWorkgroups(st.x, st.y, st.z);
        p.end();
      }
    }
    if (fold) this.encodeFold(enc, slot, n, firstFrame);
  }

  /** Max projection and block maxima of the n results in `slot.outPacked` (frames firstFrame.. of the window). */
  encodeFold(enc, slot, n, firstFrame) {
    const { dev, edges, hw } = this;
    const [gx, gy] = this.grid1d(hw);
    for (let blk = 0; blk < edges.length - 1; blk++) {
      const lo = Math.max(edges[blk], firstFrame);
      const hi = Math.min(edges[blk + 1], firstFrame + n);
      if (hi <= lo) continue;
      const q = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | DST });
      dev.queue.writeBuffer(q, 0, new Uint32Array([hw, lo - firstFrame, hi - firstFrame, blk]));
      const p = enc.beginComputePass();
      p.setPipeline(this.pipes.maxProj);
      p.setBindGroup(0, this.bind(this.pipes.maxProj, [q, slot.outPacked, this.maxp, this.blockmax]));
      p.dispatchWorkgroups(gx, gy, 1);
      p.end();
    }
  }

  upload(s, frames, shifts, n) {
    const q = this.dev.queue;
    const slot = this.slots[s];
    const bytes = n * this.hw;
    const whole = bytes & ~3;
    q.writeBuffer(slot.inPacked, 0, frames.buffer, frames.byteOffset, whole);
    if (bytes > whole) {
      const tail = new Uint8Array(4);
      tail.set(frames.subarray(whole, bytes));
      q.writeBuffer(slot.inPacked, whole, tail);
    }
    if (shifts) q.writeBuffer(slot.shifts, 0, shifts.buffer, shifts.byteOffset, n * 8);
  }

  /** Start the readback of the slot's n results; resolves to a Uint8Array copy. */
  async readback(s, n, into) {
    const slot = this.slots[s];
    const bytes = n * this.hw;
    await slot.read.mapAsync(GPUMapMode.READ, 0, ceil(bytes, 4) * 4);
    const view = new Uint8Array(slot.read.getMappedRange(0, ceil(bytes, 4) * 4), 0, bytes);
    const out = into ?? new Uint8Array(bytes);
    out.set(view);
    slot.read.unmap();
    return out;
  }

  copyOut(enc, s, n) {
    const slot = this.slots[s];
    enc.copyBufferToBuffer(slot.outPacked, 0, slot.read, 0, ceil(n * this.hw, 4) * 4);
  }

  /** upload + compute + readback of one batch, awaiting everything. */
  async runBatch(frames, shifts, n, chain, { s = 0, firstFrame = 0, fold = false, into } = {}) {
    this.upload(s, frames, shifts, n);
    const enc = this.dev.createCommandEncoder();
    this.encode(enc, s, n, chain, firstFrame, { fold });
    this.copyOut(enc, s, n);
    this.dev.queue.submit([enc.finish()]);
    return this.readback(s, n, into);
  }

  async readMaxima() {
    const { dev } = this;
    const nb = this.edges.length - 1;
    const size = (nb + 1) * this.hw * 4;
    const rb = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.maxp, 0, rb, 0, this.hw * 4);
    enc.copyBufferToBuffer(this.blockmax, 0, rb, this.hw * 4, nb * this.hw * 4);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const all = new Uint32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return { maxp: Uint8Array.from(all.subarray(0, this.hw)), blockmax: Uint8Array.from(all.subarray(this.hw)) };
  }

  destroy() {
    for (const b of this.scratch) b.destroy();
    for (const s of this.slots) for (const b of Object.values(s)) b.destroy();
    this.maxp.destroy();
    this.blockmax.destroy();
  }
}
