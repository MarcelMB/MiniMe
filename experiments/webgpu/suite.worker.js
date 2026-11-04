// Suites run inside a dedicated Worker: check (bit-exactness), bench (speed).
import { Prep, initDevice } from "./gpuprep.js";

const q = new URL(self.location.href).searchParams;
const log = (...a) => postMessage({ log: a.join(" ") });
const get = async (name) => new Uint8Array(await (await fetch(`/data/${name}`)).arrayBuffer());

export function compare(got, want) {
  let bad = 0;
  let maxAbs = 0;
  let first = null;
  for (let i = 0; i < want.length; i++) {
    if (got[i] !== want[i]) {
      bad++;
      const d = Math.abs(got[i] - want[i]);
      if (d > maxAbs) maxAbs = d;
      if (!first) first = [i, got[i], want[i]];
    }
  }
  return { bad, maxAbs, first, n: want.length };
}
const fmt = (c) =>
  c.bad === 0 ? `EXACT (${c.n} bytes)` : `MISMATCH ${c.bad}/${c.n} maxAbs=${c.maxAbs} first@${c.first}`;

async function check() {
  const { adapter, device, initMs } = await initDevice();
  log(`adapter ${adapter.info?.vendor}/${adapter.info?.architecture} device init ${initMs.toFixed(0)} ms (in worker)`);
  for (const tag of ["full", "odd"]) {
    const meta = JSON.parse(new TextDecoder().decode(await get(`${tag}_meta.json`)));
    const { n, h, w, edges } = meta;
    const hw = h * w;
    const frames = await get(`${tag}_frames.u8`);
    const shifts = new Float32Array((await get(`${tag}_shifts.f32`)).buffer);
    const B = Math.min(64, n);
    const prep = new Prep(device, { h, w, maxB: B, edges });
    const tc = await prep.compile();
    log(`[${tag}] ${n} frames ${h}x${w}, batch ${B}, shaders compiled in ${tc.toFixed(0)} ms`);
    const tests = [
      ["median 7x7", ["median"], "median"],
      ["shift (raw frame)", ["shift"], "shift_raw"],
      ["shift (median frame)", ["median", "shift"], "shift_median"],
      ["tophat 15 (raw frame)", ["tophat"], "tophat_raw"],
      ["median+shift+tophat", ["median", "shift", "tophat"], "final"],
    ];
    for (const [name, chain, ref] of tests) {
      const want = await get(`${tag}_${ref}.u8`);
      const got = new Uint8Array(n * hw);
      const fold = ref === "final";
      for (let f0 = 0; f0 < n; f0 += B) {
        const nb = Math.min(B, n - f0);
        const out = await prep.runBatch(
          frames.subarray(f0 * hw, (f0 + nb) * hw),
          shifts.subarray(2 * f0, 2 * (f0 + nb)),
          nb,
          chain,
          { firstFrame: f0, fold },
        );
        got.set(out, f0 * hw);
      }
      log(`[${tag}] ${name}: ${fmt(compare(got, want))}`);
      if (fold) {
        const { maxp, blockmax } = await prep.readMaxima();
        log(`[${tag}] max projection: ${fmt(compare(maxp, await get(`${tag}_maxproj.u8`)))}`);
        log(
          `[${tag}] block maxima (${edges.length - 1} blocks): ${fmt(compare(blockmax, await get(`${tag}_blockmax.u8`)))}`,
        );
      }
    }
    prep.destroy();
  }
}

const suite = q.get("suite") ?? "check";
try {
  if (suite === "check") await check();
  else {
    const m = await import(`./suite_${suite}.js`);
    await m.run({ log, get, compare, fmt, q });
  }
  postMessage({ done: { ok: true } });
} catch (e) {
  log(`FAILED: ${e?.stack ?? e}`);
  postMessage({ done: { error: String(e) } });
}
