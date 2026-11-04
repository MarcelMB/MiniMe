// WGSL kernels for MiniMe's per-frame preprocessing (median, shift, tophat, max projection),
// checked byte for byte against rust/minime-core/src/preprocess.rs.
//
// Layout: a batch of `n` frames of h x w pixels is uploaded as packed bytes (u32 words, little
// endian). Intermediates hold one u32 per pixel (simple and fast enough: the kernels are
// compute-bound, not bandwidth-bound). The result is packed again, 4 pixels per word.

const P = /* wgsl */ `
struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;
`;

// one thread per packed word, 2-D dispatch to stay under 65535 workgroups per dimension
const WORD_ID = /* wgsl */ `
fn word_id(g: vec3u, nw: vec3u) -> u32 { return g.y * nw.x * 256u + g.x; }
`;

export const unpack = `${P}${WORD_ID}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = word_id(g, nw);
  if (i >= (p.total + 3u) / 4u) { return; }
  let v = src[i];
  dst[i] = vec4<u32>(v & 255u, (v >> 8u) & 255u, (v >> 16u) & 255u, v >> 24u);
}`;

// last step when no tophat: pack one u32 per pixel back to bytes
export const pack = `${P}${WORD_ID}
@group(0) @binding(1) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = word_id(g, nw);
  if (i >= (p.total + 3u) / 4u) { return; }
  let a = src[i];
  dst[i] = a.x | (a.y << 8u) | (a.z << 16u) | (a.w << 24u);
}`;

// white tophat result: orig - opened (saturating), packed
export const tophatFinal = `${P}${WORD_ID}
@group(0) @binding(1) var<storage, read> orig: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> opened: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = word_id(g, nw);
  if (i >= (p.total + 3u) / 4u) { return; }
  let a = orig[i];
  let o = opened[i];
  let d = select(vec4<u32>(0u), a - o, a >= o);
  dst[i] = d.x | (d.y << 8u) | (d.z << 16u) | (d.w << 24u);
}`;

/** `cv2.medianBlur`, k x k (odd), replicated border: the smallest v with #(x <= v) > k*k/2. */
export const median = (k) => {
  const r = (k - 1) / 2;
  const n = k * k;
  const tw = 16 + 2 * r;
  return `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, ${tw * tw}>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let hw = p.h * p.w;
  let base = wid.z * hw;
  let x0 = i32(wid.x * 16u);
  let y0 = i32(wid.y * 16u);
  let lin = lid.y * 16u + lid.x;
  for (var i = lin; i < ${tw * tw}u; i += 256u) {
    let tx = i32(i % ${tw}u);
    let ty = i32(i / ${tw}u);
    let sx = clamp(x0 + tx - ${r}, 0, i32(p.w) - 1);
    let sy = clamp(y0 + ty - ${r}, 0, i32(p.h) - 1);
    tile[i] = src[base + u32(sy) * p.w + u32(sx)];
  }
  workgroupBarrier();
  var v: array<u32, ${n}>;
  for (var j = 0u; j < ${k}u; j++) {
    for (var i = 0u; i < ${k}u; i++) {
      v[j * ${k}u + i] = tile[(lid.y + j) * ${tw}u + lid.x + i];
    }
  }
  var lo = 0u;
  var hi = 255u;
  for (var it = 0; it < 8; it++) {
    let mid = (lo + hi) >> 1u;
    var c = 0u;
    for (var q = 0u; q < ${n}u; q++) { c += select(0u, 1u, v[q] <= mid); }
    if (c > ${Math.floor(n / 2)}u) { hi = mid; } else { lo = mid + 1u; }
  }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = lo; }
}`;
};

/** `cv2.warpAffine` pure shift, INTER_LINEAR, OpenCV 5 style fused float32 lerps, round half even, zeros outside. */
export const shift = `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> shifts: array<vec2<f32>>;
fn at(base: u32, yy: i32, xx: i32) -> f32 {
  if (yy >= 0 && yy < i32(p.h) && xx >= 0 && xx < i32(p.w)) {
    return f32(src[base + u32(yy) * p.w + u32(xx)]);
  }
  return 0.0;
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= i32(p.w) || y >= i32(p.h)) { return; }
  let base = g.z * p.h * p.w;
  let s = shifts[g.z];
  let sy = f32(y) + s.x;
  let sx = f32(x) + s.y;
  let y0f = floor(sy);
  let x0f = floor(sx);
  let ay = sy - y0f;
  let ax = sx - x0f;
  let y0 = i32(y0f);
  let x0 = i32(x0f);
  let p00 = at(base, y0, x0);
  let p01 = at(base, y0, x0 + 1);
  let p10 = at(base, y0 + 1, x0);
  let p11 = at(base, y0 + 1, x0 + 1);
  let top = fma(p01 - p00, ax, p00);
  let bot = fma(p11 - p10, ax, p10);
  let v = fma(bot - top, ay, top);
  dst[base + u32(y) * p.w + u32(x)] = u32(clamp(round(v), 0.0, 255.0));
}`;

/** Running min / max over 2r+1 samples along rows, samples beyond the ends ignored. */
export const morphRows = (r, isMin) => {
  const tw = 32 + 2 * r;
  return `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, ${8 * tw}>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.z * p.h * p.w;
  let x0 = i32(wid.x * 32u);
  let y0 = i32(wid.y * 8u);
  let lin = lid.y * 32u + lid.x;
  for (var i = lin; i < ${8 * tw}u; i += 256u) {
    let tx = i32(i % ${tw}u);
    let ty = i32(i / ${tw}u);
    let sx = x0 + tx - ${r};
    let sy = y0 + ty;
    var v = ${isMin ? "255u" : "0u"};
    if (sx >= 0 && sx < i32(p.w) && sy < i32(p.h)) { v = src[base + u32(sy) * p.w + u32(sx)]; }
    tile[i] = v;
  }
  workgroupBarrier();
  var acc = ${isMin ? "255u" : "0u"};
  for (var k = 0u; k < ${2 * r + 1}u; k++) { acc = ${isMin ? "min" : "max"}(acc, tile[lid.y * ${tw}u + lid.x + k]); }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = acc; }
}`;
};

/** The same along columns. */
export const morphCols = (r, isMin) => {
  const th = 8 + 2 * r;
  return `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, ${th * 32}>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.z * p.h * p.w;
  let x0 = i32(wid.x * 32u);
  let y0 = i32(wid.y * 8u);
  let lin = lid.y * 32u + lid.x;
  for (var i = lin; i < ${th * 32}u; i += 256u) {
    let tx = i32(i % 32u);
    let ty = i32(i / 32u);
    let sx = x0 + tx;
    let sy = y0 + ty - ${r};
    var v = ${isMin ? "255u" : "0u"};
    if (sx < i32(p.w) && sy >= 0 && sy < i32(p.h)) { v = src[base + u32(sy) * p.w + u32(sx)]; }
    tile[i] = v;
  }
  workgroupBarrier();
  var acc = ${isMin ? "255u" : "0u"};
  for (var k = 0u; k < ${2 * r + 1}u; k++) { acc = ${isMin ? "min" : "max"}(acc, tile[(lid.y + k) * 32u + lid.x]); }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = acc; }
}`;
};

/** Max projection and block maximum over frames f0..f1 of the packed result (u32 per pixel accumulators). */
export const maxProj = /* wgsl */ `
struct Q { hw: u32, f0: u32, f1: u32, blk: u32 }
@group(0) @binding(0) var<uniform> q: Q;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> maxp: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockmax: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let px = g.y * nw.x * 256u + g.x;
  if (px >= q.hw) { return; }
  var m = 0u;
  for (var f = q.f0; f < q.f1; f++) {
    let i = f * q.hw + px;
    m = max(m, (packed[i >> 2u] >> ((i & 3u) * 8u)) & 255u);
  }
  maxp[px] = max(maxp[px], m);
  blockmax[q.blk * q.hw + px] = max(blockmax[q.blk * q.hw + px], m);
}`;

/**
 * Pass 2 products on a resident uint8 tile `(T, P)` with the per-pixel mean removed on the fly, f32.
 * `t_matmul` (Y^T Q): out (T, N) = A (T x P) * B (P x N);  `matmul` (Y Z): out (P, N) = A^T (P x T) * B (T x N).
 * 64 x 64 output tile per workgroup of 16 x 16 threads, each 4 x 4 outputs, K in steps of 16 through shared memory.
 */
export const gemm = (transposed) => /* wgsl */ `
struct G { m: u32, n: u32, k: u32, p: u32 }
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> bm: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lin = lid.y * 16u + lid.x;
  let m0 = wid.y * 64u;
  let n0 = wid.x * 64u;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < g.k; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let idx = lin + i * 256u;
      ${
        transposed
          ? `let mm = idx % 64u; let kk = idx / 64u;
      let m = m0 + mm; let k = k0 + kk;
      var v = 0.0;
      if (m < g.m && k < g.k) { let a = k * g.p + m; v = f32((raw[a >> 2u] >> ((a & 3u) * 8u)) & 255u) - mean[m]; }`
          : `let kk = idx % 16u; let mm = idx / 16u;
      let m = m0 + mm; let k = k0 + kk;
      var v = 0.0;
      if (m < g.m && k < g.k) { let a = m * g.p + k; v = f32((raw[a >> 2u] >> ((a & 3u) * 8u)) & 255u) - mean[k]; }`
      }
      As[kk * 64u + mm] = v;
      let nn = idx % 64u;
      let kb = idx / 64u;
      let n = n0 + nn;
      let kr = k0 + kb;
      var bv = 0.0;
      if (n < g.n && kr < g.k) { bv = bm[kr * g.n + n]; }
      Bs[kb * 64u + nn] = bv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[kk * 64u + lid.y * 4u + i]; b[i] = Bs[kk * 64u + lid.x * 4u + i]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]); } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let m = m0 + lid.y * 4u + i;
      let n = n0 + lid.x * 4u + j;
      if (m < g.m && n < g.n) { c[m * g.n + n] = acc[i * 4u + j]; }
    }
  }
}`;

// ---------------------------------------------------------------------------
// Registration image: motion.bandpass = gauss_u8(den, sigma lo) - resize_up2(gauss_f32(area_down2(den), hi / 2)),
// every step ported from rust/minime-core/src/filters.rs (OpenCV 5 / KleidiCV on arm64).
// ---------------------------------------------------------------------------

const REFL = /* wgsl */ `
fn refl(i: i32, n: i32) -> i32 {
  var j = i;
  if (j < 0) { j = -j; }
  if (j >= n) { j = 2 * (n - 1) - j; }
  return j;
}`;

/** Fixed-point Gaussian on uint8 (weights with 8 fraction bits, `kernel` = the half-open list of taps). */
export const gaussU8Rows = (taps) => {
  const r = (taps.length - 1) / 2;
  return `${P}${REFL}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
const K = array<u32, ${taps.length}>(${taps.map((t) => `${t}u`).join(", ")});
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x); let y = i32(g.y);
  if (x >= i32(p.w) || y >= i32(p.h)) { return; }
  let base = g.z * p.h * p.w + u32(y) * p.w;
  var s = 0u;
  for (var j = 0; j < ${taps.length}; j++) { s += K[j] * src[base + u32(refl(x + j - ${r}, i32(p.w)))]; }
  dst[base + u32(x)] = s;
}`;
};
export const gaussU8Cols = (taps) => {
  const r = (taps.length - 1) / 2;
  return `${P}${REFL}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
const K = array<u32, ${taps.length}>(${taps.map((t) => `${t}u`).join(", ")});
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x); let y = i32(g.y);
  if (x >= i32(p.w) || y >= i32(p.h)) { return; }
  let fb = g.z * p.h * p.w;
  var s = 0u;
  for (var j = 0; j < ${taps.length}; j++) { s += K[j] * src[fb + u32(refl(y + j - ${r}, i32(p.h))) * p.w + u32(x)]; }
  dst[fb + u32(y) * p.w + u32(x)] = min((s + 32768u) >> 16u, 255u);
}`;
};

/** `cv2.resize(.., 0.5, INTER_AREA)` on uint8, result as float32 (even h, w). */
export const areaDown2 = `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let hh = p.h / 2u; let ww = p.w / 2u;
  if (g.x >= ww || g.y >= hh) { return; }
  let fb = g.z * p.h * p.w;
  let r0 = fb + 2u * g.y * p.w + 2u * g.x;
  let s = src[r0] + src[r0 + 1u] + src[r0 + p.w] + src[r0 + p.w + 1u];
  dst[g.z * hh * ww + g.y * ww + g.x] = f32((s + 2u) >> 2u);
}`;

/** `cv2.GaussianBlur` float32 on the half-resolution image: rows as a sequential fma chain, `taps` (f32) from a buffer. */
export const gaussF32Rows = (n) => {
  const r = (n - 1) / 2;
  return `${P}${REFL}
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> k: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let hh = i32(p.h / 2u); let ww = i32(p.w / 2u);
  let x = i32(g.x); let y = i32(g.y);
  if (x >= ww || y >= hh) { return; }
  let base = g.z * u32(hh * ww) + u32(y * ww);
  var s = 0.0;
  for (var j = 0; j < ${n}; j++) { s = fma(src[base + u32(refl(x + j - ${r}, ww))], k[j], s); }
  dst[base + u32(x)] = s;
}`;
};
export const gaussF32Cols = (n) => {
  const r = (n - 1) / 2;
  return `${P}${REFL}
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> k: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let hh = i32(p.h / 2u); let ww = i32(p.w / 2u);
  let x = i32(g.x); let y = i32(g.y);
  if (x >= ww || y >= hh) { return; }
  let fb = g.z * u32(hh * ww);
  var o = src[fb + u32(y * ww + x)] * k[${r}];
  for (var j = 1; j <= ${r}; j++) {
    let up = src[fb + u32(refl(y - j, hh) * ww + x)];
    let dn = src[fb + u32(refl(y + j, hh) * ww + x)];
    o = fma(up + dn, k[${r} + j], o);
  }
  dst[fb + u32(y * ww + x)] = o;
}`;
};

/** `cv2.resize(.., 2x, INTER_LINEAR)` float32, KleidiCV's resize_2x2_f32 (NEON body vs scalar tail fuse differently), minus the result from `narrow`. */
export const upSub = `${P}
@group(0) @binding(1) var<storage, read> half: array<f32>;
@group(0) @binding(2) var<storage, read> narrow: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
const W0 = 0.5625; const W1 = 0.1875; const W3 = 0.0625;
fn l1v(near: f32, far: f32) -> f32 { return fma(far, 0.25, near * 0.75); }
fn l1s(near: f32, far: f32) -> f32 { return fma(near, 0.75, far * 0.25); }
fn l2v(a: f32, b: f32, c: f32, d: f32) -> f32 { return fma(d, W3, fma(c, W1, fma(b, W1, a * W0))); }
fn l2s(a: f32, b: f32, c: f32, d: f32) -> f32 { return fma(d, W3, fma(c, W1, fma(a, W0, b * W1))); }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let w = p.w; let h = p.h;  // output size; half is (h/2, w/2)
  let ww = w / 2u; let hh = h / 2u;
  if (g.x >= w || g.y >= h) { return; }
  let hb = g.z * hh * ww;
  let ox = g.x; let oy = g.y;
  let vec_end = (ww - 1u) / 4u * 4u;
  var v = 0.0;
  if (oy == 0u || oy == h - 1u) {
    let row = hb + select(0u, (hh - 1u) * ww, oy != 0u);
    if (ox == 0u) { v = half[row]; }
    else if (ox == w - 1u) { v = half[row + ww - 1u]; }
    else {
      let x = (ox - 1u) / 2u;
      let first = ((ox - 1u) & 1u) == 0u;
      let a = half[row + x]; let b = half[row + x + 1u];
      if (x < vec_end) { v = select(l1v(b, a), l1v(a, b), first); } else { v = select(l1s(b, a), l1s(a, b), first); }
    }
  } else {
    let y = (oy - 1u) / 2u;
    let second = ((oy - 1u) & 1u) == 1u;   // d1 rows (2y + 2) vs d0 rows (2y + 1)
    let s0 = hb + y * ww; let s1 = s0 + ww;
    if (ox == 0u || ox == w - 1u) {
      let xe = select(0u, ww - 1u, ox != 0u);
      let a = half[s0 + xe]; let c = half[s1 + xe];
      v = select(l1s(a, c), l1s(c, a), second);
    } else {
      let x = (ox - 1u) / 2u;
      let right = ((ox - 1u) & 1u) == 1u;  // 2x + 2 vs 2x + 1
      let a = half[s0 + x]; let b = half[s0 + x + 1u]; let c = half[s1 + x]; let d = half[s1 + x + 1u];
      let vec = x < vec_end;
      if (!second && !right) { v = select(l2s(a, b, c, d), l2v(a, b, c, d), vec); }
      else if (!second && right) { v = select(l2s(b, a, d, c), l2v(b, a, d, c), vec); }
      else if (second && !right) { v = select(l2s(c, a, d, b), l2v(c, a, d, b), vec); }
      else { v = select(l2s(d, b, c, a), l2v(d, b, c, a), vec); }
    }
  }
  let i = g.z * h * w + oy * w + ox;
  dst[i] = f32(narrow[i]) - v;
}`;

/** The same warp on float32 frames (`shift_f32`, the registration image): no rounding, source and result float32. */
export const shiftF32 = shift
  .replace(
    "array<u32>;\n@group(0) @binding(2) var<storage, read_write> dst: array<u32>;",
    "array<f32>;\n@group(0) @binding(2) var<storage, read_write> dst: array<f32>;",
  )
  .replace("return f32(src[base + u32(yy) * p.w + u32(xx)]);", "return src[base + u32(yy) * p.w + u32(xx)];")
  .replace("u32(clamp(round(v), 0.0, 255.0))", "v");
