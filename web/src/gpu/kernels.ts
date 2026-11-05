// WGSL for the per-frame preprocessing, each kernel the same bits as `rust/minime-core`'s CPU code:
// median (`preprocess::median_blur`, replicated border), the warp (`shift` / `shift_f32`: bilinear
// lerps as single-rounding fma, round half to even, zeros outside), the top-hat (`tophat`: running
// min / max with the border ignored), the band-pass registration image (`motion::bandpass`: the
// fixed-point Gaussian, 2x2 area mean, float32 Gaussian on the half image, KleidiCV's 2x upsampling)
// and the 2x2 area mean of a float32 image (`filters::area_down2_f32`, `Correlator::downsample`).
//
// The integer kernels are exact by construction; the float ones need WGSL's `fma` fused and no
// reassociation, which the self-test (`selftest.ts`) checks on the session's frames before a run uses
// them (checked byte for byte on Dawn/Metal, WebKit/Metal, SwiftShader and wgpu/Metal).
//
// A batch of `n` frames of h x w is uploaded as packed bytes (u32 words); intermediates hold one u32 or
// f32 per pixel; results are packed again (u8) or float32.

/** `gpuplan::Kernels` (Rust), as the stage's engine gives it (`gpu_kernels()`). */
export type Kernels = { H: number; W: number; median: number; tophat: number; u8taps: number[]; f32taps: number[] };

const P = /* wgsl */ `
struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;
`;
// one thread per packed word (4 pixels), a 2-D dispatch below the 65535 workgroups per dimension
const WORD_ID = /* wgsl */ `
fn word_id(g: vec3u, nw: vec3u) -> u32 { return g.y * nw.x * 256u + g.x; }
`;
const REFL = /* wgsl */ `
fn refl(i: i32, n: i32) -> i32 {
  var j = i;
  if (j < 0) { j = -j; }
  if (j >= n) { j = 2 * (n - 1) - j; }
  return j;
}`;

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

/** White top-hat result: `orig - opened` (saturating), packed. */
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
export function median(k: number): string {
  const r = (k - 1) / 2;
  const n = k * k;
  const tw = 16 + 2 * r;
  return `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, ${tw * tw}>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.z * p.h * p.w;
  let x0 = i32(wid.x * 16u);
  let y0 = i32(wid.y * 16u);
  let lin = lid.y * 16u + lid.x;
  for (var i = lin; i < ${tw * tw}u; i += 256u) {
    let sx = clamp(x0 + i32(i % ${tw}u) - ${r}, 0, i32(p.w) - 1);
    let sy = clamp(y0 + i32(i / ${tw}u) - ${r}, 0, i32(p.h) - 1);
    tile[i] = src[base + u32(sy) * p.w + u32(sx)];
  }
  workgroupBarrier();
  var v: array<u32, ${n}>;
  for (var j = 0u; j < ${k}u; j++) {
    for (var i = 0u; i < ${k}u; i++) { v[j * ${k}u + i] = tile[(lid.y + j) * ${tw}u + lid.x + i]; }
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
}

/** `cv2.warpAffine` pure shift, INTER_LINEAR as OpenCV 5 (fused float32 lerps), zeros outside; `u8`
 * rounds half to even into 0..255, else float32 in and out. */
function warp(u8: boolean): string {
  const t = u8 ? "u32" : "f32";
  return `${P}
@group(0) @binding(1) var<storage, read> src: array<${t}>;
@group(0) @binding(2) var<storage, read_write> dst: array<${t}>;
@group(0) @binding(3) var<storage, read> shifts: array<vec2<f32>>;
fn at(base: u32, yy: i32, xx: i32) -> f32 {
  if (yy >= 0 && yy < i32(p.h) && xx >= 0 && xx < i32(p.w)) { return f32(src[base + u32(yy) * p.w + u32(xx)]); }
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
  dst[base + u32(y) * p.w + u32(x)] = ${u8 ? "u32(clamp(round(v), 0.0, 255.0))" : "v"};
}`;
}
export const shiftU8 = warp(true);
export const shiftF32 = warp(false);

/** Running min / max over `2r + 1` samples along rows (`cols`: columns), samples beyond the ends ignored. */
export function morph(r: number, isMin: boolean, cols: boolean): string {
  const pad = isMin ? "255u" : "0u";
  const op = isMin ? "min" : "max";
  const [tw, th] = cols ? [32, 8 + 2 * r] : [32 + 2 * r, 8];
  const load = cols
    ? `let sx = x0 + i32(i % 32u); let sy = y0 + i32(i / 32u) - ${r};
    if (sx < i32(p.w) && sy >= 0 && sy < i32(p.h)) { v = src[base + u32(sy) * p.w + u32(sx)]; }`
    : `let sx = x0 + i32(i % ${tw}u) - ${r}; let sy = y0 + i32(i / ${tw}u);
    if (sx >= 0 && sx < i32(p.w) && sy < i32(p.h)) { v = src[base + u32(sy) * p.w + u32(sx)]; }`;
  const read = cols ? `tile[(lid.y + k) * 32u + lid.x]` : `tile[lid.y * ${tw}u + lid.x + k]`;
  return `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, ${tw * th}>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.z * p.h * p.w;
  let x0 = i32(wid.x * 32u);
  let y0 = i32(wid.y * 8u);
  let lin = lid.y * 32u + lid.x;
  for (var i = lin; i < ${tw * th}u; i += 256u) {
    var v = ${pad};
    ${load}
    tile[i] = v;
  }
  workgroupBarrier();
  var acc = ${pad};
  for (var k = 0u; k < ${2 * r + 1}u; k++) { acc = ${op}(acc, ${read}); }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = acc; }
}`;
}

/** `filters::gauss_u8` rows (fixed-point taps, reflect-101 border): integer sums. */
export function gaussU8Rows(taps: number[]): string {
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
}
/** ... then columns, rounded and clamped to uint8. */
export function gaussU8Cols(taps: number[]): string {
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
}

/** `area_down2_u8` (rounded mean of 2x2) of the uint8 frame, as float32 (even h, w). */
export const areaDownU8 = `${P}
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let hh = p.h / 2u; let ww = p.w / 2u;
  if (g.x >= ww || g.y >= hh) { return; }
  let r0 = g.z * p.h * p.w + 2u * g.y * p.w + 2u * g.x;
  let s = src[r0] + src[r0 + 1u] + src[r0 + p.w] + src[r0 + p.w + 1u];
  dst[g.z * hh * ww + g.y * ww + g.x] = f32((s + 2u) >> 2u);
}`;

/** `area_down2_f32` = `((a + b) + (c + d)) * 0.25` of every 2x2 block (even h, w), in two passes: the
 * pair sums of each row (`areaRowsF32`, h x w/2), then the pairs of rows (`areaColsF32`). In one kernel
 * Metal's fast math reassociates the sum into `((a + b) + c) + d` (about one value in 1e5 then differs
 * by an ulp); across two dispatches it cannot. */
export const areaRowsF32 = `${P}
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ww = p.w / 2u;
  if (g.x >= ww || g.y >= p.h) { return; }
  let i = g.z * p.h * p.w + g.y * p.w + 2u * g.x;
  dst[g.z * p.h * ww + g.y * ww + g.x] = src[i] + src[i + 1u];
}`;
export const areaColsF32 = `${P}
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let hh = p.h / 2u; let ww = p.w / 2u;
  if (g.x >= ww || g.y >= hh) { return; }
  let i = g.z * p.h * ww + 2u * g.y * ww + g.x;
  dst[g.z * hh * ww + g.y * ww + g.x] = (src[i] + src[i + ww]) * 0.25;
}`;

/** `filters::gauss_f32` on the half image: rows as a sequential fma chain over the taps (`k`). */
export function gaussF32Rows(n: number): string {
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
}
/** ... then columns from the centre out, on the summed symmetric pairs. */
export function gaussF32Cols(n: number): string {
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
}

/** `resize_up2_f32` of the half image (KleidiCV: the NEON body and the scalar tail fuse differently),
 * subtracted from the narrow Gaussian: the band-pass registration image. */
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
  let w = p.w; let h = p.h;
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
    let second = ((oy - 1u) & 1u) == 1u;
    let s0 = hb + y * ww; let s1 = s0 + ww;
    if (ox == 0u || ox == w - 1u) {
      let xe = select(0u, ww - 1u, ox != 0u);
      let a = half[s0 + xe]; let c = half[s1 + xe];
      v = select(l1s(a, c), l1s(c, a), second);
    } else {
      let x = (ox - 1u) / 2u;
      let right = ((ox - 1u) & 1u) == 1u;
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
