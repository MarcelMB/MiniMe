struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;

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
}
