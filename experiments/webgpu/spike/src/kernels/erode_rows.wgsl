struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;

@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, 496>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.z * p.h * p.w;
  let x0 = i32(wid.x * 32u);
  let y0 = i32(wid.y * 8u);
  let lin = lid.y * 32u + lid.x;
  for (var i = lin; i < 496u; i += 256u) {
    let tx = i32(i % 62u);
    let ty = i32(i / 62u);
    let sx = x0 + tx - 15;
    let sy = y0 + ty;
    var v = 255u;
    if (sx >= 0 && sx < i32(p.w) && sy < i32(p.h)) { v = src[base + u32(sy) * p.w + u32(sx)]; }
    tile[i] = v;
  }
  workgroupBarrier();
  var acc = 255u;
  for (var k = 0u; k < 31u; k++) { acc = min(acc, tile[lid.y * 62u + lid.x + k]); }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = acc; }
}
