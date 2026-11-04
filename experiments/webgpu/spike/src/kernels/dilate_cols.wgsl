struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;

@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, 1216>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.z * p.h * p.w;
  let x0 = i32(wid.x * 32u);
  let y0 = i32(wid.y * 8u);
  let lin = lid.y * 32u + lid.x;
  for (var i = lin; i < 1216u; i += 256u) {
    let tx = i32(i % 32u);
    let ty = i32(i / 32u);
    let sx = x0 + tx;
    let sy = y0 + ty - 15;
    var v = 0u;
    if (sx < i32(p.w) && sy >= 0 && sy < i32(p.h)) { v = src[base + u32(sy) * p.w + u32(sx)]; }
    tile[i] = v;
  }
  workgroupBarrier();
  var acc = 0u;
  for (var k = 0u; k < 31u; k++) { acc = max(acc, tile[(lid.y + k) * 32u + lid.x]); }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = acc; }
}
