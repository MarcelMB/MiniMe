struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;

@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> tile: array<u32, 484>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let hw = p.h * p.w;
  let base = wid.z * hw;
  let x0 = i32(wid.x * 16u);
  let y0 = i32(wid.y * 16u);
  let lin = lid.y * 16u + lid.x;
  for (var i = lin; i < 484u; i += 256u) {
    let tx = i32(i % 22u);
    let ty = i32(i / 22u);
    let sx = clamp(x0 + tx - 3, 0, i32(p.w) - 1);
    let sy = clamp(y0 + ty - 3, 0, i32(p.h) - 1);
    tile[i] = src[base + u32(sy) * p.w + u32(sx)];
  }
  workgroupBarrier();
  var v: array<u32, 49>;
  for (var j = 0u; j < 7u; j++) {
    for (var i = 0u; i < 7u; i++) {
      v[j * 7u + i] = tile[(lid.y + j) * 22u + lid.x + i];
    }
  }
  var lo = 0u;
  var hi = 255u;
  for (var it = 0; it < 8; it++) {
    let mid = (lo + hi) >> 1u;
    var c = 0u;
    for (var q = 0u; q < 49u; q++) { c += select(0u, 1u, v[q] <= mid); }
    if (c > 24u) { hi = mid; } else { lo = mid + 1u; }
  }
  let x = x0 + i32(lid.x);
  let y = y0 + i32(lid.y);
  if (x < i32(p.w) && y < i32(p.h)) { dst[base + u32(y) * p.w + u32(x)] = lo; }
}
