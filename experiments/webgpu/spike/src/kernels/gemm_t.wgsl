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
      let kk = idx % 16u; let mm = idx / 16u;
      let m = m0 + mm; let k = k0 + kk;
      var v = 0.0;
      if (m < g.m && k < g.k) { let a = m * g.p + k; v = f32((raw[a >> 2u] >> ((a & 3u) * 8u)) & 255u) - mean[k]; }
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
}
