struct P { h: u32, w: u32, n: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;

fn word_id(g: vec3u, nw: vec3u) -> u32 { return g.y * nw.x * 256u + g.x; }

@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = word_id(g, nw);
  if (i >= (p.total + 3u) / 4u) { return; }
  let v = src[i];
  dst[i] = vec4<u32>(v & 255u, (v >> 8u) & 255u, (v >> 16u) & 255u, v >> 24u);
}
