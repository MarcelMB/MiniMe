# WebGPU / wgpu feasibility harness (throwaway)

Question: can the GPU take over MiniMe's per-frame preprocessing and the pass 2 GEMMs in the
browser, bit-exactly, and is it faster than CPU workers? Nothing here is wired into the app.

Inputs: real frames of `~/minian_testing_local/synth_v4_5min/raw`. Outputs: log lines (no files).

## Run

```sh
# 1. reference data (CPU results of rust/minime-core, written to web/e2e/.data/gpu, git-ignored)
cd rust/minime-core
cargo run --release --example gpu_ref -- ~/minian_testing_local/synth_v4_5min/raw ../../web/e2e/.data/gpu
cargo run --release --example make_tiles -- ~/minian_testing_local/synth_v4_5min/raw \
  ~/minian_testing_local/synth_v4_5min/processed/shifts_local.npy ../../web/e2e/.data/gpu/tiles
# 2. suites, each in a dedicated Worker of headless Chromium (port 8814, GPU_PORT to change)
cd experiments/webgpu
node probe.mjs                 # navigator.gpu per browser / launch mode, page and Worker
node run.mjs check             # median, shift, tophat, max projection vs CPU, byte for byte
node run.mjs bandpass          # registration image (float32) and shift_f32, bit for bit
node run.mjs bench             # speed: per kernel, per batch size, sequential vs pipelined
node run.mjs gemm tile=44      # pass 2 products on a real tile, CPU vs GPU, rSVD ranks
node run.mjs sanity            # can the check fail? does the shift need fma?
# wgpu crate -> wasm (WebGPU backend), then `node run.mjs wgpu`
node dump-wgsl.mjs             # kernels.js -> spike/src/kernels/*.wgsl
cd spike && RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --release --target web \
  --out-dir ../pkg_gpu --out-name gpu_spike -- --features gpu
cargo run --release --features gpu --example native -- ../../../web/e2e/.data/gpu   # native Metal
cargo run --release --features gpu --example p2_native -- ../../../web/e2e/.data/gpu
```

`GPU_CHANNEL=chromium` (default, new headless, real GPU) | `chrome` | `shell` (old headless shell,
SwiftShader software Vulkan) | `webkit`. Playwright's default headless shell has no Metal adapter.

## Files

- `kernels.js` WGSL: median 7x7, shift (u8 and f32), tophat 15 (4 separable passes), max projection, pass 2 GEMM,
  registration image (gauss u8, area down, gauss f32, resize up 2).
- `gpuprep.js` TypeScript driver (batches, slots for upload/readback overlap); `suite_*.js` the suites.
- `spike/` Rust crate: the same WGSL through `wgpu` (wasm WebGPU backend and native Metal) next to the CPU code.
