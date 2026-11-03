# 04 - How to build a browser MiniMe (tech stack), 2026-11-02

Reading and web research only, no measurements. Repo facts come from grep on `cross-platform-support`.

## What in MiniMe blocks a straight port

- **numba**: only `spatial_kernel.py` (the `prange` NNQP solver), imported only by `cnmf.py`.
  Pyodide has no numba or llvmlite, because a JIT can't run inside wasm.
- **cv2**: 16 modules. Pipeline ops are `medianBlur`, `GaussianBlur`, `morphologyEx` (tophat),
  `warpAffine`/`remap`, `phaseCorrelate`, `resize`, `dilate`/`erode`, `connectedComponents`,
  `findContours`. `VideoCapture` in `io.py` reads the AVIs. All are in opencv.js or Pyodide's
  OpenCV 4.11. Video decoding is the likely gap: I don't expect a videoio backend in wasm (not verified).
- **Process model**: pass 2 uses spawned workers and window 2 uses a memmapped tile store.
  Neither works in Pyodide, which has no working `threading` or `multiprocessing` and keeps its file system in RAM.
- **Already reusable**: `web/` (React 18, Vite, uPlot) is the `minime serve` front end and
  talks HTTP to FastAPI routes (`/runs/{rid}/frame/{t}`, `/traces/{unit}`, `/jobs`, ...).

## Comparison

| | Pyodide / PyScript | Rust -> wasm (+ WebGPU) | C++ -> Emscripten | JS/TS + WebGPU (WGSL) | Hybrid: browser UI + local `minime serve` |
|---|---|---|---|---|---|
| Reuse of MiniMe code | High (numpy, scipy 1.18, skimage 0.26, pandas 3, opencv 4.11 all ship) | Rewrite of the core, but the same crate can also become a PyO3 native module | opencv.js covers the pass-1 ops; Eigen for linear algebra | Rewrite; WGSL from the wgpu-py lane carries over almost verbatim | 100 %, it already exists |
| Threads | None (one Web Worker) | rayon via `wasm-bindgen-rayon` (needs COOP/COEP) | pthreads (needs COOP/COEP) | GPU plus Workers | Native, all cores |
| CPU speed vs native | ~3-5x slower single-thread BLAS in older benchmarks; no SIMD BLAS, no AMX | Near-native scalar code, SIMD128 (4 x f32), no Accelerate/AMX | Similar to Rust | GEMM about 1-1.7 TFLOP/s possible on M-class GPUs | Native |
| Memory ceiling | ~2 GB practical (wasm32) | 4 GB per wasm heap; Memory64 is Chrome-only and slow -> stream instead | Same | GPU buffers: 128 MB default binding, request `adapter.limits` | RAM / disk |
| Download | ~6 MB core + numpy/scipy/opencv/skimage, tens of MB total, a few s to start | ~1-3 MB wasm | ~10 MB with opencv.js | <1 MB | pip install |
| numba replacement | numpy-vectorised NNQP (slow) or a wasm side module | Port `_cd_pixel` to Rust (~150 lines) | Port to C++ | WGSL is unsuited to the per-pixel QP | n/a |
| Verdict | Good for a demo or notebook, poor for the 600 px x 36k+ frame sessions we actually record | **Best core for streaming stats and QP** | Viable, worse ergonomics | **Best for GEMM, local correlation, per-frame A^T y** | Best for long recordings and batch runs |

ML runtimes (onnxruntime-web, TF.js, transformers.js) do run GEMM and convolution on WebGPU, but
they are 10-25 MB, graph-shaped (no streaming accumulators) and have no QR or SVD. Skip them.

## Prior art worth borrowing

- **CaLab** (miniscope org): browser calcium deconvolution. Rust FISTA -> wasm-pack in Web Workers, the
  same crate as a Python module via PyO3/maturin, Vite + uPlot, GitHub Pages, "data never leaves the
  browser". This is nearly the target architecture; copy its repo layout and worker pool.
  https://github.com/miniscope/CaLab
- **Viv / Avivator**: client-side OME-TIFF and OME-Zarr viewer (deck.gl). Use it as a model for showing big
  images and tiled max/correlation maps. https://github.com/hms-dbmi/viv
- **ImageJ.JS** (CheerpJ) and **ImJoy** (Pyodide plugins) show the demand, and the cost of running a
  whole desktop stack in the browser. https://github.com/imjoy-team/ImJoy
- **ITK-Wasm**: C++ -> Emscripten/WASI pipeline tooling, if C++ is chosen. https://github.com/InsightSoftwareConsortium/ITK-Wasm
- **WebGPU GEMM**: Nussbaum's 1 TFLOP+ WGSL matmul walkthrough. https://zanussbaum.github.io/blog/2024/webgpu
- I found no browser neuron/ROI extraction from video, so MiniMe would be the first.
- **Readers**: raw/GREY AVI is a short RIFF parser over `File.slice` (no library needed). FFV1 is not in
  WebCodecs (https://www.w3.org/TR/webcodecs-codec-registry/), so it needs ffmpeg.wasm (~30 MB, slow) or
  libavcodec's FFV1 decoder alone built to wasm. TIFF: geotiff.js / UTIF.js. mp4box.js is irrelevant (no MP4).

## Deployment

- **WebGPU** is on by default in Chrome/Edge, Safari 26 and Firefox on Windows (141+). Firefox on Linux is still
  behind a flag and Chrome on Linux depends on the GPU, so a wasm-SIMD CPU fallback is required.
- **COOP/COEP**: GitHub Pages can't set headers. coi-serviceworker injects them, at the cost of one reload on the
  first visit, and it must be a same-origin file (https://github.com/gzuidhof/coi-serviceworker).
  Cloudflare/Netlify `_headers` is cleaner. A PWA service worker and the COI worker must be **one** worker:
  add the headers in the PWA worker.
- **Offline/PWA**: vite-plugin-pwa caches wasm and WGSL, so the app works with no network after the first load.
- **Privacy**: files are opened via `<input type=file webkitdirectory>` (all browsers) or `showOpenFilePicker`
  (Chromium) and read in chunks, and nothing is uploaded. This is the selling point (see CaLab's wording).
  Write results with the FS Access API (Chromium) or OPFS + download; OPFS is quota-limited and evictable.

## Hybrid vs pure browser

`minime serve` already binds only to loopback with a Host guard (`serve/app.py`) and serves the
built `web/` bundle. **Hybrid wins** for 2-24 h sessions (50 GB tile stores, 7 GB per tile in pass 2),
batch runs, auto-tune sweeps and reproducible Python outputs. **Pure browser wins** for zero install,
locked-down lab PCs and Chromebooks, conference demos, parameter preview, teaching, and the bounded-memory streaming
algorithm, whose state is only O(pixels + units). Design one front end with a `Backend` interface:
`HttpBackend` calls today's routes, and `WasmBackend` calls the Worker. Same React UI, two engines.

## Recommended first-prototype stack

TypeScript + Vite + the existing React/uPlot `web/` code. One **Rust crate** (`minime-stream`) built with
wasm-pack, `+simd128`, and wasm-bindgen-rayon for the CPU path; it holds the AVI parser, the streaming
per-pixel stats and the NNQP. A **WebGPU** module with WGSL ported from `docs/gpu-research/bench/wgpu_gemm.py`
for local correlation and per-frame traces. Hosted on GitHub Pages + coi-serviceworker, Cloudflare Pages
if headers matter. Later, the same crate as a PyO3 module lets native MiniMe check against it.

## Smallest feasibility prototype

One static page with three steps:

1. Pick a 30-min V4 raw AVI (600x600 u8, ~36k frames, ~13 GB).
2. Stream it with `File.slice` in a Worker and fold each frame into per-pixel mean, variance, max and
   4-neighbour cross products on WebGPU, then show the max projection and local-correlation image.
3. Load MiniMe's `A` (spatial footprints) from a finished run and extract traces by per-frame `A^T y`, plotted with uPlot.

Pass criteria: JS heap stays under 500 MB, throughput is at least 2x real time on the M1 Max and at least 1x on an Intel
iGPU laptop, and the correlation image and traces match MiniMe's to 1e-4 relative.
This tests the three risks together: file streaming, the WebGPU memory model, and speed. It doesn't depend on CNMF.
