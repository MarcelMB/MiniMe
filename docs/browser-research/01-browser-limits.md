# 01 · What a browser tab can do (Nov 2026)

Scale check: one 600×600 uint8 frame is 360 KB. A 10 min session at 20 fps is 4.3 GB raw. A 2 h session at 30 fps is 78 GB raw, or 311 GB as float32. Nothing at this scale fits in a tab's memory, so a browser MiniMe has to work out-of-core.

## 1. Memory

| | Chrome/Edge | Firefox | Safari macOS | Safari iPad |
|---|---|---|---|---|
| wasm32 linear memory | 4 GiB ([kV8MaxWasmMemoryPages](https://textslashplain.com/2020/09/15/browser-memory-limits/)) | 4 GiB | 4 GiB | about 2 GB in practice (Gigacage, then jetsam) |
| Memory64 | 133 | 134 | flag only (STP 251) ([features.json](https://github.com/WebAssembly/website/blob/main/features.json), [web-features](https://web-platform-dx.github.io/web-features-explorer/features/wasm-memory64)) | no |
| Memory64 cap | 16 GiB in the JS API ([V8 wasm-limits.h](https://chromium.googlesource.com/v8/v8/+/master/src/wasm/wasm-limits.h)) | 16 GiB | n/a | n/a |
| ArrayBuffer max (64-bit) | very large (2^53-1 per V8 devs) | 8 GiB ([MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Errors/Invalid_array_length)) | Gigacage-bound | about 1.9 GB seen on an 8 GB iPad ([WebKit 268816](https://bugs.webkit.org/show_bug.cgi?id=268816)) |
| Renderer process limit | up to 1 TB on Windows since 2023 | system RAM | OS pressure | about 1.5 GB on iPhone, higher on iPad ([WebKit 227636](https://bugs.webkit.org/show_bug.cgi?id=227636)) |

- **What happens on OOM.** A failed `Memory.grow` or allocation throws a `RangeError`, which the page can catch. Real exhaustion kills the renderer, which shows as "Aw, Snap" in Chrome and as a silent reload under iOS jetsam. All state is lost.
- **Memory64 costs speed.** SpiderMonkey measured 10–100 %+ slowdowns because bounds checks can no longer be elided ([blog](https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html)). Safari does not ship it. Do not rely on it.

## 2. Reading huge files and writing results

- **`File`/`Blob.slice(a,b).arrayBuffer()`** works in every browser. Reads are lazy and come from disk, so random access into a 50 GB AVI is fine: read the AVI `idx1` / OpenDML index chunk, then slice each frame by its offset. This is the only portable path.
- **Folder input.** `<input webkitdirectory>` and drag-and-drop folders via `webkitGetAsEntry()` work in all browsers.
- **File System Access pickers** (`showOpenFilePicker`, `showDirectoryPicker`, handles that survive a reload, writing next to the raw data) exist only in Chrome and Edge 86+. Firefox's position is negative and Safari opposes them ([web-features](https://web-platform-dx.github.io/web-features-explorer/features/file-system-access)). On Safari and Firefox the user has to pick the files again after every reload.
- **OPFS plus `createSyncAccessHandle`** has been Baseline since March 2023. It works only in dedicated workers, gives synchronous read/write at an offset, and has `flush()` ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle/createSyncAccessHandle)).
- **Quota** ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)):
  - Chrome: 60 % of disk per origin.
  - Safari 17+: about 60 % per origin.
  - Firefox best-effort: min(10 % of disk, 10 GiB).
  - Firefox persistent: 50 % of disk, after a prompt.
  - Eviction is LRU and removes all of an origin's storage at once.
  - Safari deletes script-written data after 7 days without user interaction.
  - `navigator.storage.persist()` is granted silently by Chrome and Safari and prompts in Firefox. Use `navigator.storage.estimate()` before writing.

## 3. Decoding video

| Codec | WebCodecs | Practical path |
|---|---|---|
| Raw / uncompressed GREY (typical for UCLA Miniscope DAQ AVI) | n/a | No decoding needed. Parse RIFF in JS and slice the bytes. This is the easy case. |
| MJPEG | not a registered `VideoDecoder` codec | `ImageDecoder` / `createImageBitmap` per frame (the result is RGBA, so take one channel), or libjpeg-turbo in wasm |
| FFV1 | no | ffmpeg.wasm or a custom wasm decoder |

- **WebCodecs support:** Chrome 94, Firefox 130 (desktop), Safari 26 ([web-features](https://web-platform-dx.github.io/web-features-explorer/features/webcodecs/)). It covers only H.264, HEVC, VP8/9 and AV1, so it does not help with miniscope codecs.
- **ffmpeg.wasm:** inputs are capped at 2 GB because the whole file sits in MEMFS. The project's own FAQ says it is "a lot slower than native", and the multithreaded build is about 2× faster but unstable ([FAQ](https://ffmpegwasm.netlify.app/docs/faq), [issue](https://github.com/ffmpegwasm/ffmpeg.wasm/issues/623)). Use it only as a fallback, decoding slice by slice.

## 4. Compute

- **wasm SIMD (128-bit):** Chrome 91, Firefox 89, Safari 16.4. Relaxed SIMD: Chrome 114 and Firefox 145; Safari has it only behind a flag.
- **Threads** (Chrome 74, Firefox 79, Safari 14.1) require `SharedArrayBuffer`, which only works on a cross-origin-isolated page. The page must be served with `COOP: same-origin` and `COEP: require-corp`.
  - GitHub Pages cannot set those headers ([discussion #13309](https://github.com/orgs/community/discussions/13309)). The workaround is the `coi-serviceworker` shim, which reloads the page on first visit and can clash with other service workers. Netlify, Cloudflare Pages and a local server can set the headers properly.
- **Workers:** there is no hard cap. Use about `navigator.hardwareConcurrency` workers. Isolated pages may coarsen or cap that number.
- **WebGPU:**
  - Chrome/Edge 113+ on Windows, macOS and ChromeOS; Linux support completed around Chrome 144.
  - Safari 26 on macOS, iOS and iPadOS.
  - Firefox 141 on Windows and 145 on Apple-silicon macOS; Linux and Intel Macs are still pending ([web.dev](https://web.dev/blog/webgpu-supported-major-browsers)).
- **WebGPU buffer limits:** the guaranteed defaults are `maxBufferSize` 256 MB and `maxStorageBufferBindingSize` 128 MB ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits)). Adapters report tiered values of 128 MB, 1 GB, 2 GB−4 or 4 GB−4 ([Dawn](https://git.axiodl.com/encounter/dawn-cmake/commit/960918b1b64f96720c2473f4133ed8ffabdfb176)). Intel/ARM iGPUs usually report 2 GB−4 ([web3dsurvey](https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize/platform/Chromium%20OS)) and Apple M-series in Chrome reports 4 GB−4 ([gist](https://gist.github.com/beaufortfrancois/8ad810913c48b636f7b44ca82ed8c4ef)). Requesting limits above the default must be done explicitly in `requestDevice`.

## 5. Long jobs (hours)

- **Throttling in Chrome:**
  - Timers on hidden pages fire at most once a second. After 5 minutes hidden, chained timers fire once a minute (intensive throttling) ([Chrome 88](https://developer.chrome.com/blog/timer-throttling-in-chrome-88)).
  - Work running inside dedicated workers is not timer-throttled in practice, so keep the compute loop in workers.
  - Memory Saver can still discard a background tab, which kills it. The user can allow-list the site ([tab discarding](https://developer.chrome.com/blog/tab-discarding/)).
- **iPadOS/iOS:** Safari suspends all JavaScript, workers included, once the tab or app leaves the foreground ([Firtman](https://frontendmasters.com/courses/background-javascript/observing-mobile-background-javascript)). A multi-hour job on an iPad only survives with the screen on and the tab in front.
- **Sleep:** Screen Wake Lock is available in Chrome 84, Firefox 126, Safari 16.4 and iOS 18.4 ([web-features](https://web-platform-dx.github.io/web-features-explorer/features/screen-wake-lock/)). It needs a visible tab and is released when the tab is hidden. There is no API to prevent lid-close sleep or CPU throttling on battery.
- **Durability:** OPFS `flush()` is the only durability primitive and does not guarantee an fsync. A tab crash or eviction can lose everything. Checkpoint every stage to OPFS and design the pipeline so it can resume.

## 6. Hybrid (already exists)

`src/minime/serve` is a FastAPI app on 127.0.0.1 with a random token and a Host-header guard against DNS rebinding. It has job routes with SSE events (`/jobs/{id}/events`) and on-demand frame, trace, seed and unit endpoints, plus a built web front end in `static/`. The browser handles only the UI and Python does the heavy work, so none of the limits above apply.

If the UI is ever hosted on a public HTTPS origin and talks to localhost, Chrome 142+ shows a Local Network Access permission prompt ([summary](https://www.dynamsoft.com/web-twain/docs/faq/chromium-142-local-network-access-issue.md)). Serving the UI from the local server avoids this, and that is how `minime serve` already works.

## Implications for MiniMe

- **Memory:** assume a working set of 2 GB or less per tab (1.5 GB on iPad) and 4 GiB or less per wasm instance. Do not count on Memory64 because Safari lacks it and it is slower.
- **Reading:** stream with `File.slice` by frame offsets from the AVI index. Raw GREY AVIs need no decoder. MJPEG goes through `ImageDecoder` per frame. Never load the whole file.
- **Storage:** write the tile store, motion fields and results to OPFS through sync access handles in workers. Call `persist()`, check `estimate()`, and expect Firefox to allow only 10 GiB unless persistence is granted. A 2 h float32 tile store (about 50 GB) is realistic only in Chrome or Safari on a large disk.
- **Compute:**
  - Run wasm SIMD plus threads in a worker pool. That requires COOP/COEP headers, so host on Netlify or Cloudflare (not plain GitHub Pages) or use the coi shim.
  - Use WebGPU for per-frame kernels such as filtering and registration in chunks of 128 MB or less unless the adapter reports a higher limit.
- **Long runs:** keep the compute loop in workers, hold a wake lock, and checkpoint every stage to OPFS so a run can resume. Treat iPad as preview-only, not for 2 h sessions.
- **Realistic split:** a browser-only build is plausible for sessions under about 10–20 min and for preview and parameter tuning. Full 2 h runs should keep using `minime serve`.
