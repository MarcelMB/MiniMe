# Browser MiniMe: agreed design (2026-11-02)

Goal: zero-install MiniMe in a browser tab, ≤ 2 GB RAM, any recording length. Python
(`src/minime` + `experiments/streaming/`) stays the reference implementation; every
browser stage is checked against it. Background and results: `docs/browser-options.md`.

## Decisions
- **Discovery is ported too, as the full window pipeline** (the `fresh` result, F1 0.74):
  a whole MiniMe run per 2.5-min window (pass 1 with motion, pass 2 compression, seeds +
  PNR/GMM, init, CNMF loop), then `stitch` (`brightest_parent`). No subsample shortcut.
- Then the streaming passes: per-frame traces with fixed footprints + per-segment sums,
  one footprint update from the sums (`spatial_from_stats`), merge.
- No deconvolution anywhere.
- Mean image and noise come from the browser's own pass, not the batch factors.

## Stack
- `rust/minime-core`: one Rust crate, built two ways:
  - native (`cargo run --release --bin minime-core`): the check harness, reads the same
    AVIs and `.npy`/`.npz` as Python;
  - wasm (`wasm-pack`, `--target web`) for a Web Worker.
- Front end: a `#/stream` page in the existing `web/` app (React + uPlot) and one worker.
  Input via `<input type=file multiple>`; frames read with `File.slice` from the AVI index.
- CPU path first, single thread, float64 where Python is float64. Later: rayon threads
  (COOP/COEP via coi-serviceworker), WebGPU (WGSL) for GEMM/local corr/traces, OPFS for
  the window tile store and checkpoints.

## Stages and status

| Stage | Python reference | Rust | Check |
|---|---|---|---|
| AVI reader (RIFF, `idx1`, chunk scan fallback) | `io.read_avi` | **done** | bit-exact frames |
| median 7, warp (shift), tophat 15 (uint8) | `preprocess`, `motion.apply_shifts` | **done** | bit-exact |
| per-frame LS traces on `[A; b]`, sums Σy·cᵀ, Σc·cᵀ, Σl·lᵀ, Σy, Σdy² | `stream.TraceModel`, `stream_pass` | **done** | rel 1e-9 |
| footprint update from sums (NNQP), size filter, merge | `spatial_from_stats`, `merge_from_stats` | **done** | rel 1e-6, same kept units |
| motion: bandpass registration image, template, FFT estimate, quality | `motion`, `pass1.build_template` | **done** | reg + prep bit-exact, shifts ≤ 5e-6 px (native and Chrome) |
| pass 1 summaries (block maxima), tile store | `pass1` | next | bit-exact |
| pass 2 rSVD + rank cut | `compress` | next | same rank; subspace / reconstruction (rSVD RNG injected for the check) |
| seeds, PNR, GMM, initA/initC, CNMF loop | `seeds`, `corr`, `init`, `cnmf` | next | per stage on one window |
| stitch | `stream.stitch` | next | identical units |

Slice 1 inputs (until discovery is ported): footprints, shifts, mean image and noise come
from Python as a small export (`ref_dump.py`'s `inputs/`).

## Check plan (slice 1)
- Session: `/Volumes/Extreme SSD/synth_v4_grin30` (30 min, 600², 36 GREY AVIs, truth).
- Reference: `experiments/streaming/ref_dump.py` runs the prototype's pass + update and
  saves traces, sums and footprints to `processed/stream/ref_<name>/`.
- Native Rust on the same inputs → compare; then the wasm build in headless Chrome
  (Playwright) on the same files → compare.

## Slice 1 result (2026-11-02): matches Python exactly
30-min GRIN synth, stitched `fresh` footprints (365 units), 36 000 frames, M1 Max:

| | Python prototype | Rust native | Rust → wasm, headless Chrome 156 |
|---|---|---|---|
| frames, traces `C`, `f`, footprints | reference | identical (float32) | identical (float32) |
| sums (L, M, Mlp, ...) | reference | ≤ 2.3e-15 rel, Σy and Σdy² exact | ≤ 2.3e-15 rel |
| units kept / after merge | 319 / 315 | 319 / 315 | 319 / 315 |
| one pass | 112 s (OpenCV threads) | 369 s, 1 thread (prep 321 s) | 471 s (76 fps), 1 worker |
| memory | — | 357 MB peak RSS | 140 MB wasm heap, state 80 MB |

- OpenCV 5's `warpAffine` on arm64 uses fused multiply-adds (NEON). The Rust port does the
  same, so "bit-exact" refers to Python on Apple silicon; x86 OpenCV may round a few
  pixels differently.
- The 7×7 median (Huang histogram, ~7.5 ms/frame) is 2/3 of the time.

Run it:
```bash
npm --prefix web run wasm   # build the wasm core into web/src/stream/pkg
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_rust.py <ref> <ref>/rust_browser
```

## Motion stage result (2026-11-02): native and Chrome match Python
In the `fresh` runs every window reuses the session's motion sidecar, so the browser needs
the session's own motion pass: template from the first `template_frames` frames of
segment 0, then a shift per frame. Checked on frames 0–3000 of the 30-min GRIN synth
(`ref_motion.py`, which also reproduces the baseline sidecar's shifts exactly):

| | Rust native vs Python | Rust → wasm, headless Chrome vs Python |
|---|---|---|
| registration image (bandpass), prepared image (down 2, mean, Hann) | identical (float32) | identical (float32) |
| template image | ≤ 2.7e-7 of max | ≤ 3.2e-7 of max |
| shifts (up to 8.3 px) | ≤ 4.6e-6 px, same integer peak in all 3000 frames | ≤ 4.1e-6 px, same integer peak |
| quality | ≤ 1.3e-5 (numpy's float32 dot products; Rust is 50x closer to float64) | ≤ 1.3e-5 |
| time, 1 thread, M1 Max | 52 s: registration 12.9 ms/frame (median 7.5), estimate 0.9, quality 1.3; template 5.6 s | 67 s (50 fps incl. the 400 template reads), template 7.4 s; 29 MB wasm heap |

- OpenCV 5 on arm64 routes `resize` INTER_LINEAR through Arm's KleidiCV HAL: its NEON
  body and scalar tail fuse the multiply-adds in different orders; `filters.rs` mirrors
  both. The uint8 Gaussian is OpenCV's fixed-point path, the float32 Gaussian its
  fused SIMD order. As with the warp, "bit-exact" means against Python on Apple silicon.
- The FFT is `rustfft`/`realfft` in float32; numpy's float32 pocketfft is not reproduced.
- The template is built from two reads of its frames, so only one frame is held at a time.
- Browser: `MotionEngine` in `wasm.rs`; the `#/stream` worker runs it when the inputs
  folder has no footprints (`ref_motion.py`'s `inputs/`: `params.json`,
  `check_frames.npy`). Native and wasm FFTs differ by ≤ 5e-6 px in the shifts.

Run it:
```bash
python experiments/streaming/ref_motion.py --t0 0 --t1 3000
minime-core motion --raw <raw dir> --params <ref>/inputs/params.json --out <ref>/rust_native --check-frames <ref>/ref/frames_t.npy
python experiments/streaming/check_motion.py <ref> <ref>/rust_native
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_motion.py <ref> <ref>/rust_browser
```

## Next
1. Discovery port, stage by stage on one 3000-frame window: ~~motion~~ (done) → pass 1 (block maxima, tile store in OPFS) → pass 2 rSVD →
   seeds/PNR/GMM → init → CNMF loop → stitch.
2. Speed: preprocessing in a worker pool (frames are independent), faster median
   (Perreault O(1) or a sorting-network median), then rayon + COOP/COEP.
3. Save outputs to a folder (`showDirectoryPicker` on Chromium, OPFS elsewhere) and
   checkpoint the per-segment sums so a run can resume.
4. WebGPU for traces and sums once the CPU path is complete.
