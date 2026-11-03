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
| pass 1: max projection, block maxima, quality, tile store (OPFS in the browser) | `pass1` | **done** | max proj, block maxima, every tile bit-exact (native and Chrome) |
| pass 2: mean, energy, noise, rSVD + rank cut | `compress` | **done** | same rank per tile, mean + energy exact, reconstruction ≤ 6e-5 (numpy's draws replayed; native and Chrome) |
| seeds, PNR + GMM threshold, opt-in `gmm_refine`, seed merge | `seeds`, `gmm`, `corr.seeds_merge` | **done** | same seeds, masks and merge pairs (native on Python's factors: PNR ≤ 1.1e-7 rel; Chrome on its own pass 2: ≤ 1.1e-4) |
| initA, initC, initial unit merge | `init` | **done** | same footprint supports and merge groups (native: traces identical; Chrome on its own pass 2: ≤ 5.7e-6 rel) |
| CNMF loop | `cnmf` | next | per stage on one window |
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

## Pass 1 result (2026-11-02): native and Chrome match Python
Pass 1 as a `fresh` window run does it (`ref_pass1.py`: the window's segments only, the
baseline sidecar's shifts and template, one worker), frames 0–3000 of the 30-min GRIN synth:

| | Rust native vs Python | Rust → wasm, headless Chrome vs Python |
|---|---|---|
| block edges, max projection, 11 block maxima | identical | identical |
| tile store (100 tiles, 1030 MB) | identical, CRC-32 per tile | identical, CRC-32 of each OPFS file read back |
| quality | ≤ 1.3e-5 (template FFT round trip: numpy float64, Rust float32) | ≤ 1.3e-5 |
| time, 1 thread, M1 Max | 54 s (18 ms/frame) | 73 s (41 fps); tile writes 2.9 s, read-back 3.6 s; 92 MB wasm heap |
| Python, 1 worker (OpenCV) | 15 s | |

- `pass1.rs`: `Pass1` takes one raw frame and a shift (or estimates it against the
  template: `--estimate`), keeps max projection and block maxima, and buffers the corrected
  frames per tile; the caller appends `take_tiles()` to the tile files every 100 frames.
- Browser: `Pass1Engine` in `wasm.rs`; the worker writes one OPFS file per tile
  (`pass1_tiles/tile_NNN.part00`, sync access handles) and runs pass 1 when the inputs
  folder has `template.npy`.
- With `--estimate` (shifts from Rust's own FFT, ≤ 4e-6 px off) 4 of 3.96 M block-max
  pixels and 17 of 100 tiles differ by one grey level. The check uses the given shifts.

Run it:
```bash
python experiments/streaming/ref_pass1.py --t0 0 --t1 3000
minime-core pass1 --raw <raw dir> --inputs <ref>/inputs --out <ref>/rust_native
python experiments/streaming/check_pass1.py <ref> <ref>/rust_native
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_pass1.py <ref> <ref>/rust_browser
```

## Pass 2 result (2026-11-02): native and Chrome match Python
Pass 2 as `pass2.pass2_worker` does it (seed = tile index, the session's parameters) on
pass 1's tile store of frames 0–3000. `ref_pass2.py` records every Gaussian draw of
numpy's rSVD; the Rust core replays them (`Recorded`), or draws its own (`Gaussian`,
xoshiro128++ with Box–Muller) when no `omega_NNN.npy` is given.

| | Rust native vs Python | Rust → wasm, headless Chrome vs Python |
|---|---|---|
| kept rank per tile (513 in all, 1–16), last rSVD rank (40 in every tile) | identical | identical |
| mean, energy | identical | identical |
| noise `sigma` (float32 FFT) | ≤ 1.1e-6 rel | ≤ 1.2e-6 rel |
| singular values | ≤ 1.4e-5 rel | ≤ 1.4e-5 rel |
| reconstruction `W V^T` (rel. Frobenius, per tile) | ≤ 5.6e-5, median 1.8e-6 | ≤ 5.5e-5, median 1.8e-6 |
| temporal subspace (1 − smallest principal cosine) | ≤ 4.3e-9 | ≤ 1.4e-5 relUB |
| time, 1 thread, M1 Max | 18.7 s (100 tiles) | 27.8 s (3.6 tiles/s), tile reads 0.4 s; 92 MB wasm heap (pass 1 + 2 in one run: 1:41) |
| Python, 1 thread (Accelerate sgemm) | 10.8 s | |

- `compress.rs`: `TileMatrix` (mean, energy, noise; products with `Y` as float32
  `matrixmultiply` GEMMs over 512-frame chunks), Householder QR in float64, the small SVD
  from the eigenvectors of `B Bᵀ` in float64, lag-1 autocorrelation cut, rank doubling.
- With Rust's own draws the kept rank differs by 1–2 in 19 of 100 tiles (511 vs 513):
  that is the rSVD's own seed sensitivity at one power iteration, not a port error, so the
  check replays numpy's draws.
- Browser: `Pass2Engine` in `wasm.rs`; when the inputs folder also has `pass2.json`, the
  worker runs pass 2 on pass 1's open OPFS tile files right after pass 1.
  `ref_pass2.py`'s `inputs/` holds pass 1's inputs too, so one run does both.

Run it:
```bash
python experiments/streaming/ref_pass2.py --t0 0 --t1 3000   # after ref_pass1.py
minime-core pass2 --tiles <pass1 ref>/rust_native/tiles --inputs <ref>/inputs --out <ref>/rust_native
python experiments/streaming/check_pass2.py <ref> <ref>/rust_native
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_pass2.py <ref> <ref>/rust_browser
python experiments/streaming/check_pass1.py <pass1 ref> <ref>/rust_browser
```

## Seeds result (2026-11-02): native and Chrome match Python
The seed stage as `pipeline.run` does it (`FactorCorr`, `seeds_init`, `pnr_refine` with the
`auto` GMM threshold, `seeds_merge`; `gmm_refine` is off in the session's parameters, so it
runs once more on the PNR survivors just to check its port) on the factors of frames
0–3000. `ref_seeds.py` builds those factors from pass 1's and pass 2's references; its
merged seeds equal the stored `t1_fresh` window run's (191). The native run reads Python's
factors (`fac_*.npy`); the browser run reads its own pass 1 and pass 2.

| | Rust native vs Python (Python's factors) | Rust → wasm, headless Chrome vs Python (own factors) |
|---|---|---|
| initial seeds + window counts (1716), PNR mask (804), merge pairs (1976), merge mask, merged seeds (191), opt-in GMM mask (145) | identical | identical |
| k-means++ draws (MT19937, seed 0) | identical | identical |
| PNR | ≤ 1.1e-7 rel | ≤ 1.1e-4 rel |
| GMM weights, means, variances (PNR and opt-in) | ≤ 6.6e-10 rel | ≤ 1.8e-6 rel |
| merge correlations | ≤ 2e-15 | ≤ 2.0e-6 |
| `FactorCorr.norm2` | ≤ 4.6e-15 rel | ≤ 7e-5 rel (123 constant pixels: ≤ 4e-13 of the median) |
| time, 1 thread, M1 Max | 0.38 s (local maxima 0.27 s) | 0.97 s (local maxima 0.70 s); 95 MB wasm heap; pass 1 + 2 + seeds in one run: 1:38 |
| Python, 1 thread | 0.29 s (cv2 morphology 0.22 s) | |

- `seeds.rs`: rolling windows (shared with pass 1's block edges), local maxima without
  morphology (disks are nested, so the union over kernels is "maximum of the radius-2 disk,
  and the contrast at the largest radius where it stays the maximum exceeds `diff`"; checked
  against `cv2.dilate`/`erode`, which ignore the border), 8-connected components with
  numpy's integer medians, PNR (`np.percentile` linear, float32 neighbours), the masks.
- `factors.rs`: the window's `Factors`; a trace is sequential float32 FMAs over the
  components plus the mean, which reproduces Accelerate's sgemm on 299 of 300 rows.
- `gmm.rs`: MT19937 as numpy's `RandomState(0)`, k-means++, Lloyd, EM, with numpy's sum
  orders (pairwise for 1-D sums, sequential over axis 0) and scipy's `logsumexp`.
- `corr.rs`: `FactorCorr` (brick-wall low-pass with `realfft`, Gram matrices, `norm2`,
  pair correlations) and `seeds_merge` (`brightest_parent` roots).
- Margins: the closest seed is 3.3e-3 in log-likelihood from the PNR decision, the closest
  merge pair 1.3e-3 from the 0.8 threshold, far above both runs' differences.
- Constant pixels (zero energy) have `norm2` = 0 up to rSVD rounding (Python 2.8e-11, Rust
  8e-27 for two of them), so `check_seeds.py` compares them against the median. A
  correlation with such a pixel is rounding over rounding in Python too; initA's windows
  should not depend on it.
- Browser: `SeedsEngine` in `wasm.rs` takes the finished `Pass1Engine` and `Pass2Engine`;
  the worker runs it after pass 2 when the inputs hold `seeds.json`. `ref_seeds.py`'s
  `inputs/` holds pass 1's and pass 2's inputs too, so one run does all three.

Run it:
```bash
python experiments/streaming/ref_seeds.py --t0 0 --t1 3000   # after ref_pass2.py
minime-core seeds --inputs <ref>/inputs --out <ref>/rust_native
python experiments/streaming/check_seeds.py <ref> <ref>/rust_native
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_seeds.py <ref> <ref>/rust_browser --own-factors
python experiments/streaming/check_pass2.py <pass2 ref> <ref>/rust_browser
python experiments/streaming/check_pass1.py <pass1 ref> <ref>/rust_browser
```

## Init result (2026-11-03): native and Chrome match Python
`pipeline.init_units` (`initA` at `inita_corr` 0.8 in a radius-10 disk, `initC` through the
normal equations, the initial `unit_merge` at `init_merge_corr` 0.8; `merge_min_dist` is 0)
on `ref_seeds.py`'s factors and its 191 merged seeds. The native run reads Python's factors
and seeds (`init_seeds.npy`); the browser run its own pass 1, pass 2 and seeds.

| | Rust native vs Python (Python's factors) | Rust → wasm, headless Chrome vs Python (own factors) |
|---|---|---|
| units after initA (191), footprint supports (36 286 pixels), overlapping pairs (13), merge groups, units after merge (190), merged supports | identical | identical |
| window correlations (the values initA thresholds) | ≤ 2.9e-15 | ≤ 2.5e-5 |
| footprint values (float32) | identical | ≤ 2.0e-5 |
| traces `C` after initC and after the merge (of each trace's largest value) | identical (float32) | ≤ 5.7e-6 |
| trace correlations of overlapping pairs | ≤ 4.4e-16 | ≤ 7.5e-7 |
| centroids | identical | ≤ 1.5e-5 px |
| `cond(A Aᵀ)` | 13.66, Cholesky | 13.66, Cholesky |
| time, 1 thread, M1 Max | 0.12 s (initA 0.012, initC 0.10, merge 0.009) | 0.15 s (initA 0.010, initC 0.13, merge 0.014); 95 MB wasm heap; pass 1 + 2 + seeds + init in one run: 1:42 |
| Python, 1 thread | 0.083 s (initA 0.018, initC 0.054, merge 0.011) | |

- Margins: the closest window pixel is 3.4e-6 from 0.8 (its Chrome diff 5.8e-8); 7 pixels
  lie within Chrome's largest window difference (2.5e-5) of the threshold and all stay on
  their side (one moved 2.0e-5 with a margin of 1.5e-5, away from 0.8). The closest
  overlapping pair is 3.1e-4 from the merge threshold (Chrome diff 2.9e-7). Native
  differences are ≤ 3e-15, far below both.
- Two constant (zero-energy) pixels fall inside seed windows: their correlation is rounding
  over rounding (`norm2` 2e-13 of the median in Python, 1e-28 in Rust), so Python gives
  -0.05 and 0.06, Chrome 0.11 and 0.23. `check_init.py` lists them apart and only requires
  them to stay below `inita_corr`.
- `init.rs`: `init_a` (disk in `np.mgrid` order, `FactorCorr::pair`), `a_ty` (scipy's
  per-tile `csc_matvecs` order in float32, then `V_bᵀ` as sequential FMAs, float64 sum),
  `gram` (shared pixels ascending), `init_c` (`linalg::NormalSolver`: Jacobi `cond`, then
  Cholesky or pinv as `_solve_normal`), `overlap_pairs`, `merge_labels` (union-find,
  numbered by first appearance), `group_mean_a`/`group_mean_c`, `centroids`.
  `corr::TracesCorr` is `traces_corr` pair by pair. initC's time is mostly the Jacobi
  eigensolve for `cond` (191³).
- Browser: `InitEngine` in `wasm.rs` runs on the finished `SeedsEngine` (its factors,
  `FactorCorr` and merged seeds); the worker runs it after the seeds when the inputs hold
  `init.json`. `ref_init.py`'s `inputs/` holds the earlier stages' inputs, so one run does
  all four.

Run it:
```bash
python experiments/streaming/ref_init.py --t0 0 --t1 3000   # after ref_seeds.py
minime-core init --inputs <ref>/inputs --out <ref>/rust_native
python experiments/streaming/check_init.py <ref> <ref>/rust_native
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_init.py <ref> <ref>/rust_browser --own-factors
python experiments/streaming/check_seeds.py <seeds ref> <ref>/rust_browser --own-factors
python experiments/streaming/check_pass2.py <pass2 ref> <ref>/rust_browser
python experiments/streaming/check_pass1.py <pass1 ref> <ref>/rust_browser
```

## 8 h scale run (2026-11-03): motion ok, stream pass fails at the end
First 8 h (576 files, 576 000 frames) of the real V4 24 h recording in headless Chrome, one
worker, M1 Max; footprints (503), mean and noise from the 3 h MiniMe run `cafast_3h_fix`.

| | motion | streaming pass |
|---|---|---|
| time | 3:08:48, 51 fps flat from start to end | 2:12 for all 576 000 frames, 72–73 fps flat |
| memory | 65 MB wasm heap | state 1273 MB (the (T, K) traces), wasm heap 2397 MB at the end |
| result | shifts of the first 3 h match the 3 h Python run within 5.0e-6 px (all 216 000 frames) | `unreachable` trap in `finish()`: writing `C.npy` (1.16 GB) needs a second copy, past wasm32's 4 GB |

- Nothing slows down over 8 h; the only thing that grows with T is the traces.
- The state also grew: the pass kept every segment's sums (13 MB each for 503 units at
  600×600, 96 segments) only to add them up at the end.
- Fixed since: the pass keeps the running segment's sums and the total of the finished ones
  (added in segment order, so the arithmetic is `stream.total`'s), and the worker writes the
  traces to `C.npy` ((K, T) float32) in OPFS every 4096 frames (`Engine::take_traces`,
  `traces_header`); the page gets that file as a `File`, not bytes. State is now 2 segments
  regardless of T (39 MB for the 365-unit synth). Checked on 3000 synth frames in three
  1000-frame segments: native and Chrome identical to `ref_dump.py` (C, f and every sum
  exact or ≤ 2e-15, same masks and footprints). Not rerun on 8 h yet.
- Harness: `web/e2e/.data/long/` (local, gitignored) saves outputs in 32 MB chunks; one
  base64 string of a > 512 MB output would exceed V8's string limit.

## Next
0. Rerun the 8 h and 24 h streaming pass with the traces in OPFS (8 h scale run above);
   machine-hours, so overnight.
1. Discovery port, stage by stage on one 3000-frame window: ~~motion~~, ~~pass 1~~, ~~pass 2~~,
   ~~seeds/PNR/GMM~~, ~~init~~ (done) → CNMF loop → stitch.
2. Speed: preprocessing in a worker pool (frames are independent), faster median
   (Perreault O(1) or a sorting-network median), then rayon + COOP/COEP.
3. Save outputs to a folder (`showDirectoryPicker` on Chromium, OPFS elsewhere) and
   checkpoint the per-segment sums so a run can resume.
4. WebGPU for traces and sums once the CPU path is complete.
