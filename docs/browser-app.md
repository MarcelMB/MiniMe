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
- Front end: the existing `web/` app (React + uPlot) built with `vite build --mode browser`,
  an in-page back end (`web/src/browser/`, see "Browser back end" below) and one worker.
  Frames are read with `File.slice` from the AVI index. (The temporary `#/stream` page that
  first drove the worker is gone; New run does.)
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
| CNMF loop: spatial update (NNQP), size filter, background, traces, loop merge | `cnmf` | **done** | same supports, masks and merge groups per iteration (native: `YCt` exact; Chrome on its own factors: ≤ 2.2e-5 rel) |
| stitch (`brightest_parent` trees of cross-window footprint cosines), window loop | `stream.stitch`, `t1_discovery.py --mode fresh` | **done** | same edges, parents, groups; footprints bit-exact on the same window units (native and Chrome); 2-window loop in Chrome = Python |
| run from the AVIs alone: default parameters, own motion chained into the windows and the streaming pass, own rSVD draws, stage rows + `report.json`, downloads | `minime run` (`params.py` defaults, `report.json`) | **done** | same decisions as the folder-driven run with own motion (shifts ≤ 4.1e-6 px); own draws move 2 of 388 window units; 5-min synth F1 0.962 vs the batch run's 0.968 |
| browser back end: main's UI answered in the page (`web/src/browser/`): router, run folders on disk or in OPFS, jobs, derived views, zip download | `routes_*.py`, `runs.py`, `derived.py`, `jobs.py` | **done** | `browserFetch` + router vs main's answers on the Python e2e run and on a browser run (contract tests, same keys, values within tolerance); Playwright: New run → job → every page, download and read-only reopen, every page's screenshot within 2% of main's |

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
segment 0 (since 2026-11-05 frames spread over the session, see "Spread template and template
fill"), then a shift per frame. Checked on frames 0–3000 of the 30-min GRIN synth
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
- Browser: `MotionEngine` in `wasm.rs`; the stream worker runs it when the inputs
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
  (`pass1_tiles/tile_NNN.part00`, sync access handles; since the back end, in the run's own
  `scratch/<job id>/`) and runs pass 1 when the inputs folder has `template.npy`.
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

## CNMF result (2026-11-03): native and Chrome match Python
`cnmf.run_cnmf` (initial background, then `n_iter` 2 x spatial update on the radius-10
dilated supports with the NNQP kernel, `size_thres` (25, none), rows scaled to maximum 1,
background (unclipped, the session's `background_clip`), `compute_trace`, loop merge at
`loop_merge_corr` 0.9) on `ref_seeds.py`'s factors and init's 190 units. `ref_cnmf.py`
records every step (wrapping `dilate_footprints`, `solve_nnqp`, `update_spatial`,
`update_background`, `compute_trace`, `unit_merge`; the wrapped run equals the plain one
bit for bit). Its 189 final units equal the stored `t1_fresh` window run's. The native run
reads Python's factors and init units (`cnmf_A_*.npy`, `cnmf_C.npy`); the browser run its
own pass 1, pass 2, seeds and init.

| | Rust native vs Python (Python's factors) | Rust → wasm, headless Chrome vs Python (own factors) |
|---|---|---|
| per iteration: dilated supports (144 097 / 142 343 active pixels, 184 405 / 184 175 pairs), NNQP zero pattern, support size per unit, size mask, footprint supports (37 103 / 36 881 pixels), overlapping pairs (37 / 36), merge groups, merged supports; units 190 → 190 → 189 → 189 → 189 | identical | identical |
| `B` = `YCt` (float32) | iteration 0 identical, iteration 1 ≤ 8.0e-8 rel | ≤ 3.3e-6 rel |
| NNQP solution `w` | ≤ 9.0e-8 rel | ≤ 2.4e-6 rel |
| footprint values, `norm_fac` | ≤ 1.1e-6 rel | ≤ 2.2e-5 rel |
| traces `C`, `YrA` (of each trace's largest value) | ≤ 1.7e-6 | ≤ 1.9e-5 |
| background `b`, `f` | ≤ 1.1e-7 rel (initial `b` identical) | ≤ 3.3e-6 rel |
| merge correlations | ≤ 1.6e-9 | ≤ 1.2e-6 |
| `lam` (noise) | identical | ≤ 5.4e-7 rel |
| time, 1 thread, M1 Max | 0.32 s (spatial 0.13 per iteration, background ≤ 0.01, traces 0.007, merge 0.01) | 2.6 s (spatial 1.1 per iteration, background ≤ 0.1, traces 0.1, merge 0.015); 125 MB wasm heap; pass 1 + 2 + seeds + init + CNMF in one run: 1:44 |
| Python, 1 thread (numba NNQP, warm) | 1.09 s (of a spatial update: `cv2.dilate` per unit 0.33, NNQP 0.09) | |

- Margins: the closest NNQP decision is a non-zero of 2.2e-5 (in `B` units) in iteration 0,
  where Chrome's `B` differs by 2.4e-7 (native: 0); in iteration 1 the closest is 2.1e-3
  (a non-zero) against a largest `B` difference of 6.2e-5 (Chrome) / 1.5e-6 (native). No
  pair lies within the largest `B` difference of the boundary. Sizes: the closest unit is
  10 pixels from `size_thres`. Merge: the closest overlapping pair is 5.5e-3 from 0.9
  (Chrome diff 1.3e-7), then 1.4e-2.
- `cnmf.rs`: `dilate` (`stream::Support`), `update_spatial` (centring and unit norms with
  numpy's pairwise sums, `G` as syrk, `YCt` at the support pairs only: `V_b^T Ct` as
  sequential float32 FMAs over the frames, then `W_b` over the components, which is
  Accelerate's sgemm order, bit-exact natively; `nnqp::solve_nnqp`; zero elimination,
  size mask, row maxima), `update_background` (unclipped on the factors; clipped as
  Minian's, `BG_CHUNK` frames per tile with `f64_spot_check`), `compute_trace` (`A Y` in
  float64 through `a_ty64`, `A A^T` and the norms in scipy's orders), `unit_merge`
  (pairs with a positive float32 product, `TracesCorr`, `init::group_mean_a`, which now
  drops zero sums as scipy's `csr_matmat` does), the finite checks.
- The NNQP stops on a relative duality gap of 1e-10 and its `yn2` comes from a Jacobi
  eigensolve instead of LAPACK's, so `w` differs from Python by its convergence error even
  where `B` is identical (≤ 9e-8 here).
- `background_clip=True` (Minian's clipped background) is checked natively only
  (`ref_cnmf.py --clip`, `ref_cnmf_clip_0_3000`): same decisions, `b`, `f` ≤ 5.5e-8 rel,
  initial `b` identical, spot check 1.0e-6; 3.9 s native vs Python 2.6 s (threaded).
- Chrome is 8x slower than native in the spatial update, most likely because wasm simd128
  has no fused multiply-add: the `mul_add`s of `YCt` (needed for Accelerate's rounding)
  become `fmaf` calls (not profiled).
- Browser: `CnmfEngine` in `wasm.rs` runs on the finished `SeedsEngine` (factors) and
  `InitEngine` (merged units); the worker runs it after init when the inputs hold
  `cnmf.json`. `ref_cnmf.py`'s `inputs/` holds the earlier stages' inputs, so one run does
  all five.

Run it:
```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python experiments/streaming/ref_cnmf.py --t0 0 --t1 3000   # after ref_init.py
minime-core cnmf --inputs <ref>/inputs --out <ref>/rust_native
python experiments/streaming/check_cnmf.py <ref> <ref>/rust_native
STREAM_FILES=3 STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_cnmf.py <ref> <ref>/rust_browser --own-factors
python experiments/streaming/check_init.py <init ref> <ref>/rust_browser --own-factors
python experiments/streaming/check_seeds.py <seeds ref> <ref>/rust_browser --own-factors
python experiments/streaming/check_pass2.py <pass2 ref> <ref>/rust_browser
python experiments/streaming/check_pass1.py <pass1 ref> <ref>/rust_browser
```

## Pass 2 and CNMF speed (2026-11-04, branch `speed/gemm`)
Where pass 2 spent its 28 s in Chrome (100 tiles of 3000 frames, 1 thread, M1 Max, load 4-7;
`rust/minime-core/tools/wasm_pass2/` for wasm, `examples/bench_pass2.rs` native, both
`--features profile,reference`, old and new code alternating per tile): the two GEMMs
`Y Z` and `Y^T Q`, twice each, 12.3 s (44 %, with the float32 copy of the tile), the noise
10.9 s (39 %: 4096 FFTs of 3000 points per tile 3.8 s, a strided column gather 2.6 s, a libm
`ln` and `hypot` per bin 4.4 s), QR 3.4 s (12 %), small SVD 1.1 s, mean/energy 0.5 s.
`matrixmultiply` 0.3.11 *has* a wasm simd128 kernel (8 x 8) and runs at 38 GFLOP/s of the 51
an M1 core has without a fused multiply-add. A hand-written 8 x 5 / 12 x 4 kernel packing
straight from the uint8 tile was within +-7 % of it (bit-identical results), so it was
dropped. With `-C target-feature=+relaxed-simd` (fused `f32x4.relaxed_madd`; Chrome and Edge
114+, Firefox 146+, Safari unconfirmed) the same GEMMs run 1.26x faster; not enabled, it
needs a second wasm file for browsers without it and the sums stop being the same on every
machine.

What changed (`compress.rs`, `bfft.rs`, `fastmath.rs`): mean and energy from integer column
sums (exact: same bits); the noise of four pixels per vector in a Stockham FFT of the pixels'
bytes (`bfft.rs`; any `T` whose half is 5-smooth, `realfft` otherwise), the logarithms of the
bins as one vectorised Cephes `logf`, summed in order in float32 per pixel as numpy does; the
QR's trailing updates for all columns at once, eight columns in registers, in the order of
the column-by-column code (the same Q bit for bit); the small SVD's products with the
eigenvectors the same way (bit for bit).

| 100 tiles, 3000 frames, 1 thread | before | after | |
|---|---|---|---|
| pass 2, Chrome, in-process A/B | 28.3 s | 18.3 s | 1.54x |
| pass 2, Chrome, whole stage of the e2e run | 27.7 s | 17.7 s | 1.57x |
| pass 2, native, in-process A/B (`minime-core pass2`: 11.8 s) | 18.6 s | 11.6 s | 1.60x |
| noise / QR / small SVD / mean+energy, Chrome | 10.9 / 3.4 / 1.1 / 0.5 s | 3.3 / 1.9 / 0.6 / 0.2 s | 3.3 / 1.8 / 1.8 / 2.5x |
| GEMMs, Chrome and native | 12.3 s, 6.5 s | unchanged | 1.0x |

Checks on the browser run with numpy's draws replayed (`check_pass2.py`, `check_seeds.py`,
`check_init.py`, `check_cnmf.py`): the kept and last rank in 100 of 100 tiles, mean and energy
exact; `W`, `V`, `s`, ranks of the native run bit-identical to the previous native run;
`sigma` <= 1.27e-6 relative to numpy (was 1.1e-6 native, 1.2e-6 Chrome: another FFT, the
same sums); reconstruction <= 5.5e-5; seeds, masks, merge pairs, units identical, every float
within the old tolerances and as close as before (`B` 3.2e-6, footprints 2.2e-5, traces
1.8e-5, 189 final units). A constant pixel's `sigma` is a float32 sum of 1320 equal logarithms
and agrees with numpy only if summed in order: a faster sum through mantissa products was
1e-4 off numpy's on the 123 constant pixels of this session and was dropped.

CNMF in Chrome was 8x slower than native because `f64::mul_add` on wasm32 is a software
`fma`: 15.6 ns, against 0.9 ns for a multiply and an add. The Gram matrix `Cc Cc^T` alone is
54 M of them per iteration, 0.84 s of the 1.1 s of a spatial update (`f32::mul_add` is 2 ns
there and stays). `linalg::madd64` is fused natively (the native CNMF outputs are
bit-identical, 53 of 53 `.npy` files) and a multiply and an add on wasm32, and the Gram runs
four sums side by side. CNMF stage in Chrome 2.63 s -> 0.38 s (spatial 1.09 / 1.06 ->
0.16 / 0.14 s per iteration, traces 0.098 -> 0.018 s, background 0.06-0.10 -> 0.004-0.02 s).

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

## Stitch result (2026-11-03): native and Chrome match Python
(2026-11-05: the stitch also joins groups closer than `merge_min_dist` across windows,
`stitch.min_dist`; without it units grew with the session length. `docs/unit-inflation.md`.)
`stream.stitch` at `cos_thres` 0.7 (`t1_discovery.py`'s default, recorded in
`t1_fresh/summary.json`). `ref_stitch.py` redoes it step by step on the stored `fresh` window
units and reproduces the stored `stitched_A.npz` / `stitched.npz` exactly (12 windows,
2278 → 365 units, 8946 links).

| | Rust native vs Python | Rust → wasm, Chrome vs Python |
|---|---|---|
| (a) 12 stored windows: cross-window pairs (12 900), edges (8946), parents, roots, groups, n_windows, stitched supports | identical | identical |
| norms, cosines, stitched footprints (float32), brightness | identical (bit for bit) | identical (bit for bit) |
| time, 1 thread, M1 Max | 0.03 s | 0.07 s; 47 MB wasm heap |
| Python | 0.02 s | |

(b) Window loop in Chrome on frames 0–6000 (6 files, 2 windows; numpy's draws replayed in
pass 2 of both windows), against the stored `win_000000_003000` / `win_003000_006000` and a
Python stitch of those two:

| | browser vs Python |
|---|---|
| units per window (189, 199), their supports | identical |
| window footprint values / brightness (YrA peaks) | ≤ 2.2e-5 / ≤ 2.5e-6 rel |
| pairs (219), edges (144), parents, groups, n_windows, stitched supports (388 → 244 units) | identical |
| norms / cosines / stitched values / brightness | ≤ 3.6e-6 / ≤ 1.5e-6 / ≤ 2.0e-5 / ≤ 2.5e-6 rel |
| time, 1 worker | 3:26 (≈ 103 s per window: pass 1 72 s, pass 2 27 s, seeds + init + CNMF 4 s; stitch < 0.1 s); wasm heap 92 MB after window 1, 168 MB after window 2 (wasm memory never shrinks; whether it levels off over 12 windows is for the overnight run) |

- Margins: 12 windows: the closest cosine is 5.6e-4 from 0.7; the closest brightness tie
  between linked units is a relative gap of 2.9e-5 (units 978/1539). Two windows: 5.0e-2 and
  6.5e-4. Chrome's own-factor differences in the 2-window run are ≤ 1.5e-6 (cosine) and
  ≤ 2.5e-6 (brightness), so only the 12-window tie is within 10x of them: check it first in
  the overnight run.
- With the browser's own draws (no replay) pass 2's ranks differ in a few tiles (the rSVD's
  seed sensitivity, pass 2 result above): window 0 gives 191 units instead of 189 and the
  stitch 245 instead of 244. `minime-core stitch` on those browser windows equals the
  browser's stitch bit for bit.
- Sum orders follow scipy 1.18: `vstack(...).astype(float64)` canonicalises the stack (rows
  sorted); `diags @ A` and `M @ A` are `csr_matmat` (FMA into a zeroed accumulator, rows in
  reverse first-appearance order); `A.multiply(A)` keeps or reverses rows by its canonical
  check; a row sum is `np.add.reduceat` (first value + pairwise sum of the rest).
- `stitch.rs`: `stitch` (sorted stack, row max one, norms, cross-window cosines per row as
  `csr_matmat`, `corr::brightest_parent`, `roots`, group means, `n_windows`),
  `time_windows`, `window_inputs`; `minime-core stitch --inputs DIR --out DIR`.
- Browser: `StitchEngine` in `wasm.rs` (`add_cnmf` takes a finished `CnmfEngine`'s final A
  and YrA peaks; `add_inputs` reads `ref_stitch.py`'s `win_*`). The worker runs:
  - `stitch.json` in the inputs: the stitch of those window units (check a);
  - `session.json` in the inputs: the window loop. Per-window inputs come from **one
    session folder** (`ref_stitch.py --session-inputs`): `session.json` (window length,
    every stage's parameters without `T`/`t0`/`t1`, `stitch.cos_thres`), the session's
    `shifts.npy` (T, 2) and `template.npy`, optionally `omega_<t0>_<k>.npy` to replay
    numpy's draws. The worker indexes the AVIs, takes `time_windows(T, length, segment
    starts)`, and per window builds the stage files with `window_inputs` (Rust: JSONs with
    `T`/`t0`/`t1`, the shift slice), runs pass 1 → pass 2 → seeds → init → CNMF, hands the
    CNMF engine to `add_cnmf`, keeps only the stage summaries (`win_NN_summary_*.json`),
    frees the engines and deletes the OPFS tile store. Then it stitches and outputs
    `A_*` (stitched), `bright`, `n_windows`, the decisions and the window units in
    `ref_stitch.py`'s inputs layout (`stitch.json`, `win_*`), so Python can redo the stitch.
- File inputs skip symlinks, so the session folder holds copies of the omega files.

Run it:
```bash
python experiments/streaming/ref_stitch.py                          # 12 windows: inputs/ + ref/
minime-core stitch --inputs <ref>/inputs --out <ref>/rust_native
python experiments/streaming/check_stitch.py <ref> <ref>/rust_native
STREAM_FILES=1 STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_stitch.py <ref> <ref>/rust_browser
# window loop on the first 2 windows (after ref_pass1.py/ref_pass2.py --t0 3000 --t1 6000 for the replay)
python experiments/streaming/ref_stitch.py --n 2 --session-inputs
STREAM_FILES=6 STREAM_RAW=<raw dir> STREAM_INPUTS=<ref n2>/session STREAM_OUT=<ref n2>/rust_browser npm --prefix web run e2e:stream
python experiments/streaming/check_stitch.py <ref n2> <ref n2>/rust_browser --own-windows
```

## Mean image and noise from the windows (2026-11-03): decided and chained
The streaming pass after the stitch needs the session's mean image `ymean` (background of the
first pass) and noise `sn` (NNQP penalty of the update); until now they came from the batch
factors. Decided: both from the windows' pass-2 factors (`stream.window_mean_noise`), summed
in the window loop (`stitch::MeanNoise`, 2 × P float64):

- `ymean` = Σ T_w mean_w / T. Equal to the batch factors' up to rounding (rel ≤ 6e-8, 12 windows).
- `sn` = sqrt(mean_w σ_w²), the **quadratic** mean. The geometric mean (option "sn 1" as
  first listed) breaks on motion-border pixels that are empty in some windows: their σ_w ≈ 0
  pulls the log-mean to zero (16 291 pixels off by > 10 %, 1818 in footprint supports, ratio
  median 0.13 among them). The quadratic mean scales with the share of frames with signal, as
  the whole-trace FFT does: 12 pixels off by > 10 %, none in a support; in the supports the
  1st–99th percentile of window / batch is 0.977–1.012.
- F1 (`t2_stream.py --footprints stitched --n-updates 1 --mean-noise windows`): pass 0 0.748
  (same), after the update 0.738 vs 0.740 with the batch factors, 314 vs 315 units, trace
  correlation 0.770 both.

Browser: `StitchEngine.add_pass2` (worker hook `onPass2`) adds each window's pass-2 tiles;
`finish` adds `ymean.npy`, `sn.npy`. With a `stream` object in `session.json`
(`ref_stitch.py --stream`) the worker runs the streaming pass right after the stitch, on the
stitched footprints, the session's shifts and that mean and noise (`stream_params` builds its
`params.json`). Checked on frames 0–6000 in Chrome (2 windows, replayed draws, 4:48):

| | browser vs reference |
|---|---|
| `ymean` vs `window_mean_noise` of the stored Python windows | identical |
| `sn` | ≤ 1.1e-6 rel (pass 2's noise) |
| stitch (388 → 244 units, 144 links) | `check_stitch.py --own-windows` passes |
| streaming pass vs the native pass on the browser's own inputs (`check_chain.py`) | C, f, b, every sum, masks, 240 units: identical (sums ≤ 4e-16) |

```bash
python experiments/streaming/ref_stitch.py --n 2 --session-inputs --stream
STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/session STREAM_OUT=<ref>/rust_chain STREAM_FILES=6 npm --prefix web run e2e:stream
python experiments/streaming/check_stitch.py <ref> <ref>/rust_chain --own-windows
python experiments/streaming/check_chain.py <ref> <ref>/rust_chain <dir with the 6 AVIs>
```

## Trace gap result (2026-11-03): the background, not the denoising
Streamed traces scored 0.77 (stitched footprints) and 0.81 (batch footprints) against the
batch loop's 0.86. The hypothesis was the denoised movie: the batch traces come from the
pass-2 factors, the streaming pass fits raw frames. `t2_stream.py --project` (the
`t3_project.py` script of this test is removed) projects every frame's tiles onto a per-tile spatial basis before the per-frame least squares (`stream.TileBasis`,
`WindowProjector`), footprints fixed, 30-min GRIN synth, median trace correlation with
the truth (q25 in brackets):

| frames fitted | basis state | batch footprints (262) | stitched + 1 update (315) |
|---|---|---|---|
| raw (today, `b` = mean image minus its LS fit on `A`) | — | 0.808 (0.61) | 0.770 (0.54) |
| projected, session basis (the batch's own factors; needs all of T) | 11 MB, rank sum 675 | 0.806 (0.59) | 0.768 (0.54) |
| projected, each frame onto its own `fresh` window's basis | 9 MB per window, one at a time (rank sum 408–513) | 0.815 (0.63) | 0.779 (0.58) |
| projected, windows folded into one basis per tile (incremental PCA, cap 24) | 34 MB | 0.809 (0.62) | 0.772 (0.56) |
| raw, **`b = mean image`** (`--background mean`) | — | **0.923 (0.83)** | **0.910 (0.78)** |
| projected onto each window's basis, `b = mean image` | 9 MB per window | — | 0.913 (0.82) |
| batch loop (`YrA`) | | 0.855 (0.68) | |

- Projection cannot help a least-squares trace: a footprint that the basis spans has the
  same inner product with the projected frame as with the raw one (`X U Uᵀ y = X y`), so
  the noise along the footprint stays. The session basis reproduces the factors'
  reconstruction to 1e-5 and changes the score by 0.002. Cost, numpy: 0.9 ms/frame
  (2 GEMVs per tile, `2 P r` flops), 31–34 s of a 110-s pass.
- The gap is the background vector `b`. On the factors with the batch footprints: LS with
  the batch's own `b, f` 0.856 (= `YrA`), joint `[A; b_batch]` 0.854, joint `[A; b_stream]`
  0.806, joint `[A; mean image]` **0.923**. `b_stream = clip(ymean − Aᵀ cbar)` with `cbar`
  the LS fit of the mean image gives every unit its whole local mean, so `b` keeps only
  10 % of the mean image under the footprints (batch `b`: 58 %) and the background's
  fluctuations under a cell (neuropil, hemodynamics, drift) go into its trace. With the
  plain mean image as `b` the resting levels follow `f`. Rebuilding `b` from the clipped
  traces' mean, the batch recipe, drifts back toward the batch (0.910, 0.897, 0.889 over
  three rounds).
- Every category gains (factors, batch footprints, batch → `b = mean`): dim 0.68 → 0.83,
  small 0.86 → 0.94, near out-of-focus 0.82 → 0.91, overlap 0.89 → 0.91, late 1.00 → 0.99;
  after a 1-min high-pass still 0.84 → 0.90, so it is the fast background leak, not drift.
- Full chain (`t2_stream.py --footprints stitched --n-updates 1 --background mean`, both
  passes): traces **0.920**, but the update keeps 298 of 365 units instead of 319 (F1 0.714
  vs 0.740, recall 0.586 vs 0.623, 16 more true cells lost, mostly small): the cleaner
  traces have a smaller spread, so `sparse_snr` 1.0 prunes harder (nnz median 116 vs 130);
  footprint stability improves (late-segment cosine 0.98 vs 0.96). With `--sparse-snr 0.7`
  the chain keeps 339 units: recall 0.664, precision 0.903, **F1 0.765, traces 0.907**
  (`sparse_snr` 0.5: 346 units, F1 0.766, 0.907), better than today's 0.740 / 0.770 on
  both counts. Keeping the update on the `fit` pass's sums and `b = mean` only in the trace
  pass is the other option: F1 0.740, traces 0.910 (the stitched column above).
- Decision: `b = mean image` in every streaming pass, `sparse_snr` 0.7 for the streamed
  update (`t2_stitched_u1_lp_bmean_snr0.7`).
- Extra state and per-frame cost of `b = mean`: none (the same `[A; b]` solve; the `cbar`
  solve at build time goes away).
- Rust port: `TraceModel::build` takes `ymean` as `b` (no `cbar` solve); nothing else
  changes. The tile projection is not needed; if wanted later it is one `(P_b, r_b)` basis
  per tile in the streaming state and two GEMVs per tile per frame; on top of `b = mean`
  it adds 0.003 to the median and 0.03 to the lower quartile.
- Ported 2026-11-05 (until then the Rust core still carved `b`, so the app scored 0.75):
  `TraceModel::build` uses `b = clip(ymean, 0)`, and `stream.py`'s default is `"mean"` so
  `ref_dump.py` and `check_rust.py` check it (2000 frames: PASS, C 7e-13). App on the 30-min
  GRIN synth, Python auto-tune parameters (`params_tuned/grin.json`), 8 CPU workers, scored
  by `compare.py` against main's `minime run` with the same parameters: traces (C) 0.761 →
  **0.904** (q25 0.54 → 0.76; on the 244 truth cells both runs find 0.823 → 0.930, Python
  C 0.801, YrA 0.837); F1 0.773, recall 0.679, precision 0.897 and the 349 units unchanged
  (the background only changes traces). Trace agreement with Python's C on matched units
  drops from 0.994 to 0.907 by design: the two now fit different backgrounds.

## Run from the AVIs alone (2026-11-03): no Python involved
The worker runs the whole pipeline from the raw AVIs: pick the files, press Run (first on a
`#/stream` page, now New run; `e2e/stream.spec.ts` starts the same worker through the e2e hook).
The reference folder is optional and only overrides what it holds.

- Parameters: `web/src/stream/session.ts` builds `session.json` from MiniMe's defaults in
  `ref_stitch.py --session-inputs --stream`'s layout (3000-frame windows, `cos_thres` 0.7,
  `ref_dump.py`'s streaming layout); `session.test.ts` checks it equals what Python wrote for
  the 600² GRIN synth (`session.ref.json`), and the AVI-only run's `session.json` equals the
  reference's bit for bit.
- Motion: `runWindows` runs `MotionEngine` over every frame first (template from the first
  200 frames of segment 0) unless the inputs hold `shifts.npy` and `template.npy`; its
  float64 shifts go to the windows (`window_inputs`) and the streaming pass in memory, cast to
  float32 as the Python sidecar is (`npy::read_f32`). Pass 2 draws its own Gaussians
  (`Gaussian`, seed = tile index) unless `omega_*.npy` files are given.
- Page (the removed `#/stream` page; New run's job page shows the same rows):
  `minime serve`'s job-page layout (status + elapsed, one `StageRows` row per stage with the
  running stage's frames, fps and wasm heap live, a per-window stage adds up over the
  windows; log; results). The worker posts `progress.jsonl`-shaped stage events and
  writes `report.json` with MiniMe's field names where they exist (`stages`, `counts`,
  `motion.max_abs_shift_px` / `quality_median`, `pass1_cpu_seconds`, `wall_total`,
  `workers`) plus `windows` (per-window seconds and counts) and `wasm_heap_peak_mb`.
  Downloads: stitched `A_*`, `C.npy`, `units.csv` (centroid, size, windows, brightness,
  kept), `shifts.npy`, `Amerged_*`, `report.json`, `session.json`, then every other file.
- Checks on frames 0–6000 of the GRIN synth (6 files, 2 windows), one worker, M1 Max:

| run | inputs | window units → stitched | against the reference (`check_stitch.py --own-windows`, `check_chain.py`) |
|---|---|---|---|
| reference folder (`STREAM_INPUTS=<ref>/session`) | `session.json`, Python's shifts and template, numpy's draws | 388 → 244, 144 links | PASS as before: decisions identical, footprint values ≤ 2.0e-5, chain `C`, `f`, sums exact |
| own motion (`session.json` + `omega_*` only) | browser motion, numpy's draws | 388 → 244, 144 links | shifts ≤ 4.1e-6 px of Python's (same integer peak, all 6000 frames), template ≤ 3.2e-7; every decision identical (supports, pairs, edges, parents, groups); window footprint values ≤ 3.9e-3 (the one-grey-level pixels a 4e-6 px shift moves, pass 1 result above), brightness ≤ 1.1e-5, cosines ≤ 1.2e-5 |
| AVIs alone | nothing | 390 → 245, 145 links | own draws: window 1 keeps 2 more units (pass 2's seed sensitivity); `session.json` identical to Python's |

  Time: motion 128 s (47 fps incl. the template), pass 1 144 s (42 fps), pass 2 55 s,
  seeds + init + CNMF 8 s, stitch < 0.1 s, streaming pass 78 s (77 fps); 6:55 wall, wasm
  heap peak 177 MB. The folder-driven run without motion took 5:03.
- 5-minute synthetic session (`synth_v4_5min`, 9 AVIs, 9000 frames at 30 fps, 600²), AVIs
  alone, headless Chrome: 10:44 wall (motion 187 s, pass 1 217 s, pass 2 85 s, CNMF 25 s,
  streaming pass 127 s), heap peak 273 MB; 3 windows (409, 400, 396 units) stitched into
  419, 418 kept by the update, 417 after the merge; largest shift 12.1 px, quality median
  0.83. Scored against the truth (421 cells; then `browser_run_dir.py` laid the outputs out
  as a run dir; the run folder is now scored as it is, `minime score <run>`): F1 **0.962** (recall 0.960, precision 0.964, detectable
  recall 0.965), trace correlation median **0.995**. The Python batch run on the same
  session (`minime_2026-10-28_1717`): F1 0.968 (recall 0.948, precision 0.990), traces
  0.994. (The truth is regenerated from `meta.json`; the generator still reproduces the raw
  frames bit for bit.)

Run it:
```bash
npm --prefix web run wasm
STREAM_RAW=<raw dir> STREAM_OUT=<out> npm --prefix web run e2e:stream          # AVIs alone
STREAM_FILES=6 STREAM_RAW=<raw dir> STREAM_INPUTS=<ref>/session STREAM_OUT=<ref>/rust_browser_full npm --prefix web run e2e:stream
python experiments/streaming/check_stitch.py <ref> <ref>/rust_browser_full --own-windows
python experiments/streaming/check_chain.py <ref> <ref>/rust_browser_full <raw dir>
minime score <out>/run --raw <raw dir>      # <out>/run is the run folder the page wrote
```
`e2e:stream` builds and serves its own copy of the browser build (port 8796), so edits to
`web/src` after it started do not reach a running one; do not edit while it builds.

## Browser back end (2026-11-03): main's UI, answered in the page

`browser-app` is main's app with the same pages (Runs, New run, the job page, Overview to
Traces; no Tune) but every `/api/...` request is answered in the tab: `browserFetch` and a
router (`web/src/browser/`) over the run folder, the wasm worker for New run. Hosted as static
files (`npm run build:browser` → `web/dist-browser/`, `.wasm` as `application/wasm`).

- **Where a run goes**: `<raw parent>/processed/<name>` when the raw folder is inside a known
  folder (a picked folder or Browser storage), as `minime serve` does; else a picked output
  folder `<picked>/<name>` (Chrome, Edge: the dialog opens at Start); else Browser storage
  (OPFS `runs/<name>`, other browsers). The run folder is MiniMe's layout (`params.json`,
  `report.json`, `cnmf/`, `factors/`, ...), so `minime serve` and `minime score` open it; the
  window factors and the tile store are not Python's formats, so main shows no recon or kept
  preprocessed frames, noise image or motion residual of a browser run.
- **Folders**: picked folders are remembered (handles in IndexedDB). After a reload Chrome needs
  permission again: requests into the folder answer 403 "Reconnect the folder ...", shown as
  Home's error note; Change folder, then pick it again, and it is the same folder as before.
  A downloaded, unzipped run opens read-only from Change folder in every browser.
- **Browser storage**: a run there has "Download run (.zip)" (a stored zip64 of the run
  folder, streamed from OPFS; the button says "Preparing…" while the zip is put together).
  Before a run into it the quota is checked (`navigator.storage.estimate()` against twice the
  traces at an assumed 500 units, the window factors and, with Keep store, the tile store) and a
  run that does not fit is refused with the numbers; the browser is asked once to keep the
  storage (`persist()`). Once the stitch has found the units the worker checks again with the
  real K, before the streaming pass allocates its traces (for a run on disk: the one scratch
  copy of the traces).
- **Jobs and tabs**: a run holds the Web Lock `minime-run` from Start to its end, so one run
  goes at a time across the site's tabs; a second tab refuses Start ("a run is already going in
  another tab") and shows the first tab's job as running, not interrupted. A job is marked
  interrupted only when no tab holds the lock. Each run keeps its working files (the pass-1
  tile store, the traces before they reach the run folder) in OPFS `scratch/<job id>/`,
  removed when it ends, is cancelled or was interrupted. The worker writes `progress.jsonl` as
  the run goes; the job page of a job from before a reload replays it, as main does.
- **Keep the tab**: the run lives in its tab. Keep it open and preferably in view (the Wake
  Lock only holds while the tab is visible); Chrome's Memory Saver may discard a background
  tab, so for a run of hours add the site under Settings, Performance, "Always keep these
  sites active". A closed tab leaves the job interrupted; start it again under the same name.
- **A Python run in the browser**: opens with every page. Its `raw_dir` is an absolute disk
  path the page cannot open, so there are no raw frames and no frame rate from the raw folder;
  recon frames and the noise image are not available (404 "not available for this run"); the
  preprocessed frames and the residual come from Python's flat tile store and
  `factors/block_max.npy`.
- **Same look**: `e2e/look.spec.ts` shows the Python e2e run in `minime serve` and in the
  browser build and compares every page's screenshot (`maxDiffPixelRatio` 0.02). Identical
  by construction: pages, components and CSS are main's. Left out of both screenshots because
  they differ by design: absolute against virtual path texts (set to the same text), recent
  folders (the browser adds Browser storage), the Tune tab and auto-tune switch, the
  reconstruction frame and the noise image of a Python run. The look run's copy of the run has
  its `raw_dir` set to the browser's copy of the raw folder (as in the contract tests), so both
  show raw frames. At zero tolerance the pages measured pixel-identical (Home was off by 23
  pixels in one run), so the 2% is slack.

Run it (in `web/`; Python tools as `MINIME_PYTHON=<venv>/bin/python PYTHONPATH=../src`):
```bash
npm run wasm                  # after Rust changes (needs Rust and wasm-pack)
npm run dev:browser           # the browser build with hot reload
npm run fixtures              # the Python e2e run, main's answers on it, the params snapshot
npm run test                  # vitest: jsdom, then browser mode (the router and contract tests)
npm run build && npm run e2e  # main's smoke test against minime serve (port 8799)
npm run e2e:browser           # New run -> job -> every page, zip, folders (builds, port 8797)
npx playwright test -c playwright.look.config.ts   # same look as main (ports 8799 and 8797)
# the browser run folder the contract tests read, and main's answers on it
STREAM_KEEP_STORE=1 STREAM_RAW=e2e/.data/raw STREAM_OUT=e2e/.data/browser-run npm run e2e:stream
python scripts/contract_dump.py e2e/.data/browser-run/run e2e/.data/contract-browser
```
The look check and `npm run e2e` need main's build (`npm run build`) and start `minime serve`
themselves; run only one Playwright config at a time (they share ports and `dist-browser/`).

## Worker pool (2026-11-04): the frame-parallel stages on N workers, same outputs bit for bit
The run's worker (`stream.worker.ts`) keeps the stages that cannot be split (seeds, init, CNMF,
stitch, the footprint update) and hands the frame-parallel ones to a pool of frame workers
(`pool.ts`, `frame.worker.ts`, `poolStages.ts`: nested Web Workers, each its own wasm instance and
AVI index). No SharedArrayBuffer, so no COOP/COEP: a worker gets its inputs in a message and returns
its results in one; static hosting is unchanged. `params.workers` sets the count (a whole number
>= 1; `1` runs everything in the run's own worker as before); the default is the machine's cores less
two, at most 8. If the pool cannot start, the run goes on with one worker.

- **Motion** is folded into pass 1: the motion stage is now only the template (200 frames of segment
  0, its frames prepared on the pool, added in frame order); every window's pass 1 estimates its frames'
  shifts against that template (`Pass1::push(raw, None)`, one median shared), and the windows'
  shifts and quality are put together for the streaming pass and the run folder. Pass 1 rebuilds the
  stage's own `Motion` from `template_src.npy` (the image before its FFT round trip; `template.npy`
  would give shifts 1e-7 px off).
- **Pass 1**: a window is cut into chunks (about two per worker, whole multiples of 100 frames);
  a worker runs `Pass1Engine` from its chunk's first frame (`start_at`) and returns its max
  projection, block maxima, shifts and quality, which the window's engine folds in (`merge_chunk`:
  maxima are exact in any order). The tile store is one slab file per chunk (`slabs.ts`), written
  through one sync access handle: a handle per tile and chunk exceeds WebKit's ~250 open handles
  with 3 workers. Pass 2 reads the slabs through `File` snapshots, which need no exclusive handle.
- **Pass 2**: tiles are independent (seed = tile index): a worker compresses any tile
  (`compress_at`) and the window's engine takes the factors in tile order (`set_tile`).
- **Streaming pass**: a worker preprocesses and solves 16-frame chunks (`StreamSolver`, the same
  `prep().frame` and `TraceModel::solve`), the run's worker folds the solved frames in frame order
  (`push_solved`), so every sum, the low-pass state (it runs across segment boundaries), `dsq`
  (needs the previous frame across them) and the traces are the one-worker pass's. Splitting by
  6000-frame segment would not be bit-identical (and would leave 2 jobs for a 5-min session).
- **Overlap**: window w + 1's pass 1 is dealt to the workers when window w's pass 2 is done, so the pool
  works while the run's worker does that window's seeds, init and CNMF (the stage row counts from the
  deal).
- Measured, 3000 frames (one window) of `synth_v4_5min`, headless Chrome 156, M1 Max (8 P + 2 E cores,
  other agents' load average 4-6), seconds or rate per stage, `workers` 1 / 2 / 4 / 8:

  | stage | 1 | 2 | 4 | 8 | 8 vs 1 |
  |---|---|---|---|---|---|
  | template (200 frames x 2) | 7.6 s | 3.9 s | 2.0 s | 1.1 s | 6.9x |
  | pass 1 | 38 fps | 72 fps | 140 fps | 264 fps | 6.9x |
  | pass 2 | 3.5 tiles/s | 6.3 | 12.6 | 21.7 | 6.2x |
  | streaming pass | 72 fps | 136 | 258 | 436 | 6.1x |
  | seeds + init + CNMF (serial) | 10.5 s | 10.5 s | 10.5 s | 10.5 s | 1x |
  | whole run | 167 s | 94 s | 53 s | 34 s | 4.9x |

  Whole 5-min session (9000 frames, 3 windows, overlap on): 79 s with 8 workers against 487 s with one
  worker (the motion stage folded in) and 644 s before (8.2x); its outputs are the one-worker run's bit
  for bit (92 files, `C.npy` included); F1 0.962, recall 0.960, precision 0.964, trace correlation median
  0.995, 419 stitched, 418 kept, 417 merged units, as in the reference browser run.
- Memory: the wasm heap over all workers peaks at 1.0 GB with 8 (the run's worker 0.28 GB, about 90 MB
  per frame worker). Chrome's peak resident memory on 3000 frames in a profile on disk: 1.9 GB with 1
  worker, 2.6 GB with 4, 3.3 GB with 8 (the renderer 1.0, 1.6, 2.4 GB: about 0.2 GB per worker); in an
  ephemeral profile, where OPFS lives in memory and holds about 2 GB at most, 5.2 GB over the 9000
  frames.
- Chrome, Firefox and WebKit (Playwright's) give the same outputs as Chrome with one worker (63 files on
  6000 frames, 4 workers). WebKit allows about 250 open sync access handles for the whole page: one
  per tile and chunk (100 tiles) did not run, hence the slabs. Terminating the run's worker stops its
  frame workers (all three browsers); an error in a frame worker ends the run with its message.
- Checks (`web/e2e/stream.spec.ts` with `STREAM_WORKERS=n`, `STREAM_TILE_CRC=1`,
  `STREAM_KEEP_STORE=1`): every output file of the run (A, C, units, shifts, quality, template,
  window factors, kept tiles, tile CRCs) is identical for 1 and N workers and to the run before the
  pool; only timings in the JSON differ. Native: `cargo test` (chunks merged = one run, solved frames
  folded = `push`, template from prepared frames, tile factors round trip).

## GPU path (2026-11-04): the per-frame kernels on WebGPU, same outputs bit for bit
Each frame worker can run its per-frame kernels on the GPU (`web/src/gpu/`, `web/src/stream/gpuStages.ts`,
`rust/minime-core/src/{gpu,gpuplan}.rs`): median, band-pass registration image and its 2x2 mean, the warp and
top-hat of the denoised frame, the warp of the registration image and its 2x2 mean. The CPU keeps the FFT shift
estimate, the quality, the fold and the tile writes, the same Rust code as the CPU path (`Motion::estimate_down`,
`quality_down`, `Pass1::fold_corrected`, `StreamSolver::solve_prepped`). Batches of 8 frames; the next batch's
GPU work overlaps this batch's CPU work. Pass 1 (with the shift estimate), the template and the streaming pass's
preprocessing use it. WebGPU works in nested workers in Chrome and WebKit, so every frame worker has its own device.

- **Same bits**: integer kernels are exact by construction; the float ones need a fused `fma` and no
  reassociation. Metal's fast math reassociates `(a + b) + (c + d)` (one value in ~1e5 off by an ulp), so the
  float 2x2 mean runs in two dispatches. Before a run uses the GPU, a **self-test** sends 3 of the session's
  frames (start, middle, end; shifts with a tie and a border) through every kernel and compares each output with
  the CPU's (`gpu_reference`) byte for byte; any difference turns the GPU off for the run, with what differed.
  A device lost or a GPU error during the run switches that worker's GPU off; the frames from the failed batch
  on go through the CPU path (nothing of the failed batch was folded).
- **Device** (`ToWorker.device`, New run's Speed card in the browser build): `auto` (default: the fastest the
  speed test measured), `cpu`, `gpu`, `both` (the first frame workers on the GPU, the rest on the CPU; a dealt
  pass 1 sizes the workers' chunks by their rates). A GPU run with one worker still gets a pool of one.
  Software adapters (SwiftShader, `isFallbackAdapter`) and unsupported parameters (`motion_before_tophat` off,
  `motion_down` 1, odd sizes) are refused with the reason.
- **Speed test** (~2 s at the run's start, cached in IndexedDB `minime-speed` per GPU, cores, frame size, workers
  and `SPEED_TEST_VERSION`): pass 1 on 32 session frames per worker, CPU / GPU (one, then all workers) / split,
  twice each, the faster run counts. `report.json` `device`: asked, used, GPU, why, self-test, speed test, frames
  each stage did on the GPU, fallbacks. The job page shows "Ran on ...".
- Measured (headless Chrome 156, M1 Max, load 4-10 from other sessions, bench lock), every run folder's 87 (5-min)
  / 147 (V4) non-JSON files identical across modes and worker counts; 5-min F1 0.962, traces 0.995:

  | session | workers | CPU | GPU | both | pass 1 CPU -> GPU |
  |---|---|---|---|---|---|
  | 5-min synth (9000 frames) | 8 | 31.4 s | 26.4 s | 27.5 s | 12.8 -> 8.7 s |
  | | 4 | 48.3 s | 35.3 s | 39.5 s | 20.5 -> 10.1 s |
  | | 2 | 87.9 s | 64.0 s | 74.8 s | 37.8 -> 17.8 s |
  | 20-min V4 (23711 frames) | 8 | 114.8 s | 98.5 s (auto 95.2 s) | 105.9 s | 39.1 -> 21.3 s |

  With 8 workers pass 1 is held by the tile writes to OPFS (~3 ms per frame per worker with eight writing) and the
  FFT estimate (1.3 ms); pass 2 (CPU) is then the largest stage. Both was never the fastest here (the M1 Max GPU
  outruns its CPU workers); it is for a weak GPU beside many cores.
- Fallbacks checked: no WebGPU (`STREAM_GPU_TEST=nogpu`, and Playwright's headless shell: no adapter), a
  self-test mismatch (`mismatch`), devices lost after the 3rd / 30th batch (`lost`, `lost:30`, in the template and
  mid pass 1): CPU path, the same files. WebKit: self-test passes, speed test runs (`e2e/speed.spec.ts`); its full
  e2e did not run here because OPFS fails in Playwright's WebKit on this machine (`UnknownError`, CPU mode and a
  bare probe alike). Firefox: not installed for this Playwright.

Run it: `STREAM_DEVICE=auto|cpu|gpu|both [STREAM_GPU_TEST=...] STREAM_RAW=... STREAM_OUT=... npm run e2e:stream`;
the speed test alone: `e2e/speed.spec.ts` with the stream config's server.

## Spread template and template fill (2026-11-05): MiniMe's new motion defaults, same as Python
MiniMe's defaults changed to `template_source = "spread"` and `motion_fill = "template"` (Python on
`claude/motion-defaults-bc3667`, brought over here): a first-frames template sat where the field was in
the first seconds, often px away from the rest of the session, and the warp's zero band that came and
went with the motion became false units along the edges. `session.json` carries both keys (`pass1`;
`motion_fill` also in `stream`); inputs without them (written before) mean `first` / `zero`.

- **Template** (`motion::SpreadBuilder`, `spread.ts`): 10 runs of 20 frames spread over the session
  (`np.linspace(0, T - 20, 10).round()`), read with `File.slice` through the AVI index. Five reads: the
  runs' prepared means (the best-agreeing run starts), twice the shifts and correlation peaks of every frame
  against the current template (search 2 x `motion_max_shift`) and the mean of the frames warped by their
  shift minus `_reachable_median` (median of the frames whose peak is at least half the median, clipped so
  they stay within 0.75 x `motion_max_shift`; numpy's percentile and float32 sums reproduced). The frame
  workers prepare (`SpreadWorker`, CPU only: 200 frames, 1.2 s with 8 workers); the run's `MotionEngine` adds
  prepared frames in frame order, so 1 and N workers give the same bits.
- **Fill image**: a sixth read: each template frame's denoised image warped against the final template with
  the edge pixels replicated, top-hat, summed (uint32) over the frames that hold data at each pixel, floor of
  the mean; workers return partial sums in any order. In the run folder: `motion/arrays.npz` `fill`.
- **Pass 1 and the streaming pass**: the denoised frame is warped with `BORDER_REPLICATE` (`warp.rs`
  `replicate`; checked against `cv2.warpAffine`: 300 random cases and 600² frames, bit for bit), top-hat,
  then `fill_outside` (`fill.rs`: rows above `y0` / from `y1`, columns left of `x0` / from `x1`,
  `motion._valid_rects` of the float64 estimate, or of the float32 shift the streaming pass reads). The fill
  is a CPU copy in `Pass1::fold_corrected` and `StreamSolver::solve_prepped`, so the GPU path shares it; the
  GPU's warp of the denoised frame has a replicate variant (`kernels.ts` `shiftU8Replicate`, `Kernels.replicate`)
  and the self-test's CPU reference uses it.
- Checks (30-min GRIN synth, its new-defaults batch run `processed/motion_defaults_browser` as the baseline):

  | check | native vs Python | Chrome vs Python |
  |---|---|---|
  | template (frames 0-3000 estimated against it) | ≤ 2.6e-7 of max | ≤ 3.4e-7 of max, 1 and 8 workers identical |
  | shifts | ≤ 4.4e-6 px, same integer peak in all 3000 frames | ≤ 5.1e-6 px, same integer peak |
  | fill image | identical | identical |
  | pass 1 with the fill (given shifts): max projection, 11 block maxima, 100 tiles | identical (CRC) | |
  | streaming pass with the fill (3000 frames, 228 units): preprocessed frames, `C`, sums | frames and `C` identical, sums ≤ 4.5e-15 | |
  | 2 windows (6000 frames, `ref_stitch.py --session-inputs`): window units, supports, pairs, edges, parents, groups | | identical (193 + 181 -> 231 units); chain PASS |
  | the same on the GPU (8 workers) | | 44 files identical to the CPU run |

  5-min synth from the AVIs alone (Chrome, 8 workers): 398 units, F1 **0.967** (precision 0.995, detectable
  recall 0.965), traces 0.995, 2 false units within 15 px of an edge; the run before (2026-11-03, first-frames
  template, zero fill): 419 units, F1 0.962 (precision 0.964, detectable recall 0.965), 14 false edge units.
  Python's batch run with the new defaults: F1 0.963, 393 units. GPU and CPU runs give the same 88 files.
- Not ported: `motion_outlier` (opt-in in Python; New run lists it as a parameter the browser ignores).
  The speed test's engines run with the zero fill (no fill image yet; the same speed).

Run it (the e2e on its own port and build folder when another run uses 8796):
```bash
python experiments/streaming/ref_motion.py --baseline motion_defaults_browser --name ref_motion_spread --t0 0 --t1 3000
minime-core motion --raw <raw> --params <ref>/inputs/params.json --out <ref>/rust_native --check-frames <ref>/inputs/check_frames.npy
STREAM_PORT=8831 STREAM_DIST=dist-stream-md STREAM_WORKERS=8 STREAM_RAW=<raw> STREAM_INPUTS=<ref>/inputs STREAM_OUT=<ref>/rust_browser npm run e2e:stream
python experiments/streaming/check_motion.py <ref> <ref>/rust_browser
python experiments/streaming/t1_discovery.py --mode fresh --windows 3000 --baseline motion_defaults_browser --name t1_fresh_fill --n 2
python experiments/streaming/ref_stitch.py --session-inputs --stream --n 2 --baseline motion_defaults_browser --t1 t1_fresh_fill --name ref_stitch_fill
```

## Auto-tune and reruns (2026-11-05, branch `autotune-browser`)

What runs where: the run's windows are kept in OPFS; `New run` with "auto-tune after the run" chains a tune job.
The tune workers (`TuneWorkers`, one `WindowEngine` per worker, windows spread over the pool) run each trial on
all 12 windows (seeds, init, CNMF, stitch not included) and the per-window scores are summed; `LocalSearch` walks
the grids one parameter at a time. Then the 12 windows are rerun with the proposal (`window2`), stitched and
streamed once (`stream`, the scored traces) into `reruns/auto/`. `flags`/`pairs` (the unit table and pair counts)
are computed by one `score_units` wasm call on the main thread with the 202 pattern (the spec said "in the tune
worker"; the plan moved it: K x T float32 takes well under a second on grin30).

Layout: `reruns/<name>/` is a full run folder (`cnmf/A.npz`, `C.npy`, `YrA.npy`, `params.json`, `report.json`)
plus `autotune.json`, `sweep.csv`, `unit_flags.csv`. A rerun reads the run's `factors/` and needs `sigma.npy`;
runs made before 2026-11-05 have none and must be run again to be tuned.

| `synth_v4_grin30` | browser | Python auto-tune |
|---|---|---|
| run F1 (YrA) | 0.757 | 0.680 |
| proposal F1 | 0.748 (R 0.664, P 0.857, 357 units) | 0.682 |
| proposed | merge_corr 0.7, inita_corr 0.98 (at the grid edge), rest the run's | sparse_snr 0.714, merge_corr 0.85, inita_corr 0.85, loop_merge_corr 0.75 |
| auto-tune wall | 61 s (rerun + stream 26 s) | 192 s |
| trials | 17 | 16 |
| per-setting cost | 0.5 to 3 s (12 windows, parallel) | |
| tune-worker wasm heap peak | 173 MB | |
| Chromium tree RSS peak | 3.7 GiB (3.58 GiB for a plain run) | |
| click to done (run + tune) | 173 s (run 116 s) | main run 95.5 s |

Memory: the tune workers' wasm heap peaks at 173 MB; the Chromium process tree peaks at 3.7 GiB against 3.58 GiB for
a plain run, so the tune adds little. The tree RSS counts the GPU process and the renderer baseline and shared
pages once per process, so it is not a budget for the tune alone.

Window trials: a window's units come out of the CNMF loop as `merge.c` = the merged YrA and `merge.extra` = the
merged C (`cnmf::outputs`). The window engine first took `extra` as YrA, so trials were scored on C and the stitch
brightness (max per unit) came from C; the rerun of the run's own parameters then stitched to 365 units against the
run's 366. Fixed (`WindowSession::run` takes `merge.c`): the rerun now reproduces the run (2276 window units, 366
stitched). When nothing beats the run the rerun uses the run's own parameters, as Python's `Sweep.run`.

Result after the fix: the search finds a gain on the summed window scores (1646 vs 1608: merge_corr 0.7, inita_corr
0.98, at the edge of its grid), but the proposal's real F1 is lower than the run's (0.748 vs 0.757, 357 vs 366 units;
precision 0.857 vs 0.855, recall 0.664 vs 0.679): the window-score gain is flagged-unit bookkeeping (fewer duplicates
and low-PNR units) that costs true units. Python's search on the whole-session score picked other values (sparse_snr
0.714). Open: why the summed window score and the truth disagree here (sparse_snr 0.714 now scores 1440 vs 1608, with
527 flagged).

## Next
0. Rerun the 8 h and 24 h streaming pass with the traces in OPFS (8 h scale run above);
   machine-hours, so overnight.
1. ~~Discovery port~~, ~~run from the AVIs alone~~ (done). Next: the full 36-file GRIN
   session with New run (~45 min single core), its run folder scored with `minime score`
   against the batch run's F1 0.740 / the `fresh` chain's 0.738; and the same with `STREAM_INPUTS=<ref>/session` (`ref_stitch.py
   --session-inputs`, all windows) checked with `check_stitch.py --own-windows` against
   `ref_stitch_w3000`, watching the 2.9e-5 brightness tie.
1b. ~~`ymean`/`sn` for a browser-only run, window loop chained into the streaming pass~~
   (done). A second streaming pass on the updated footprints (`t2_stream.py --n-updates
   1`'s pass 1, the scored traces) is one more read of the movie; not chained yet.
1c. ~~Fold the shift estimate into pass 1~~ (done, "Worker pool" below): the motion stage is now the
   template build only.
2. Speed: ~~worker pool~~ (done, below); faster median (Perreault O(1) or a sorting-network
   median), a faster NNQP/CNMF loop (now the largest serial part), then rayon + COOP/COEP.
3. ~~Save outputs to a folder~~ (done: picked folder or Browser storage, "Browser back end"
   above). Still open: checkpoint the per-segment sums so a run can resume.
4. WebGPU for traces and sums once the CPU path is complete.
