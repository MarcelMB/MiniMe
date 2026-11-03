# MiniMe in the browser — state (2026-11-02)

Research only, nothing run. Lane reports: `docs/browser-research/01-04`.

## Verdict
- Compression alone doesn't fit: a 2 h session is ~1.9 GB of factors (+ ~1.7 GB in window 2)
  against a tab budget of **≤ 2 GB** (wasm32 hard cap 4 GB; iPad ~1.5 GB; OOM kills the tab).
- Streaming does fit: `File.slice` random-reads any frame of a 50 GB AVI, so the movie can stay
  on disk and be re-streamed. A streaming MiniMe needs **~100–200 MB state, independent of T**.
- Long runs are the second limit: tabs get discarded/paused, no sleep guard → every pass must
  checkpoint to OPFS and resume.

## Candidate algorithm (ranked)
1. **Three passes, tiny state** (recommended)
   - A: summary images (mean, max, local corr ~10 sums/px, noise, PNR) + a fixed-size frame
     subsample spread over the whole session → existing seeds → init → CNMF on the subsample.
   - B: per-frame NNLS with fixed A; accumulate L = Σ y·cᵀ (on supports), M = Σ c·cᵀ per
     segment + residual PNR image (catches cells missed by the subsample) → one spatial update
     from L, M with no data access.
   - C: final per-frame traces (no deconvolution: MiniMe does none).
   - State O(nnz A + n_seg·(s·K + K²) + 10·d).
2. **OnACID-E-style single pass** (Friedrich 2021): proven, detection ≈ batch (F1 0.89–0.96),
   traces worse (r 0.94 vs 0.98), late/weak cells missed.
3. **Keep factors, stream V into OPFS**: closest to today's code, but still O(T) disk + reads.

Already streaming-friendly in MiniMe: pass 1, `initC`, `compute_trace`, default background,
`update_spatial` / `solve_nnqp` (need only Grams). Needs all of T: pass-2 tile load + noise FFT,
`FactorCorr` non-causal filter, `rolling_windows`, `pnr_refine` percentiles. Streaming needs the
`brightest_parent` merge rule too (seed windows grow without bound).

## Tech stack
- Rust → wasm core (AVI parser, per-pixel stats, NNQP; rayon threads + SIMD) + WebGPU for GEMM,
  local corr, per-frame traces (WGSL from the wgpu-py work carries over). CPU fallback required
  (WebGPU shaky on Linux).
- Pyodide runs MiniMe almost as is (numpy/scipy/opencv 4.11; numba only in `spatial_kernel.py`)
  but single-thread, ~2 GB → demo only.
- Prior art: miniscope **CaLab** (Rust/wasm in workers, GitHub Pages, data stays local) — same
  architecture; nothing in-browser extracts cells from video yet.
- Video: raw GREY AVI = trivial parser; MJPEG via `ImageDecoder` per frame; FFV1 needs
  ffmpeg.wasm (slow, 2 GB cap).
- Hosting: threads need COOP/COEP → coi-serviceworker on GitHub Pages, or Netlify/Cloudflare.
- One React UI, two backends: `minime serve` (2–24 h, batch) or wasm worker (zero-install, demos,
  short sessions, tuning).

## Results: 30-min GRIN synth (2026-11-02, `experiments/streaming/`)
Batch baseline: 262 units, recall 0.53, precision 0.94, F1 0.68, trace corr 0.86.

| Test | Result |
|---|---|
| 1. Window discovery (`subset`, 3000 / 6000 frames) | 307 / 309 units, recall **0.62**, precision 0.93, F1 **0.74** (beats batch); late cells 0.85 (same as batch) |
| 2. One update from sums (from init footprints) | F1 0.687 after 1 update, 0.679 after 2 (batch 0.680) → one streamed update is enough |
| 3. Fixed footprints, per-frame traces | corr with batch traces 0.98, with truth **0.81** (batch 0.86) |
| 4. Footprint stability (6 × 5 min) | cosine 0.975–0.988, centroid shift 0.26–0.34 px → fixed footprints OK |
| 1b. Window discovery, `fresh` (12 × 2.5 min, own runs) | 365 units, recall **0.70**, precision 0.85, F1 0.75 (better than `subset`) |
| **Full streaming pipeline** (fresh windows → stitch → 1 update → traces) | 319 units, recall **0.66**, precision 0.91, F1 **0.74** (batch 0.68); trace corr **0.77** (batch 0.86); state 66–72 MB |

- State between frames: **62 MB**, independent of T. Each pass takes about 95 s for 30 min,
  single-threaded Python: 40 s read and preprocessing, 42 s sums, 13 s solve.
- Peak RSS is 1.8–2.6 GB, from the scripts' own overhead rather than the streaming state:
  - the dense spatial solve
  - loading the factors for the mean image and noise shortcut
  - the truth arrays for scoring
- Figure: `processed/stream/figures/streaming_results.png` (`plot_results.py`).
- Unmatched units (full pipeline): 29 against batch's 16. 16 sit within 8 px of a real
  cell (duplicates or splits), 10 on out-of-focus cells, 3 on nothing.
- The low-pass merge (causal Butterworth at `noise_freq`, as batch) barely helps:
  - 1 update: 315 units, precision 0.911, F1 0.740.
  - 2 updates: 302 units, precision 0.921, recall 0.638, F1 0.729.
  - It also merges real cells. The duplicates mostly don't overlap and correlate, so the
    fix belongs in the stitching and demixing, not in the merge.
- Full pipeline: 335 s for the windows plus 2 × 105 s of streaming passes.
- Stitched footprints drift more late in the session: per-segment cosine falls from 0.99 to
  0.96, and the lower quartile to 0.83.
- Caveats:
  - The worry that `subset` flatters discovery was wrong: `fresh` finds more cells (0.70
    vs 0.65).
  - The 1000-frame windows are invalid: Minian's rolling seed windows are not aligned to
    1000-frame bounds, so 34 of 36 windows had no seed window.
  - Stitching loses 0.04 recall against the plain union (overlap 0.51 → 0.38).

## Open actions
- [x] Honest discovery + full pipeline: done, F1 0.74 vs 0.68.
- [ ] Main gap is traces: 0.77 with stitched footprints, 0.81 with batch footprints, 0.86 batch.
- [ ] Close the trace gap (0.81 vs 0.86): the batch traces come from the low-rank, denoised
  movie. Try projecting each frame onto a running per-tile spatial basis (no deconvolution).
- [ ] Stitch: tune `--cos`, or keep same-cell links only when the footprints also match in
  size, so that overlapping pairs survive.
- [ ] Align windows to seed windows, so windows the length of `seeds_wnd` work.
- [x] Browser slice 1 (2026-11-02): raw AVI → preprocessing → traces + sums → footprint
  update, Rust → wasm in a worker (`#/stream`), matches Python exactly; see `docs/browser-app.md`.
- [ ] Browser: port discovery (full window pipeline), then threads, WebGPU, OPFS checkpoints
  (`docs/browser-app.md`, Next).
- [ ] Check: does Pyodide's OpenCV `VideoCapture` read AVI at all (expected no).
- [ ] Reconcile V size: 1.66 GB (lane 03, measured rank sum, float32) vs ~3 GB (lane 02).
