# 03 — What each MiniMe stage needs from the data

Read-only code review of `cross-platform-support` (working tree as is). Notation: T frames,
d = H·W pixels, P_b pixels per tile, r_b tile rank, R = Σ r_b, K units, n_s seeds.
There is **no deconvolution** in MiniMe: the final trace is `YrA` (`cnmf.py:333-344`).

## Stage map

| stage (code) | reads | holds | writes | O(T) | (a) 1 pass, small state | (b) time subsample | (c) summaries only |
|---|---|---|---|---|---|---|---|
| template `pass1.py:67-72` | first 200 frames | 200·d | – | O(1) | yes | native | – |
| pass 1 `pass1.py:122-171` | raw frames, 1000/batch (`:37`) | template FFT, `max_proj` d, `block_max` n_blk·d, shifts T·2 | tile store = raw size (`tiles.py:52`), sidecar | O(T·d) disk | **yes**, all per-frame; only `block_max` grows with T | no (motion is per frame) | – |
| pass 2 `compress.py:176-212` | a tile's **all T** frames into RAM (`tiles.py:68-79`) | T·P_b uint8 + Ω, Bt (T, r+10) f32; ≥5 reads per tile, more on rank doubling (`:198-203`) | V (T, r_b), W (P_b, r_b) | O(T·P_b) RAM/worker | partly: mean/energy are running sums (`:48-55`), `Y·Z` sums over chunks (`:71-77`); `t_matmul` output and the full-length noise FFT (`:98-109`) need all T | basis yes; noise has a segment mode (`noise_frames`, `:111-124`); lag-1 rank cut needs contiguous runs | no |
| corr setup `corr.py:48-74` | all V | **second copy** `Vlp` (R·T f32), Gaa r²/tile, lazy Gab (`:76-82`) | – | O(R·T) | Grams yes (Σ v_lp v_lpᵀ); brick-wall low-pass over the session (`:34-40`) is non-causal → needs a causal filter | yes (Grams average) | no |
| seeds_init `seeds.py:149-170` | `block_max` only | n_blk·d uint8 | csv | O(T/250·d) | yes: running window max + count image O(d); but window centres use `linspace(0, T-1)` (`:37-44`), so T must be known | yes | **yes** |
| pnr_refine `seeds.py:246-330` | rebuilt full trace per seed (`factors.py:147-157`) | batches ≤2 GB (`:242-243`) | – | O(n_s·T·r), n_s itself ∝ T → ~T² | no: 0.1/99.9 % percentiles of whole traces; needs per-seed quantile sketch or running extremes | yes (percentiles approx.) | no (std-only score tried, lost units: bench/README.md:336) |
| seeds_merge `corr.py:174-204` | pairs ≤10 px via factor Grams, `max_proj` | pairs | csv | O(pairs·r²), T-free | yes, given Grams | yes | needs Grams |
| initA `init.py:32-61` | corr in a 10 px disk | sparse A | npz | T-free | yes, given Grams | yes | needs Grams |
| initC `init.py:93-99` | `AtY` (`factors.py:159-169`) | C (K, T) f64 | – | O(K·T) | **yes**: c_t = G⁻¹ A y_t per frame | yes | no |
| unit_merge `init.py:123-156` | A Aᵀ, `traces_corr(C)` (`corr.py:124-134`) | K², low-passed C (K, T) f64 | – | O(K·T) | yes with K² Gram of causally low-passed c_t; averaging is linear | yes | no |
| background `cnmf.py:273-298` (default unclipped) | per-tile V·(Wᵀb), means | b (d), f (T) | – | O(R·T) | yes: b from running means, f_t per frame | b yes | b yes |
| update_spatial `cnmf.py:94-141` | G = Cc Ccᵀ, B = Y Ccᵀ via `YCt` (`factors.py:171-185`) | K², nnz(dilated A), C f64 | – | O(R·K·T) | **yes**: Σ c cᵀ, Σ c, Σ y cᵀ on the support, Σ y → centred, unit-norm G and B exactly; NNQP needs only G, B, λ (`spatial_kernel.py:5-11`) | yes | no |
| compute_trace `cnmf.py:333-344` | `AtY` | YrA (K, T) | C, YrA, A, b, f | O(K·T) | **yes**, per-frame linear map | – | no |

`run_cnmf` (`cnmf.py:371-418`) does `n_iter=2` loops (`params.py:77`), each re-reading all V
three times (background, `YCt`, `AtY`): exact batch, not one pass. A streaming version is
OnACID-style: the C behind the statistics is the online C, not the converged one.

Already streaming-shaped: `Pass1.process`, `TileMatrix` mean/energy and `matmul`,
`_tile_yct_f64` (chunked VᵀC, `factors.py:50-60`), `AtY`, `_background_unclipped`,
`compute_trace`, `solve_nnqp`. Needs all of T at once: tile read in pass 2, `t_matmul`,
FFT noise, `lowpass` of V and of C, `rolling_windows`, `seed_ptp` percentiles.

## Stored factors, 2 h · 600×600 · 20 fps (defaults)

T = 144,000; d = 360,000; `tile=64` → 10×10 = 100 tiles (81 of 4096 px, 18 of 1536, 1 of 576).
Rank is data-driven (start `svd_k0=40`, doubling to `MAX_RANK=300`, `compress.py:18,196-203`);
measured V4 2 h rank sum R = 2,885 (bench/README.md:291).

| item | formula | size |
|---|---|---|
| V | R·T·4 | **1.66 GB** (2.30 GB if every tile kept 40; 17.3 GB at the 300 cap) |
| W | Σ P_b r_b·4 | ~42 MB |
| mean, sigma, energy | d·(4+4+8) | 5.8 MB |
| block_max | ~577 blocks·d uint8 (289 windows, ~2 edges each, `pass1.py:45-57`) | ~0.21 GB |
| total factors | | **~1.9 GB** vs 52 GB raw / tile store (~27×) |

Window 2 then adds `Vlp` (+1.66 GB) and several (K, T) f64 arrays (K≈500: 0.58 GB each).
`block_max` is O(T·d) too: about half its blocks are 1–2-frame slivers between a window
stop and the next start, which could be dropped.

## Long-recording issues visible in code and notes

- Penalty ∝ √T: Minian `sparse_penal` thresholds grow as √T; replaced by `sparse_snr`,
  λ = √T·sparse_snr·sn on unit-norm traces (`params.py:71-75`, `cnmf.py:105-109`,
  `params.py:127-131`). T-invariant, which a streaming version needs.
- Merge percolation: 280 windows seed the whole halo (48k seeds at 2 h vs 7.9k at 10k);
  connected components collapsed 32k → 166. Fixed by `brightest_parent` (`corr.py:151-204`,
  bench/README.md:269-288). In a stream the window count grows without bound, so this rule
  (or a count cap) is required.
- `pnr_refine` cost seeds × frames (`seeds.py:260-262`); rank cut `z/√T` loosens with T,
  capped by `rho_min` (`compress.py:155-168`); densest tile 124 → 176 components at 140k.
- TODO.md:17-21: 24 h has V ≈ 20–23 GB in RAM; branch `stream-pass2` adds windowed rSVD →
  merged spatial basis → exact projection, and `traces.py` spills (K, T) arrays to mmaps.

## Minimum state for a "streaming MiniMe" (600×600, K≈500)

| state | size |
|---|---|
| motion template FFT + last shift | ~1 MB |
| running max, Σy, Σy², noise accumulator (e.g. frame-difference MAD/variance) | ~6 MB |
| current seed-window max + seed count image | 0.7 MB |
| per-tile spatial basis + sketch (Frequent Directions, P_b × 2r) | ~90 MB |
| Grams Σ v_lp v_lpᵀ for each tile and its 8 neighbours (seed corr ≤10 px) | ~7 MB |
| A (sparse) + dilated support, Σ y cᵀ on support, Σ c cᵀ, Σ c, chol(AAᵀ), b | ~5 MB |
| causal low-pass state + K² trace Gram (merge), per-seed PNR sketch | ~2–20 MB |
| optional residual buffer for new units (~200 frames uint8) | ~70 MB |

About **100–200 MB, independent of T** (O(d·r̄ + nnz(A) + K²)), against 1.9 GB of factors.
Output traces (K floats per frame) stream to disk or the browser.

## Browser port: reuse vs rewrite

- **Reuse (math is per-frame or T-free; port to TS/WASM or run in Pyodide):** `preprocess.py`,
  `motion.estimate/apply_shifts/frame_quality`, `seeds.local_max_*`, `label_medians`,
  `contrast_hist/knee`, `gmm.py`, `corr.brightest_parent`, `seeds_merge` and `initA` given a
  Gram-based `pairs()`, `init._solve_normal`/`unit_merge`/`centroids`,
  `cnmf.dilate_footprints/_size_mask`, the algebra of `compute_trace` and
  `_background_unclipped`, `spatial_kernel.solve_nnqp` (numba → needs a WASM/JS port).
- **Rewrite:** AVI decoding and the multiprocessing/tile-store plumbing (`io.py`, `pass1`
  workers, `tiles.TileStore`, `pass2`); `compress.py` rSVD (needs the whole tile → incremental
  basis + per-frame projection); `Factors` (V as T×r → accumulated stats); `FactorCorr`
  (global FFT low-pass → causal); `rolling_windows` (needs T); `seed_ptp` (percentiles →
  sketches); `run_cnmf` loop (multi-pass → online); `pipeline.py`/`window2.py` drivers.
  The existing `web/` React app is a viewer on the Python server's API, not a compute path.
