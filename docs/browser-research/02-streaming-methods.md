# 02 – Streaming / "don't need all the data" methods

The question: which CNMF-E variants never hold anything that grows with T, apart from the
output traces C (K×T, which can be streamed to disk or OPFS)?

Notation: d = pixels, K = units, T = frames, s = size of the dilated support per unit.

## 1. Online CNMF: OnACID and OnACID-E

- **OnACID** ([Giovannucci et al., NeurIPS 2017](https://papers.NeurIPS.cc/paper_files/paper/2017/file/4edaa105d5f53590338791951e38c3ad-Paper.pdf),
  [bioRxiv](https://www.biorxiv.org/content/10.1101/193383v1)) is online NMF in the style of
  [Mairal et al. 2010](https://jmlr.org/papers/v11/mairal10a.html) applied to CNMF.
  - It starts with batch CNMF on the first few hundred frames.
  - For each new frame it solves for c_t with A held fixed (a few warm-started HALS/NNLS sweeps).
  - It keeps running averages of the sufficient statistics
    **L = (1/t)Σ y_t c_tᵀ (d×K, stored on the supports only)** and **M = (1/t)Σ c_t c_tᵀ (K×K)**,
    and updates A periodically by block-coordinate descent (HALS) on those two matrices alone.
  - It finds new cells in a residual buffer of the last l_b frames.
- **OnACID-E** ([Friedrich, Giovannucci & Pnevmatikakis, PLoS CB 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7842953/))
  adds the CNMF-E ring background and CNMF-E detection.
  - Extra state: χ = XXᵀ for the per-pixel ring regression W (d × ~2πl nonzeros), a mean
    baseline, AᵀW, AᵀWA, and the residual buffer (l_b ≈ 100 frames).
  - It initializes with batch CNMF-E on about 200 frames and refreshes shapes and background
    every 200 frames.
  - New neurons come from an incrementally updated PNR × local-correlation image of the residual.
    Each candidate is fitted with a local rank-1 NMF and must pass shape and SNR tests.
  - The authors report memory as constant in recording length, about 10× below batch CNMF-E.
- **Accuracy vs batch** (from the same paper):
  - Detection F1 is the same or slightly higher: 0.89–0.96 on real data, 1.0 on simulated data.
  - Traces are slightly worse. Hippocampus: correlation to ground truth 0.938 (online) vs 0.983
    (batch). BNST: 0.854 vs 0.912.
  - The alternative "ring-CNN" background (2 conv kernels, learned at init, then frozen) is
    faster. On real data its F1 was lower (0.85), and the background can't follow changes.
- **Failure modes the authors state:**
  - Weak or rarely firing cells that appear late may never trigger detection.
  - Thresholds have to be tuned per dataset.
  - A frozen background breaks if the background changes (drugs, optogenetics).
  - Motion correction uses a template averaged over about 50 recent frames. No drift benchmark
    is reported.
- **FIOLA** ([Cai et al., Nat Methods 2023](https://github.com/nel-lab/FIOLA)) uses the same split:
  footprints from an initial batch, then a GPU per-frame NNLS trace solve with motion
  correction at >300 Hz. It is tracking-only, with no new cells.

## 2. Sufficient statistics in MiniMe's terms

MiniMe's loop is already almost frame-separable (`cnmf.py`):

| Stage | What it needs | Streamable? |
|---|---|---|
| `update_spatial` | G = CcCcᵀ (K×K) and `YCt` restricted to the dilated supports (d_sup×K), plus sn and T | Yes. Accumulate Σy_tc_tᵀ, Σc_tc_tᵀ, Σc_t and the mean image; then Y Ccᵀ = Σy cᵀ − T ȳ c̄ᵀ |
| Background, unclipped (default) | b = clip(ȳ − A c̄, 0), with f_t computed per frame | Yes |
| Background, `background_clip=True` | Needs the full residual | No. Clipping is nonlinear, so it needs its own pass |
| `compute_trace` | Aᵀ(y_t − b f_t), with AAᵀ fixed | Yes, frame by frame |
| Seeds, PNR, GMM | Max projection, local correlation, noise | Yes, at O(d) (see below) |

**The catch:** L depends on C, so every CNMF iteration needs one more read of the data. There
are two options:

- **Store factors** (today). The spatial part W_b is O(d·r) and fine. The temporal part
  V_b (T × Σr_b) is what grows with T: about 3 GB in float32 for 2 h at 20 fps with Σr ≈ 5000.
- **Re-read the file each iteration.** In a browser, `File.slice()` streaming makes extra reads
  cheap in memory; they cost decode time only.

So the realistic target is a small number of streaming passes, each with tiny state.

The pass-A summary images all fit in O(d):

- **Running max:** catches rare events, which a mean would wash out.
- **Local correlation:** needs Σy, Σy², and Σy_i y_j over 8 neighbours, about 10·d float64
  (29 MB at 608²). MiniMe's band-limited correlation also needs a per-pixel IIR low-pass
  state, O(d·order).
- **Noise sn:** high-pass variance via Welford, approximating the PSD estimate.
- **PNR:** from the filtered running max and sn.
- **Block maxima:** already stored; keep the number of windows fixed rather than growing with T.

## 3. Subsampling and sketching

- **Temporal subsample for footprints.** Suite2p builds its detection SVD on at most `nbins`
  = 5000 binned frames, whatever T is ([docs](https://suite2p.readthedocs.io/en/latest/parameters/)).
  CaImAn initializes on `tsub`/`ssub`-downsampled data and OnACID on the first ~200 frames.
  - I found no paper that gives a frame budget for detecting rarely firing cells.
  - Rough model: a cell with rate λ shows about s·λ·T events in a fraction s of the session.
    At λ = 0.01 Hz over 2 h with s = 10%, that is about 7 events.
  - Spread the subsample as random blocks across the whole session, not the first N minutes.
- **Single-pass sketches** give a spatial basis in O(d·ℓ) memory, independent of T.
  - [Tropp et al. 2017](https://arxiv.org/abs/1609.00048) and
    [streaming follow-up 2019](https://arxiv.org/abs/1902.08651): single-pass randomized
    low-rank sketch.
  - Frequent Directions ([Liberty 2013](https://pdf.arxiv.org/pdf/1307.7454)): with ℓ = 2/ε,
    the covariance error ‖YYᵀ − BBᵀ‖₂ ≤ ε‖Y‖²_F.
  - These would replace the in-RAM per-tile rSVD, but only W, not V, becomes independent of T.
- **NMF coresets** ([Feldman & Tassa, KDD 2015](https://cris.openu.ac.il/en/publications/more-constraints-smaller-coresets-constrained-matrix-approximatio/))
  are weighted frame subsets of size independent of T, built in one pass. They are the
  principled version of "pick frames". The theory exists, but I found no calcium-imaging use.
- **Compressive imaging** ([Pnevmatikakis & Paninski, NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/file/5e9f92a01c986bafcabbafd145520b13-Paper.pdf)):
  with known A, a few random projections per frame are enough to recover sparse activity.
  This supports "known footprints, then a cheap per-frame solve", not detection.

## 4. Other memory-light pipelines

- **EXTRACT** ([Inan et al.](https://github.com/schnitzer-lab/EXTRACT-public)): robust
  estimation, GPU, works in spatial partitions. It is still batch over T.
- **Real-time / closed-loop systems** are bounded-memory by construction, and all of them
  freeze footprints after an initial window:
  - **ACTEV** ([UCLA FPGA, BioCAS 2021](https://vast.cs.ucla.edu/sites/default/files/publications/BioCAS21_LiveDemo_Zhe.pdf)):
    Miniscope, motion correction plus up to 1024 cell traces per frame, under 1 ms latency.
  - **DeCalciOn** ([eLife 2023](https://elifesciences.org/articles/78344.pdf)): real-time
    imaging and decoding.
  - **improv** ([Nat Commun 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12606217/)): a
    platform for real-time, adaptive experiments.
- **Browser:** [miniscope/CaLab](https://github.com/miniscope/CaLab) runs Rust→WASM FISTA
  deconvolution on traces only. I found no browser tool that extracts cells from video.
- **Deconvolution:** [OASIS](https://pmc.ncbi.nlm.nih.gov/articles/PMC5370160) is online and
  sequential, ran on about 10⁵ traces in real time on a laptop, and its per-trace state is small.

## 5. Risks over a 2 h session

1. **Rare or late cells missed by the subsample.**
   - The max-projection seeds cover the whole session.
   - Keep a residual max/PNR image during the trace pass and add flagged cells afterwards.
2. **Motion.** A rolling template (mean of recent corrected frames) streams fine, but a fixed
   template drifts over 2 h.
3. **Slow drift (z or non-rigid).** Footprints frozen in the trace pass go stale.
   - Keep L and M per segment (for example every 10 min). Each segment adds only
     O(s·K + K²) state.
   - That gives a spatial update per segment and a measure of footprint stability.
4. **Photobleaching and background drift.** The per-frame f_t absorbs the global part. Use a
   running-percentile baseline per trace for ΔF/F. The tophat is stateless, so it streams.
5. **Online traces trail batch traces** (0.94 vs 0.98 in OnACID-E). A final pass with the
   refined A recovers most of the gap.

## Ranked shortlist

1. **Seeded three-pass with sufficient statistics** (closest to MiniMe).
   - *Pass A:* motion correction, tophat, O(d) summaries, plus a stratified frame subsample of
     size S (fixed, e.g. 3–6k frames). Run the existing window-2 pipeline on factors of the
     subsample only.
   - *Pass B:* fixed A. Per-frame warm-started NNLS gives c_t and f_t. Accumulate per-segment
     L and M and a residual PNR/max image. Then do one spatial update from L and M with no data
     access, and add the missed seeds.
   - *Pass C:* final traces, then OASIS.
   - **State:** O(S·d) or O(S·r + d·r) in pass A, and O(nnz A + n_seg(s·K + K²) + 10·d) after
     that. None of it depends on T.
2. **OnACID-E-style single pass**: batch init on the first T_b frames, then online tracking,
   periodic HALS from L and M, and a residual buffer.
   - **State:** O(d·l_b + s·K + K²), plus O(d·2πl) for a ring background. MiniMe's tophat +
     rank-1 background avoids that last term.
   - Fewest reads, and it follows drift. On the other hand it has more thresholds, weaker
     traces, and early frames are solved with immature footprints.
3. **Keep factor-first, bound RAM rather than T**: a single-pass sketch for W_b, with V_b
   quantized to float16/int8 and streamed to OPFS.
   - RAM is O(d·r); disk is O(T·Σr).
   - It changes the least code, but the browser storage quota becomes the limit.

**Test first on the synthetic sessions (ground truth):**

- (a) Footprint recall and precision against subsample size S and layout (first N minutes vs
  strided blocks), split by true firing rate. `Factors.subset(t0, t1)` already gives
  contiguous slices.
- (b) Trace correlation to truth: fixed-A per-frame NNLS vs today's full iterated loop.
- (c) One spatial update from L and M vs `n_iter` iterations.
- (d) `grin_v4`: does a residual PNR image recover the cells that (a) missed?
