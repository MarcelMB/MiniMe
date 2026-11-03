# Streaming MiniMe prototype (browser branch)

Can MiniMe find footprints and traces without holding the whole session, not even as
factors? The design is in `docs/browser-options.md`. This folder tests it in Python on the
30-min synthetic GRIN session, which has ground truth. The browser port (Rust core in
`rust/minime-core`, page `#/stream` in `web/`) is checked against these scripts.

**Inputs:** `/Volumes/Extreme SSD/synth_v4_grin30/` (`raw/`, `meta.json`, `truth.npz`) and
its batch run `processed/baseline/`. From the baseline the scripts take the motion
shifts, the parameters and, as reference, `score.json`. Recall of detectable cells there
is 0.57, precision 0.94, trace correlation 0.86.

**Outputs:** `processed/stream/` in the session folder. Raw data is never written.

| Script | Question |
|---|---|
| `t1_discovery.py` | Test 1: do footprints from short windows, stitched together, find as many cells as the batch run (late and dim cells especially)? |
| `t2_stream.py --footprints baseline_final` | Test 3: are per-frame traces with fixed footprints as good as the batch loop's? |
| `t2_stream.py --footprints baseline_init --n-updates 1` (and `2`) | Test 2: does a footprint update from streamed sums match the batch loop? |
| `t2_stream.py --footprints stitched --stitched DIR --n-updates 1` | The whole bounded-memory pipeline. Test 4 (footprint stability per segment) runs after every first update. |
| `t2_stream.py --footprints stitched_u1 --project windows` | Test 5: do per-frame traces on frames projected onto a per-tile pass-2 basis (session, per window, or merged with a cap) beat raw frames? No: ±0.01. `--background mean` (`[A; mean image]` instead of carving the units' LS share out of `b`) is what closes the gap: 0.77 → 0.91. |
| `ref_dump.py --footprints stitched --stitched DIR [--frames N]` | Reference for the Rust core: one pass + update, with its inputs as plain `.npy` (`processed/stream/ref_<name>_<T>/`). |
| `check_rust.py REF_DIR RUST_DIR` | Compares a native or browser Rust run with that reference (bit-exact frames, sums to 1e-9, footprints to 1e-6). |
| `ref_motion.py` / `check_motion.py` | Reference and check for the Rust motion stage on one window. |
| `ref_pass1.py` / `check_pass1.py` | Reference and check for the Rust pass 1 on one window (max projection, block maxima, tile store by CRC-32). |

## Run

Run from the repo root with the project venv:

```bash
uv venv -p 3.11 && uv pip install -e ".[dev]"
.venv/bin/python experiments/streaming/t1_discovery.py --mode subset --windows 1000 3000 6000
.venv/bin/python experiments/streaming/t2_stream.py --footprints baseline_final
.venv/bin/python experiments/streaming/t2_stream.py --footprints baseline_init --n-updates 2
.venv/bin/python experiments/streaming/t1_discovery.py --mode fresh --windows 3000
.venv/bin/python experiments/streaming/t2_stream.py --footprints stitched --n-updates 1 \
    --stitched "/Volumes/Extreme SSD/synth_v4_grin30/processed/stream/t1_fresh/w3000"
```

Cost:
- `subset` windows take seconds each.
- `fresh` windows run the full pipeline per window, about one batch run in total.
- Every `t2` pass reads the 13 GB movie once and runs median, shift and tophat per frame.

## Shortcuts and limits
- The first pass takes its mean image (for the background) and per-pixel noise from the
  baseline factors. A browser would collect both in its discovery pass. `--sn diff` uses
  a noise estimate from frame differences instead.
- Traces are per-frame least squares on `[A; b]`, clipped at 0, with no deconvolution.
  The batch loop's `YrA` is one correction step from the previous traces. By default `b` is
  the mean image minus its LS fit on `A` (the batch's initial background), which carves a
  hole per unit and leaks the background into the traces; `--background mean` uses the mean
  image itself (test 5).
- The merge uses correlations of the raw per-frame traces from `Σ c cᵀ`. The batch merge
  correlates low-passed traces.
- `subset` windows flatter discovery: the factors' tile bases were fitted to all frames.
  `fresh` is the honest test.
- There is no residual search for cells the windows missed yet.
- State per pass (`state_mb` in `results.json`) counts what a browser would keep between
  frames. The spatial solve builds a dense `(active pixels, K)` matrix as the batch code
  does, which a browser version would need to avoid.
