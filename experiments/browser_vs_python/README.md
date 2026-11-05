# Browser vs Python runs

Runs one session through main's Python pipeline and through the browser build, then
compares the two run folders. Inputs → outputs:

- `measure.py -- CMD`: runs CMD and samples its process tree's resident memory → `mem.json`
  (wall time, peak tree RSS).
- `run_browser.mjs`: drives the built browser app in headless Chromium like a user of a
  browser without the folder dialog (New run → upload the raw folder → Start run with
  default parameters → Open run → Download run (.zip)) → the unzipped run folder and
  `job.json` (job record, click-to-done wall time, console errors).
- `compare.py` (main's Python): two run folders (+ `--raw` for a synthetic session) →
  `metrics.json`, `stages.csv`, `figure.png`, `README.md` in `--out`.

```bash
# Python (main checkout, its .venv, no PYTHONPATH)
python measure.py --out mem_python.json -- .venv/bin/python -m minime run RAW --name main_DATE --fps FPS
# browser (serve the build first: cd web && npm run build:browser &&
#   npx vite preview --mode browser --outDir dist-browser --port 8795 --strictPort)
python measure.py --out mem_browser.json --match '(?i)chrom' -- node run_browser.mjs \
  --web web --base http://127.0.0.1:8795 --raw RAW --name browser_DATE --fps FPS \
  --dest SESSION/processed/browser_DATE --profile EMPTY_DIR --log job.json
# compare
python compare.py --python SESSION/processed/main_DATE --browser SESSION/processed/browser_DATE \
  --out SESSION/processed/comparison [--raw RAW] --mem-python mem_python.json \
  --mem-browser mem_browser.json --job-browser job.json
```

Footprints are matched one-to-one by centroid distance (≤ 5 px, `minime score`'s rule);
the cosine of matched footprints and the correlation of their traces (C vs C) measure how
alike the pairs are. Browser runs keep no `YrA.npy`, so their scores use `C.npy` (the
Python run is scored both ways).

## Run + auto-tune (2026-11-05)

`run_browser.mjs --tune` ticks "Auto-tune the seed, init and CNMF parameters ..." on the New run page; the
job then runs the run and the auto-tune (`reruns/auto/`, zipped with the run). Needs an absolute `--web`; vite
preview listens on `localhost` only (IPv6), so use `--base http://localhost:8795`.

```bash
python measure.py --out mem_browser_tune.json --match '(?i)chrom' -- node run_browser.mjs \
  --web $PWD/../../web --base http://localhost:8795 --raw RAW --name browser_tune_DATE --fps 20 \
  --dest SESSION/processed/browser_tune_DATE --profile EMPTY_DIR --log job_tune.json --tune
```

`synth_v4_grin30` (30 min, 36 AVIs; local outputs `job_tune.json`, `mem_browser_tune.json` here, not committed):
run F1 0.757 (scored on YrA, the new baseline), proposal F1 0.755 (target >= the run's: missed by 0.0015), auto-tune
64 s (`wall_s`; target <= 120 s: met), 14 trials, wasm heap peak 173 MB, click to done 181 s, Chromium tree RSS
peak 3.89 GiB (target <= 2 GB: missed; RSS counts shared pages once per process, so an upper bound).
