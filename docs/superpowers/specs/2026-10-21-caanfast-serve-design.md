# `caanfast serve`: run viewer and launcher in the browser

Date: 2026-10-21 · Branch `feat/serve` (from `rename/caanfast`) · Spec 1 of 3

## Goal

Scientists who do not code can open a finished CaAnFast run and follow every stage
(raw → preprocessed → motion → factors → seeds → units → traces) in their browser, and
start new runs from it. The views of `bench/stage_viewer.py` become interactive pages.
It stays smooth on 3 h recordings (216,000 frames, 600 × 600, traces 434 MB per run).

Decided with the user:

- Scope: viewer of finished runs + run launcher. Not in this spec: parameter tuning and
  window-2 reruns (spec 2, building on the unmerged `claude/tuning-viewer-13a50e` and
  `claude/sleepy-shamir-32bf6b` branches), installer (spec 3), shared multi-user servers.
- Name: CaAnFast. Code in `src/caanfast/serve/`, command `caanfast serve`.
- Same computer only: listens on 127.0.0.1; remote machines via an SSH tunnel.
- Python back end (FastAPI) separate from a React front end.
- Data transport "B": the back end sends raw arrays, the browser draws them.
- Look: bento + glass (artboard 4 of the "CaFast UI styles" canvas): cleaned max
  projection on top with contours and a video scrubber.
- Traces: optional low-pass as in CaTune (CaLab), cutoff set in Hz. Nothing else from
  CaTune.

## 1. Architecture

```
caanfast serve [FOLDER] → FastAPI on 127.0.0.1:<port> → opens http://127.0.0.1:<port>/?token=…
                              │
      run reader (read-only) · job runner (subprocess) · static files (built React app)
```

### Back end, `src/caanfast/serve/`

| Module | Job |
|---|---|
| `app.py` | FastAPI app, routes, token and `Host` check, static files |
| `runs.py` | Find run folders; `RunView` = one opened run: H, W, T, fps, which parts exist (tile store, raw AVIs, saved seeds/init) |
| `frames.py` | Frame `t` as raw (AVI), preprocessed (tile store, if kept) or reconstruction (factors). Moved from `bench/stage_viewer.py` (`raw_frame`, `store_frame`, `recon_frame`, `block_residual`, `outline`) |
| `arrays.py` | Binary array responses: raw bytes, `X-Shape` (comma list) and `X-Dtype` headers |
| `traces.py` | Min/max overview, window slicing, low-pass filter |
| `contours.py` | Footprint → polygon outlines |
| `jobs.py` | Queue; one run at a time as a subprocess; progress from `progress.jsonl` |
| `cache.py` | Results computed for older runs, in `~/.cache/caanfast/<run hash>/` |

The run folder is never written by `serve`. `stage_viewer.py` keeps working and imports
the moved functions from `caanfast.serve.frames`.

**Frame rate:** `meta.json` `fps` (existing `io.RawInfo.fps`), else `frameRate` from a
Miniscope `metaData.json` next to the raw videos, else entered by the user in the UI and
kept in the cache. Without a frame rate, time axes show frames and the low-pass is
disabled.

**Contrast of frames:** fixed per run and source from the 1st/99.9th percentiles of
~50 frames spread over the recording, computed once and cached, so the image does not
flicker while scrubbing.

### Pipeline change (new runs)

`pipeline.run` also writes, into the run folder:

- `seeds/init.npy`, `seeds/pnr.npy`, `seeds/merged.npy`: seed positions `(n, 2)` after
  each step.
- `init/A.npz`: initial footprints before CNMF.
- `progress.jsonl`: one JSON line per stage start and end
  (`{"stage": "pass2", "event": "end", "t": <unix s>, "seconds": 498.8}`), written by
  `report.StageTimer`.

Older runs: the seeds and initial units are computed on demand from the saved factors
(as `stage_viewer.py` does, ~2.5 min on 3 h) into the cache.

### Front end, `web/`

- Vite + React + TypeScript.
- Base UI for controls (accessible, unstyled), own CSS for the bento + glass look
  (Geist / Geist Mono, frosted cards over soft colour blobs, dark map tile with glass
  controls).
- uPlot for traces and time series; canvas 2D for frames, footprints, contours, seeds.
- `npm run build` writes to `src/caanfast/serve/static/`, which ships in the wheel;
  users need no Node. In development the Vite dev server proxies `/api` to the back end.

### Access protection

Bound to 127.0.0.1 only. Requests whose `Host` is not `127.0.0.1:<port>` or
`localhost:<port>` are rejected (stops DNS rebinding). Every `/api` request carries the
random token from the opening URL (header `X-Token`; the page stores it in
`sessionStorage`). Another website in the same browser can then neither read runs nor
start jobs.

## 2. Data API

All under `/api/`. JSON for metadata; arrays in the `arrays.py` format. `{id}` is a short
hash of the run folder's absolute path (no paths in URLs).

| Route | Returns |
|---|---|
| `GET /fs?path=` | Subfolders of `path` (folder picker); flags folders that hold AVIs or a run |
| `GET /runs?root=` | Run folders under `root` (have `params.json` + `report.json`) |
| `GET /runs/{id}/summary` | Params, stage timings, seed/unit counts, H/W/T/fps, available parts |
| `GET /runs/{id}/image/{name}` | `max_proj`, `background`, `rank` (per tile), `noise` |
| `GET /runs/{id}/frame/{t}?source=raw\|pre\|recon` | One frame, uint8 `(H, W)` with the run's fixed contrast |
| `GET /runs/{id}/motion` | Per-frame shifts `(T, 2)` and motion quality (from `motion/`) |
| `GET /runs/{id}/motion/residual` | Residual shift and correlation per block and chunk of 32 frames (computed, cached) |
| `GET /runs/{id}/seeds` | Seed positions per step: init, pnr, merged |
| `GET /runs/{id}/units?stage=init\|final` | Per unit: id, centroid, size_px, peak, contour polygons |
| `GET /runs/{id}/traces/overview?kind=C\|YrA&bins=2000` | float32 `(n_units, 2, bins)` min/max |
| `GET /runs/{id}/traces/{unit}?start=&stop=&kind=&lowpass_hz=` | float32 trace slice, optionally low-passed |
| `GET /runs/{id}/zooms` | Suggested windows: 10 brightest units, 5 min windows, 3 per hour |
| `GET /params/defaults` | Every `Params` field: default, type, group, help text |
| `POST /jobs` | Start a run: `raw_dir`, output name, params, `keep_store` → job id |
| `GET /jobs` | Jobs: queued, running, done, failed, cancelled, interrupted |
| `GET /jobs/{id}/events` | Server-sent events: stage start/end, log lines, done/failed |
| `POST /jobs/{id}/cancel` | Stop the run's process group |

### Low-pass filter (`traces.py`)

The low-pass half of CaLab's `BandpassFilter` (`crates/solver/src/filter.rs`), with the
cutoff `f_lp` given directly in Hz instead of derived from τ_rise:

- real FFT of the trace (or of the requested window plus a margin of 5 s on each side,
  cut off after filtering, so window edges match the full-trace result);
- gain 1 for `f < 0.5·f_lp`, `0.5·(1 + cos(π·(f − 0.5·f_lp) / f_lp))` for
  `0.5·f_lp ≤ f < 1.5·f_lp`, 0 above; `f_lp` clamped to Nyquist;
- inverse FFT; traces shorter than 8 samples are returned unchanged.

UI default 5 Hz, slider from 0.5 Hz to Nyquist.

## 3. Pages

### Top area (every run page)

Large dark tile with the cleaned max projection and glass controls over it: source
(max projection / raw / preprocessed / reconstruction), contours on/off, contrast, play
(1×, 4×, 16×) and the scrub bar. Frame `t` and the selected unit are shared by all pages;
clicking a footprint selects that unit everywhere.

### Stage pages

A segmented switcher under the top area; each page a bento grid.

| Page | Tiles |
|---|---|
| Overview | Units, run time, peak memory, seed → unit funnel, stage timings, params changed from the defaults |
| Frames | Raw, preprocessed, reconstruction side by side at frame `t`. No tile store: greyed tile saying to run with `--keep-store` |
| Motion | x/y shifts over time with a playhead tied to `t`, motion quality, residual shift per block |
| Compression | Rank per tile (map + histogram), noise map, rank sum |
| Seeds | Max projection with the seeds of each step (switchable), counts |
| Units | Initial vs final units, unit browser, sortable table (size, peak, position) |
| Traces | Stacked traces of the selected units (overview when zoomed out, full resolution when zoomed in), C / YrA switch, low-pass on/off with a Hz slider, suggested zoom windows as jumps |

### Home

Runs found under a chosen folder, and the jobs list with status and progress.

### New run (launcher)

1. Raw folder via the folder picker; shows AVI count, frames, frame size, fps, size on
   disk.
2. Output name; default `<raw>/../processed/<name>`.
3. Parameters: the common ones first (`workers`, `tophat_wnd`, `merge_dist`,
   `pnr_thres`, `sparse_snr`), "All parameters" for the rest, grouped as in `params.py`,
   each with its help text; `keep_store` switch.

Start queues the run. The job page shows each stage with a tick and its time, the
elapsed time and the last log lines; when done, "Open run" goes to its pages.

## 4. Errors and testing

### Errors

- Missing parts of a run (no tile store, raw drive unplugged, older run without saved
  seeds): the summary lists what exists; the tile is greyed with the reason; never a
  page error.
- Routes that must compute first (seeds and initial units of older runs; for every run
  the frame contrast and the per-block residual shift, `block_residual` on the factors) return
  `202` with progress; the tile polls and shows a bar; the result goes to the cache.
- Bad requests (frame out of range, unknown unit, low-pass without fps) → `4xx` with a
  message.
- Jobs: a failed run shows the tail of its stderr; cancel kills the process group; the
  job list is kept in `~/.cache/caanfast/jobs.json`; a job running when the back end
  stopped is marked `interrupted`.
- Speed: the back end keeps recent frames in an LRU cache; the browser prefetches frames
  around `t` while scrubbing; the browser never holds more than the overview plus one
  window of a 3 h run.

### Testing

Back end (pytest, `TestClient`): a tiny synthetic run (`testing/synth.py`, ~1 min)
built once per session; every route; low-pass (gain 1 below `0.5·f_lp`, 0 above
`1.5·f_lp`, 0.5 at `f_lp`; a sine above the cutoff removed; windowed result equals the
full-trace result inside the window); contours of a known footprint; token and `Host`
rejection; cache (older run → seeds computed once, run folder unchanged); a synthetic job
end to end; cancel. `ruff format` and `ruff check` on all Python.

Front end: TypeScript type check; vitest for array decoding, the frame cache and the
scrubber; one smoke test that loads the built app against the test run and opens every
page.

## Notes

- `rename/caanfast` is not merged yet; this branch starts from it. The uncommitted
  pipeline changes on `fix/rank-vs-T` are not on this branch and are not needed for it.
- The 20 min Miniscope workshop dataset on UCLA Dataverse (doi:10.25346/S6SGHPCZ) is the
  intended public example for the launcher (download button: spec 3).
