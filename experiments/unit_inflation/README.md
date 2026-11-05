# Browser unit inflation

Why the browser build finds more units the longer the session, and which of them are real.
Findings and numbers: `docs/unit-inflation.md`.

Inputs → outputs (all under `out/`, footprints only unless noted; main checkout's `.venv`,
`PYTHONPATH=src` from the worktree):

- `classify.py --raw SESSION/raw --browser RUN [--python PYRUN] [--stitch BROWSER_FOLDER]`:
  every unit of a run against the synthetic truth → matched / duplicate / piece /
  out_of_focus / nothing, the true cells only one run found (by category), and for the
  browser's duplicates why the stitch kept them apart → `<tag>_classify.json`.
- `prefix_stitch.py --raw SESSION/raw --run BROWSER_FOLDER --rule RULE`: the stitch of the
  first k windows again (the browser keeps every window's units, `win_NN_*`), so one run gives
  the units-vs-length curve → `<tag>_prefix_<rule>.json`. Rules in `stitch_rules.py`: `base`
  (the stitch until 2026-11-05), `mindistbpD` (the fix: cross-window `min_dist` with
  brightest-parent joins = `stream.stitch(..., min_dist=D)`), `mindistD` (connected
  components, rejected: chains on dense real data), `attachNN` (rejected: joins units ~18 px
  apart on real data).
- `analyse.sh SESSION BROWSER_FOLDER PYRUN TAG`: both of the above for one session.
- `real_v4.py`: real V4 15/30/60 min browser runs re-stitched with and without the fix,
  matched to the Python run (no truth) → `v4_real.json`.
- `score_runs.py RAW RUN ...`: `minime score` (with traces) of run folders, one JSON line each.
- `summarize.py TAG ...`: the markdown tables of the report.
