# TODO

Last updated: 2026-10-27

## Now
- [ ] Add a LICENSE: GPL-3.0, since parts are ported from Minian (GPL-3.0). Needed before the repo goes public or to the lab.
- [ ] Auto-tune: rebuild factors, then fix the search (Obsidian "MiniMe auto-tuning"). `src/minime/autotune/`
- [ ] Recheck the auto-tune `sparse_snr` grid (0.4–4.0, ported from `sparse_penal`) on WL25. `src/minime/autotune/sweep.py`
- [ ] Minian 3 h overnight run for the comparison (command in the Obsidian note). `bench/minian_full.py`

## Bugs

## Ideas
- Orbit-style motion metrics (ECC, correlation to the template) in `report.json` and the viewer.
- Accept/reject column in the unit browser, saved next to `units.csv`.
- Distribution testing: Windows, Linux, Intel and Apple-silicon macOS, low-RAM laptops; CI matrix before release.
- Pass-2 speed-ups still to test: no rSVD restarts, faster reads. `bench/pass2_variants.py`

## Parked
- cala 3 h comparison. Parked because: ~36x slower, the run died at 37k frames. Revisit when: a rerun with periodic export is worth it (`bench/cala_full.py`).

## Done
- [x] Merge auto-tune, the tuning viewer and `serve` into main; rename to MiniMe; private repo MarcelMB/MiniMe (2026-10-27)
