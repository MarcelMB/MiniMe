"""Make the tiny run the smoke test serves (once; reused while web/e2e/.data exists)."""

import dataclasses
import sys
from pathlib import Path

from minime import pipeline
from minime.params import Params
from minime.testing import synth


def main(root: Path) -> None:
    if (root / "processed" / "t" / "report.json").exists():
        return
    raw = root / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    params = Params(
        tile=32,
        seeds_wnd=100,
        seeds_stp=50,
        seeds_max_wnd=8,
        template_frames=50,
        inita_wnd=8,
        dl_wnd=5,
        size_thres=(9, None),
        svd_k0=16,
        motion_before_tophat=False,
    )
    pipeline.run(raw, params, run_name="t", workers=1, keep_store=True)
    add_rerun_and_autotune(root / "processed" / "t")


def add_rerun_and_autotune(run: Path) -> None:
    """A rerun `r1` and an auto-tune (`reruns/auto`) for the Tune page's contract requests."""
    from minime import window2
    from minime.autotune import run as at
    from minime.autotune import sweep

    window2.rerun(run, dataclasses.replace(window2.run_params(run), sparse_snr=0.5), name="r1")
    small = sweep.Grids(
        seeds_diff_thres=(2, 3),
        pnr_thres=("auto", 1.0),
        merge_corr=(0.8,),
        init_merge_corr=(0.8,),
        inita_corr=(0.8,),
        sparse_snr=(0.5, 1.0),
        loop_merge_corr=(0.9,),
        n_confirm=2,
        max_extend=0,
    )
    at.autotune(run, 7.0, grids=small, strategy="grid")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
