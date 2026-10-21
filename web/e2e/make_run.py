"""Make the tiny run the smoke test serves (once; reused while web/e2e/.data exists)."""

import sys
from pathlib import Path

from caanfast import pipeline
from caanfast.params import Params
from caanfast.testing import synth


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


if __name__ == "__main__":
    main(Path(sys.argv[1]))
