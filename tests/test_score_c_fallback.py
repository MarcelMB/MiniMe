"""A run folder with cnmf/C.npy and no YrA.npy (a browser run) scores like one with YrA.npy."""

import json
import shutil
from pathlib import Path

from minime import score


def test_score_reads_c_when_yra_is_missing(tiny_run: Path, tmp_path: Path) -> None:
    run = tmp_path / "t"
    shutil.copytree(tiny_run, run)
    want = score.score_pipeline_dir(run)
    (run / "cnmf" / "C.npy").unlink(missing_ok=True)
    (run / "cnmf" / "YrA.npy").rename(run / "cnmf" / "C.npy")
    got = score.score_pipeline_dir(run)
    assert json.dumps(got, sort_keys=True, default=str) == json.dumps(
        want, sort_keys=True, default=str
    )
