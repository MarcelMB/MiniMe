# fastminian

Factor-first CNMF pipeline for one-photon miniscope calcium imaging.

Status: scaffold. Currently provides a synthetic session generator
(`fastminian synth`) with ground truth, used for tests and benchmarks.

    uv venv -p 3.11 && uv pip install -e ".[dev]"
    .venv/bin/pytest
    .venv/bin/fastminian synth <out>/raw --minutes 10

Input: nothing. Output: `N.avi` segments (GREY) in `<out>` and `meta.json` in its parent.
