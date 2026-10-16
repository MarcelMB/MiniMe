"""Pipeline parameters (one flat dataclass, JSON round-trip)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Params:
    # preprocessing + motion
    median_ksize: int = 7
    tophat_wnd: int = 15
    motion_max_shift: int = 20
    motion_down: int = 2
    template_frames: int = 200
    # tiling + compression
    tile: int = 64
    svd_k0: int = 40
    svd_power_iter: int = 1
    rank_rho_min: float = 0.2
    rank_z: float = 4.0
    # seeds
    seeds_wnd: int = 1000
    seeds_stp: int = 500
    seeds_max_wnd: int = 15
    seeds_diff_thres: float = 3
    noise_freq: float = 0.06
    pnr_thres: float | str = "auto"
    merge_dist: int = 10
    merge_corr: float = 0.8
    # initialization
    inita_corr: float = 0.8
    inita_wnd: int = 10
    init_merge_corr: float = 0.8
    loop_merge_corr: float = 0.9
    # CNMF
    dl_wnd: int = 10
    sparse_penal: float = 0.01
    size_thres: tuple[float | None, float | None] = (25, None)
    n_iter: int = 2
    # execution
    workers: int = 1
    sub_batch: int = 100

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def motion_params(self) -> dict[str, Any]:
        """The subset of parameters that invalidates a motion sidecar."""
        return {
            "median_ksize": self.median_ksize,
            "tophat_wnd": self.tophat_wnd,
            "motion_max_shift": self.motion_max_shift,
            "motion_down": self.motion_down,
            "template_frames": self.template_frames,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.asdict(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> Params:
        d = json.loads(Path(path).read_text())
        if "size_thres" in d:
            d["size_thres"] = tuple(d["size_thres"])
        return cls(**d)
