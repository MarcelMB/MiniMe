"""Pipeline parameters (one flat dataclass, JSON round-trip)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# keys a run's params.json carries besides the parameters; ignored when it is read back
# older runs: cafast_version / fastminian_version
RUN_METADATA_KEYS = frozenset(
    {"raw_dir", "raw_hash", "caanfast_version", "cafast_version", "fastminian_version"}
)


@dataclass
class Params:
    # preprocessing + motion (median_ksize <= 1 / tophat_wnd <= 0 skip that step)
    median_ksize: int = 7
    tophat_wnd: int = 15
    motion_max_shift: int = 20
    motion_down: int = 2
    template_frames: int = 200
    # estimate motion on band-passed denoised frames (DoG sigmas in px), then warp, then
    # tophat: the tophat removes the static structure (vessels) that registration needs, and
    # on real data (WL25) a fixed template on tophat frames went stale (bench/README.md)
    motion_before_tophat: bool = True
    motion_bandpass: tuple[float, float] = (1.0, 6.0)
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
    # off by default: Minian's reference run skipped it, and on a session whose cells share one
    # amplitude distribution its 2-GMM splits real cells into bright/dim (bench/README.md)
    gmm_refine: bool = False
    merge_dist: int = 10
    merge_corr: float = 0.8
    # initialization
    inita_corr: float = 0.8
    inita_wnd: int = 10
    init_merge_corr: float = 0.8
    loop_merge_corr: float = 0.9
    # CNMF
    dl_wnd: int = 10
    # sparseness penalty of the spatial update as a per-pixel signal-to-noise threshold: a
    # pixel keeps a unit only where its std along the unit's trace exceeds sparse_snr x sn.
    # Minian's sparse_penal is sparse_snr / sqrt(T) (0.01 on 10,000 frames = sparse_snr 1),
    # so Minian's penalty grows with the length of the recording; this one does not
    sparse_snr: float = 1.0
    size_thres: tuple[float | None, float | None] = (25, None)
    n_iter: int = 2
    # background of the unclipped residual Y - A C, computed on the factors without
    # reconstructing the movie (False), or of clip(Y - A C, 0) as Minian (True, ~2x slower
    # cnmf); same scores on synth, WL25, WL27 (bench/README.md)
    background_clip: bool = False
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
            "motion_before_tophat": self.motion_before_tophat,
            "motion_bandpass": self.motion_bandpass,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.asdict(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> Params:
        """Parameters from a JSON file: a partial set, or a run's ``params.json``.

        Run metadata (``RUN_METADATA_KEYS``) is ignored; any other unknown key raises.
        """
        d = dict(json.loads(Path(path).read_text()))
        for k in RUN_METADATA_KEYS:
            d.pop(k, None)
        if "sparse_penal" in d:
            raise ValueError(
                f"{path}: sparse_penal was replaced by sparse_snr = sparse_penal * sqrt(T) "
                "(T = frames of the session it was tuned on)"
            )
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown parameter(s) in {path}: {', '.join(unknown)}")
        for k in ("size_thres", "motion_bandpass"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)
