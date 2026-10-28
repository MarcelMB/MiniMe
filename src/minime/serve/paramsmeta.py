"""Parameter groups and help text for the launcher (one entry per ``Params`` field)."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

from minime.params import Params

GROUPS: dict[str, list[str]] = {
    "Preprocessing + motion": [
        "median_ksize",
        "tophat_wnd",
        "motion_max_shift",
        "motion_down",
        "template_frames",
        "motion_before_tophat",
        "motion_bandpass",
    ],
    "Tiling + compression": [
        "tile",
        "svd_k0",
        "svd_power_iter",
        "rank_rho_min",
        "rank_z",
        "noise_frames",
    ],
    "Seeds": [
        "seeds_wnd",
        "seeds_stp",
        "seeds_max_wnd",
        "seeds_diff_thres",
        "noise_freq",
        "pnr_thres",
        "gmm_refine",
        "merge_dist",
        "merge_corr",
    ],
    "Initialization": ["inita_corr", "inita_wnd", "init_merge_corr", "loop_merge_corr"],
    "CNMF": ["dl_wnd", "sparse_snr", "size_thres", "n_iter", "background_clip"],
    "Execution": ["workers", "sub_batch"],
}

COMMON = ["workers", "tophat_wnd", "merge_dist", "pnr_thres", "sparse_snr"]

HELP: dict[str, str] = {
    "median_ksize": "Median filter size in px before motion correction (<= 1 skips it).",
    "tophat_wnd": "Tophat window in px that removes the background glow; about one cell diameter.",
    "motion_max_shift": "Largest motion correction shift in px.",
    "motion_down": "Downsampling factor of the frames used to estimate motion.",
    "template_frames": "Frames averaged into the motion template.",
    "motion_before_tophat": "Estimate motion before the tophat, on band-passed frames.",
    "motion_bandpass": "Difference-of-Gaussians sigmas (px) of the band-pass for motion.",
    "tile": "Tile side in px for the low-rank compression.",
    "svd_k0": "Starting rank of each tile's randomized SVD.",
    "svd_power_iter": "Power iterations of the randomized SVD.",
    "rank_rho_min": "Smallest correlation a component needs to be kept.",
    "rank_z": "z-score a component's energy needs over noise to be kept.",
    "seeds_wnd": "Frames per window when picking seed pixels.",
    "seeds_stp": "Step in frames between seed windows.",
    "seeds_max_wnd": "Window in px of the local-maximum search for seeds.",
    "seeds_diff_thres": "Smallest brightness difference that makes a local maximum a seed.",
    "noise_frames": "Frames used for the per-pixel noise (0 = all); faster, but changes the units.",
    "noise_freq": "Frequency (fraction of the frame rate) above which a trace counts as noise.",
    "pnr_thres": "Peak-to-noise threshold for seeds; 'auto' picks it from the data.",
    "gmm_refine": "Also filter seeds with a 2-component Gaussian mixture (off by default).",
    "merge_dist": "Seeds closer than this (px) can merge.",
    "merge_corr": "Seeds whose traces correlate above this merge.",
    "inita_corr": "Correlation a pixel needs with its seed to join the initial footprint.",
    "inita_wnd": "Window in px around a seed for its initial footprint.",
    "init_merge_corr": "Initial units whose traces correlate above this merge.",
    "loop_merge_corr": "Units whose traces correlate above this merge during CNMF.",
    "dl_wnd": "Dilation window in px that bounds where a footprint may grow.",
    "sparse_snr": "Per-pixel signal-to-noise a pixel needs to stay in a footprint.",
    "size_thres": "Smallest and largest footprint size in px (empty = no limit).",
    "n_iter": "CNMF iterations (spatial + temporal update).",
    "background_clip": "Background of the clipped residual as Minian (about 2x slower).",
    "workers": "Worker processes; about the number of CPU cores minus one or two.",
    "sub_batch": "Frames per pass-1 sub-batch.",
}


def defaults() -> list[dict[str, Any]]:
    d = Params().asdict()
    types = {f.name: str(f.type) for f in fields(Params)}
    return [
        {
            "name": name,
            "group": group,
            "type": types[name],
            "default": json.loads(json.dumps(d[name])),
            "help": HELP[name],
            "common": name in COMMON,
        }
        for group, names in GROUPS.items()
        for name in names
    ]
