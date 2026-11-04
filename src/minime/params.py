"""Pipeline parameters (one flat dataclass, JSON round-trip)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# keys a run's params.json carries besides the parameters; ignored when it is read back
# older runs: caanfast_version / cafast_version / fastminian_version
RUN_METADATA_KEYS = frozenset(
    {
        "raw_dir",
        "raw_hash",
        "fps",
        "minime_version",
        "caanfast_version",
        "cafast_version",
        "fastminian_version",
        "source_run",
    }
)
# motion prototypes: in a sidecar's parameters only when not at their default, so sidecars
# written before they existed stay valid
_OFF: dict[str, Any] = {
    "template_source": "first",
    "template_chunks": 10,
    "template_recentre": False,
    "motion_refine": False,
    "motion_outlier": 0.0,
    "motion_fill": "zero",
}
NEW_MOTION_PARAMS = tuple(_OFF)


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
    # motion template (prototypes, defaults = the first-frames template): "first" = the
    # first template_frames frames of segment 0; "spread" = template_frames frames in
    # template_chunks runs spread evenly over the session, aligned and put at their median
    # position. template_recentre moves the "first" template to the median position of
    # template_chunks probe frames spread over the session
    template_source: str = "first"
    template_chunks: int = 10
    template_recentre: bool = False
    # re-estimate each sub-batch against its own mean (tied back to the template)
    motion_refine: bool = False
    # px: replace a shift this far from its neighbours' when its correlation peak is weak
    # (black or corrupted frame); 0 = off
    motion_outlier: float = 0.0
    # what the warp puts where a shifted frame has no data: "zero", "replicate" (the
    # nearest edge pixel) or "template" (the template frames' mean there): a zero band that
    # comes and goes with the motion is an on/off signal that pass 2 / CNMF turn into units
    motion_fill: str = "zero"
    # pixels the motion warp reads from inside the frame in fewer than this fraction of the
    # frames (zero fill otherwise) are set to 0 before pass 2: their on/off steps would
    # otherwise become band-shaped units along the border. 0 = off
    border_coverage: float = 0.0
    # tiling + compression
    tile: int = 64
    svd_k0: int = 40
    svd_power_iter: int = 1
    rank_rho_min: float = 0.2
    rank_z: float = 4.0
    # pass 2 compresses a tile of a longer session in time windows of at most this many frames
    # (each read once) and merges their factors, so a worker holds one window, not the whole
    # tile (24 h at 600 x 600: ~6 GB per tile); 0 = whole tile. Shorter sessions are unchanged
    svd_window: int = 144_000
    # per-pixel noise from this many frames (contiguous 1000-frame segments spread over the
    # session) instead of all of them; 0 = all frames (Minian). Pass 2 27 % faster, but it
    # changes the units (near-black pixels; bench/README.md), so off by default
    noise_frames: int = 0
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
    # units whose centroids are closer than this (px) merge whatever their traces do: two cells
    # cannot sit that close, only a split one can. 0 = off; auto-tune sets 0.4 x cell diameter
    merge_min_dist: float = 0.0
    # CNMF (dl_wnd < 0: no growth limit, every unit may take every pixel; dense, K x P)
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

    def __post_init__(self) -> None:
        # motion._prep only knows 2x downsampling, but estimate scales shifts by motion_down
        if self.motion_down not in (1, 2):
            raise ValueError(f"motion_down must be 1 or 2, not {self.motion_down!r}")
        if self.motion_fill not in ("zero", "replicate", "template"):
            raise ValueError(
                f"motion_fill must be 'zero', 'replicate' or 'template', not {self.motion_fill!r}"
            )
        if self.template_source not in ("first", "spread"):
            raise ValueError(
                f"template_source must be 'first' or 'spread', not {self.template_source!r}"
            )

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
            **{k: getattr(self, k) for k in NEW_MOTION_PARAMS if getattr(self, k) != _OFF[k]},
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.asdict(), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path, base: Params | None = None) -> Params:
        """Parameters from a JSON file: a partial set, or a run's ``params.json``.

        Keys the file leaves out come from ``base`` (default: the built-in defaults).
        Run metadata (``RUN_METADATA_KEYS``) is ignored; any other unknown key raises.
        """
        text = Path(path).read_text(encoding="utf-8-sig")  # a BOM from Windows Notepad is fine
        return cls.from_dict(json.loads(text), base, source=str(path))

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], base: Params | None = None, source: str = "params"
    ) -> Params:
        """Like ``from_json`` for an already parsed mapping."""
        d = dict(d)
        for k in RUN_METADATA_KEYS:
            d.pop(k, None)
        if "sparse_penal" in d:
            raise ValueError(
                f"{source}: sparse_penal was replaced by sparse_snr = sparse_penal * sqrt(T) "
                "(T = frames of the session it was tuned on)"
            )
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown parameter(s) in {source}: {', '.join(unknown)}")
        for k in ("size_thres", "motion_bandpass"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{**(base.asdict() if base is not None else {}), **d})
