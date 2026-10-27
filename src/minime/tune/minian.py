"""A Minian result aligned and matched to a minime run (reuses ``bench/compare_minian.py``).

``MINIAN_NPZ`` holds ``A (K, H*W)``, ``C (K, T)`` and ``max_proj (H, W)``. Minian's
coordinates are moved onto minime's by phase-correlating the two max projections; units are
matched one-to-one by centroid within 5 px and matched traces correlated.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps

if TYPE_CHECKING:
    from minime.tune.figures import RunData

_REPO = Path(__file__).resolve().parents[3]  # bench/ is not packaged


def _bench() -> ModuleType:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from bench import compare_minian

    return compare_minian


@dataclass
class MinianOverlay:
    A: sps.csr_matrix  # (K, P) in Minian's own coordinates
    C: np.ndarray  # (K, T)
    max_proj: np.ndarray
    offset: np.ndarray  # (dy, dx) added to Minian coordinates to land on minime's
    centroids: np.ndarray  # (K, 2) in minime coordinates
    ui: np.ndarray  # matched minime unit indices
    mi: np.ndarray  # matched Minian unit indices
    corr: np.ndarray  # trace correlation of each matched pair
    dist: np.ndarray  # centroid distance of each matched pair (px)
    size10: np.ndarray  # (K,) Minian footprint pixels above 10 % of peak


def load(npz: Path, run: RunData) -> MinianOverlay:
    """Minian's units from ``npz``, aligned and matched to ``run``."""
    cm = _bench()
    mn = cm.load_minian(Path(npz))
    res = cm.compare(
        {
            "A": run.A,
            "C": run.YrA,
            "max_proj": run.max_proj.astype(np.float32),
            "H": run.H,
            "W": run.W,
        },
        mn,
    )
    return MinianOverlay(
        A=mn["A"],
        C=mn["C"],
        max_proj=mn["max_proj"],
        offset=np.asarray(res["offset_px"], float),
        centroids=res["minian_centroids"],
        ui=res["ui"],
        mi=res["mi"],
        corr=res["corr"],
        dist=res["dist"],
        size10=res["size10"][1],
    )
