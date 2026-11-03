"""Tiny .npz files the browser's zip and npz readers are tested on (scipy and numpy writers)."""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    a = sps.csr_matrix(np.array([[0, 1.5, 0, 2], [0, 0, 0, 0], [3, 0, 0.25, 0]], np.float32))
    sps.save_npz(out / "csr_compressed.npz", a)  # deflated, as minime's cnmf/A.npz
    sps.save_npz(out / "csr_stored.npz", a, compressed=False)
    np.savez(
        out / "arrays.npz",
        shifts=np.arange(6, dtype=np.float32).reshape(3, 2),
        q=np.array([0.5, 1.0]),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
