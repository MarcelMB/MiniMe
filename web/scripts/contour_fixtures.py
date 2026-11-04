"""cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) on random masks, for contours.test.ts.

Run from web/ with the repository's Python (PYTHONPATH=../src), then format the output the way
it is committed:

    python scripts/contour_fixtures.py src/browser/__fixtures__/contours.json
    npx biome format --write src/browser/__fixtures__/contours.json
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np


def contours(m: np.ndarray) -> list[list[list[int]]]:
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :].tolist() for c in cs]


def ring_with_island() -> np.ndarray:
    """A ring around a hole that holds a one-pixel island: the island is not an outer border."""
    m = np.zeros((9, 9), np.uint8)
    m[1:8, 1:8] = 1
    m[2:7, 2:7] = 0
    m[4, 4] = 1
    return m


def main(out: str) -> None:
    rng = np.random.default_rng(0)
    masks = []
    for h, w, p in [(1, 1, 1.0), (5, 5, 0.3), (12, 9, 0.5), (16, 16, 0.2), (20, 31, 0.6)]:
        masks.append((rng.random((h, w)) < p).astype(np.uint8))
    # more variety (drawn after the five above, so those stay as they were)
    for h, w, p in [(1, 12, 0.7), (11, 1, 0.7), (30, 30, 0.45), (25, 40, 0.55), (40, 25, 0.7)]:
        masks.append((rng.random((h, w)) < p).astype(np.uint8))
    masks += [ring_with_island(), np.ones((4, 6), np.uint8), np.zeros((3, 3), np.uint8)]
    cases = [
        {"h": m.shape[0], "w": m.shape[1], "mask": m.ravel().tolist(), "contours": contours(m)}
        for m in masks
    ]
    with open(out, "w") as f:
        json.dump(cases, f)


if __name__ == "__main__":
    main(sys.argv[1])
