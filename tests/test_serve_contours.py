import numpy as np
import scipy.sparse as sps

from caanfast.serve.contours import unit_polygons


def _footprints(H, W, masks):
    rows = [m.astype(np.float32).ravel() for m in masks]
    return sps.csr_matrix(np.stack(rows))


def test_square_footprint_gives_one_polygon_on_its_border():
    H, W = 32, 40
    m = np.zeros((H, W))
    m[10:15, 20:25] = 1.0
    m[12, 22] = 3.0  # the peak; everything else is above 0.2 x peak
    (polys,) = unit_polygons(_footprints(H, W, [m]), H, W)
    assert len(polys) == 1
    pts = np.array(polys[0])
    assert (pts[:, 0].min(), pts[:, 0].max()) == (20, 24)
    assert (pts[:, 1].min(), pts[:, 1].max()) == (10, 14)


def test_two_islands_and_an_empty_unit():
    H, W = 20, 20
    a = np.zeros((H, W))
    a[2:5, 2:5] = 1.0
    a[12:15, 12:15] = 1.0
    out = unit_polygons(_footprints(H, W, [a, np.zeros((H, W))]), H, W)
    assert len(out[0]) == 2 and out[1] == []


def test_pixels_below_rel_of_the_peak_are_outside():
    H, W = 20, 20
    a = np.zeros((H, W))
    a[5:10, 5:10] = 0.1
    a[7, 7] = 1.0
    (polys,) = unit_polygons(_footprints(H, W, [a]), H, W, rel=0.2)
    assert np.array(polys[0]).tolist() == [[7.0, 7.0]]
