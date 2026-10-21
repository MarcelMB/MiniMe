import numpy as np
import pytest

from caanfast.serve.arrays import array_response, decode


def test_array_roundtrip_keeps_shape_and_dtype():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    r = array_response(a)
    assert r.headers["X-Shape"] == "3,4"
    assert r.headers["X-Dtype"] == "float32"
    assert r.media_type == "application/octet-stream"
    np.testing.assert_array_equal(decode(r.body, "3,4", "float32"), a)


def test_array_response_makes_non_contiguous_input_contiguous():
    a = np.arange(20, dtype=np.uint8).reshape(4, 5)[:, ::2]
    r = array_response(a)
    np.testing.assert_array_equal(decode(r.body, r.headers["X-Shape"], "uint8"), a)


def test_empty_array_has_its_shape():
    r = array_response(np.zeros((0, 2), np.float32))
    assert r.headers["X-Shape"] == "0,2" and r.body == b""


def test_array_response_rejects_float64():
    with pytest.raises(TypeError):
        array_response(np.zeros(3))
