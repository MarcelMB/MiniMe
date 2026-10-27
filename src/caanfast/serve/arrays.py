"""Binary array responses: little-endian bytes plus ``X-Shape`` / ``X-Dtype`` headers."""

from __future__ import annotations

import numpy as np
from fastapi import Response

DTYPES = ("uint8", "int32", "float32")


def array_response(a: np.ndarray) -> Response:
    """``a`` as an ``application/octet-stream`` body the front end decodes with the headers."""
    a = np.asarray(a)
    if a.dtype.name not in DTYPES:
        raise TypeError(f"unsupported dtype {a.dtype} (one of {', '.join(DTYPES)})")
    a = np.require(a, dtype=a.dtype.newbyteorder("<"), requirements="C")
    return Response(
        a.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Shape": ",".join(str(n) for n in a.shape), "X-Dtype": a.dtype.name},
    )


def decode(body: bytes, shape: str, dtype: str) -> np.ndarray:
    """Inverse of ``array_response`` (tests and scripts)."""
    dims = tuple(int(n) for n in shape.split(",")) if shape else ()
    return np.frombuffer(body, dtype=np.dtype(dtype).newbyteorder("<")).reshape(dims)
