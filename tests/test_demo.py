import hashlib
import io
import json

import pytest

from minime import demo


def fake_server(
    monkeypatch: pytest.MonkeyPatch, blobs: dict[str, bytes], md5s: dict, dirs: dict | None = None
) -> list:
    """Serve a dataset listing of ``blobs`` (by filename) and their bytes; log each URL."""
    names = list(blobs)
    listing = {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "directoryLabel": (dirs or {}).get(n),
                        "dataFile": {
                            "id": i,
                            "filename": n,
                            "filesize": len(blobs[n]),
                            "md5": md5s[n],
                        },
                    }
                    for i, n in enumerate(names)
                ]
            }
        }
    }
    calls: list[str] = []

    def urlopen(url: str) -> io.BytesIO:
        calls.append(url)
        if "/api/datasets/" in url:
            return io.BytesIO(json.dumps(listing).encode())
        return io.BytesIO(blobs[names[int(url.rsplit("/", 1)[1])]])

    monkeypatch.setattr(demo.urllib.request, "urlopen", urlopen)
    return calls


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_downloads_only_raw_segments_and_skips_intact(monkeypatch, tmp_path):
    blobs = {"10.avi": b"ten", "2.avi": b"two", "behavior.mp4": b"b", "._0.avi": b"x"}
    calls = fake_server(monkeypatch, blobs, {n: md5(b) for n, b in blobs.items()})
    demo.download(tmp_path, echo=lambda s: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["10.avi", "2.avi"]
    assert (tmp_path / "10.avi").read_bytes() == b"ten"
    calls.clear()
    demo.download(tmp_path, echo=lambda s: None)
    assert len(calls) == 1  # listing only; both segments already intact


def test_md5_mismatch_leaves_no_file(monkeypatch, tmp_path):
    fake_server(monkeypatch, {"0.avi": b"data"}, {"0.avi": md5(b"other")})
    with pytest.raises(OSError, match="MD5"):
        demo.download(tmp_path, echo=lambda s: None)
    assert list(tmp_path.iterdir()) == []


def test_grin_puts_segments_in_raw_and_truth_files_next_to_it(monkeypatch, tmp_path):
    blobs = {"0.avi": b"a", "1.avi": b"b", "meta.json": b"{}", "truth.npz": b"t", "README.md": b"r"}
    dirs = {"0.avi": "raw", "1.avi": "raw"}
    calls = fake_server(monkeypatch, blobs, {n: md5(b) for n, b in blobs.items()}, dirs)
    raw = demo.download(tmp_path / "raw", echo=lambda s: None, dataset="grin")
    assert demo.DATASETS["grin"][0] in calls[0]
    assert sorted(p.name for p in raw.iterdir()) == ["0.avi", "1.avi"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "README.md",
        "meta.json",
        "raw",
        "truth.npz",
    ]


def test_v4_ignores_top_level_extras(monkeypatch, tmp_path):
    blobs = {"0.avi": b"a", "meta.json": b"{}"}
    fake_server(monkeypatch, blobs, {n: md5(b) for n, b in blobs.items()})
    demo.download(tmp_path / "raw", echo=lambda s: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw"]
