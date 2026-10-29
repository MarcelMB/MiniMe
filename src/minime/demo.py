"""Download the demo session's raw N.avi segments from UCLA Dataverse."""

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

DOI = "doi:10.25346/S6SGHPCZ"
SERVER = "https://dataverse.ucla.edu"


def raw_files(dataset: dict) -> list[dict]:
    """The ``N.avi`` dataFile records of a Dataverse dataset JSON, sorted by integer stem."""
    files = [f["dataFile"] for f in dataset["data"]["latestVersion"]["files"]]
    avis = [f for f in files if f["filename"].endswith(".avi") and f["filename"][:-4].isdigit()]
    return sorted(avis, key=lambda f: int(f["filename"][:-4]))


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path, echo: Callable[[str], None] = print) -> Path:
    """Fetch every raw segment into ``dest``; files whose MD5 already matches are skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{SERVER}/api/datasets/:persistentId/?persistentId={DOI}") as r:
        files = raw_files(json.load(r))
    for i, f in enumerate(files, 1):
        out = dest / f["filename"]
        tag = f"[{i}/{len(files)}] {f['filename']} ({f['filesize'] / 1e6:.0f} MB)"
        if out.exists() and md5_of(out) == f["md5"]:
            echo(f"{tag} already there")
            continue
        echo(f"{tag} downloading")
        part = out.with_name(out.name + ".part")  # a broken download never looks complete
        url = f"{SERVER}/api/access/datafile/{f['id']}"
        with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        if md5_of(part) != f["md5"]:
            part.unlink()
            raise OSError(f"MD5 mismatch for {f['filename']}; run the command again")
        part.replace(out)
    return dest
