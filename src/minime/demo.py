"""Download a demo session (raw N.avi segments, plus truth files for synthetic ones) from
UCLA Dataverse."""

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

SERVER = "https://dataverse.ucla.edu"
# name -> (DOI, top-level files that go next to raw/, i.e. into the session folder)
DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    # real V4 CA1 open-field recording (Miniscope workshop example), 24 segments, ~8.5 GB
    "v4": ("doi:10.25346/S6SGHPCZ", ()),
    # synthetic V4 through a GRIN lens with ground truth (minime synth --scenario grin_v4),
    # 36 segments, ~13 GB; meta.json and truth.npz make `minime score` work
    "grin": ("doi:10.25346/S6WHMSXK", ("README.md", "meta.json", "truth.npz")),
}
DOI = DATASETS["v4"][0]


def _files(dataset: dict) -> list[dict]:
    return [
        {**f["dataFile"], "dir": f.get("directoryLabel") or ""}
        for f in dataset["data"]["latestVersion"]["files"]
    ]


def raw_files(dataset: dict) -> list[dict]:
    """The ``N.avi`` dataFile records of a Dataverse dataset JSON, sorted by integer stem."""
    avis = [f for f in _files(dataset) if f["filename"].endswith(".avi")]
    avis = [f for f in avis if f["filename"][:-4].isdigit()]
    return sorted(avis, key=lambda f: int(f["filename"][:-4]))


def extra_files(dataset: dict, names: tuple[str, ...]) -> list[dict]:
    """The top-level (no folder) dataFile records named in ``names``."""
    return [f for f in _files(dataset) if f["dir"] == "" and f["filename"] in names]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(f: dict, out: Path, tag: str, echo: Callable[[str], None]) -> None:
    """Download one dataFile to ``out`` unless an intact copy (same MD5) is there."""
    if out.exists() and md5_of(out) == f["md5"]:
        echo(f"{tag} already there")
        return
    echo(f"{tag} downloading")
    part = out.with_name(out.name + ".part")  # a broken download never looks complete
    url = f"{SERVER}/api/access/datafile/{f['id']}"
    with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    if md5_of(part) != f["md5"]:
        part.unlink()
        raise OSError(f"MD5 mismatch for {f['filename']}; run the command again")
    part.replace(out)


def download(dest: Path, echo: Callable[[str], None] = print, dataset: str = "v4") -> Path:
    """Fetch every raw segment of ``dataset`` into ``dest`` and its extra files into
    ``dest.parent``; files whose MD5 already matches are skipped. Returns ``dest``."""
    doi, extras = DATASETS[dataset]
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{SERVER}/api/datasets/:persistentId/?persistentId={doi}") as r:
        listing = json.load(r)
    todo = [(f, dest / f["filename"]) for f in raw_files(listing)]
    todo += [(f, dest.parent / f["filename"]) for f in extra_files(listing, extras)]
    for i, (f, out) in enumerate(todo, 1):
        _fetch(f, out, f"[{i}/{len(todo)}] {f['filename']} ({f['filesize'] / 1e6:.0f} MB)", echo)
    return dest
