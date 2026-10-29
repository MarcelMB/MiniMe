import json

import pytest
from typer.testing import CliRunner

from minime import cli, places


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIME_CACHE", str(tmp_path / "cache"))
    return tmp_path


def test_remember_puts_newest_first_without_duplicates(cache):
    a, b = cache / "a", cache / "b"
    (a / "raw").mkdir(parents=True)
    b.mkdir()
    places.remember(a, "demo", a / "raw")
    places.remember(b, "run")
    places.remember(a, "run", a / "raw")
    got = places.known()
    assert [(p["path"], p["kind"]) for p in got] == [(str(a), "run"), (str(b), "run")]
    assert got[0]["raw"] == str(a / "raw") and "raw" not in got[1]


def test_known_drops_missing_folders_and_survives_a_broken_file(cache):
    gone = cache / "gone"
    gone.mkdir()
    places.remember(gone, "run")
    gone.rmdir()
    assert places.known() == []
    places.places_file().write_text("not json", encoding="utf-8")
    assert places.known() == []
    places.remember(cache, "run")  # a broken file is replaced, not fatal
    assert [p["path"] for p in places.known()] == [str(cache)]


def test_list_is_capped(cache):
    for i in range(places.MAX_PLACES + 5):
        (cache / str(i)).mkdir()
        places.remember(cache / str(i), "run")
    assert len(json.loads(places.places_file().read_text(encoding="utf-8"))) == places.MAX_PLACES


def test_a_finished_run_remembers_its_session_folder(tiny_run):
    # tiny_run wrote into the test cache that conftest points MINIME_CACHE at
    session = tiny_run.parent.parent
    hit = [p for p in places.known() if p["path"] == str(session)]
    assert hit and hit[0]["kind"] == "run" and hit[0]["raw"] == str(session / "raw")


def test_demo_command_remembers_the_folder(cache, monkeypatch):
    monkeypatch.setattr("minime.demo.download", lambda dest, echo, dataset: dest)
    (cache / "demo" / "raw").mkdir(parents=True)
    r = CliRunner().invoke(cli.app, ["demo", str(cache / "demo"), "--dataset", "grin"])
    assert r.exit_code == 0, r.output
    assert "minime score" in r.output
    first = places.known()[0]
    assert (first["path"], first["kind"], first["raw"], first["dataset"]) == (
        str(cache / "demo"),
        "demo",
        str(cache / "demo" / "raw"),
        "grin",
    )
    bad = CliRunner().invoke(cli.app, ["demo", str(cache / "demo"), "--dataset", "nope"])
    assert bad.exit_code != 0


def test_folders_route_lists_known_folders(client, cache):
    places.remember(cache, "demo")
    r = client.get("/api/folders")
    assert r.status_code == 200
    assert r.json()["folders"][0]["path"] == str(cache)
