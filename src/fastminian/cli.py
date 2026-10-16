from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Factor-first CNMF pipeline for one-photon miniscope data."""


@app.command()
def synth(
    out: Path, minutes: float = 10.0, fps: int = 30, segment: int = 1000, seed: int = 0
) -> None:
    """Write a synthetic Miniscope session (N.avi, GREY) and meta.json."""
    from fastminian.testing.synth import write_session

    typer.echo(write_session(out, minutes, fps, segment, seed))
