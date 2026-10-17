from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Factor-first CNMF pipeline for one-photon miniscope data."""


@app.command()
def run(
    raw_dir: Annotated[Path, typer.Argument(help="Folder of N.avi segments (read-only).")],
    params: Annotated[Path | None, typer.Option(help="Params JSON (default: built-in).")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes (default: params).")] = None,
    name: Annotated[str, typer.Option(help="Run name under ../processed/.")] = "fastminian",
    keep_store: Annotated[bool, typer.Option(help="Keep the temporary tile store.")] = False,
) -> None:
    """Run the pipeline on RAW_DIR; outputs go to RAW_DIR/../processed/NAME/."""
    import json

    from fastminian import pipeline
    from fastminian.params import Params

    prm = Params.from_json(params) if params is not None else Params()
    out = pipeline.run(raw_dir, prm, run_name=name, keep_store=keep_store, workers=workers)
    counts = json.loads((out / "report.json").read_text())["counts"]
    typer.echo(str(out))
    typer.echo(json.dumps(counts, indent=1))


@app.command()
def synth(
    out: Path, minutes: float = 10.0, fps: int = 30, segment: int = 1000, seed: int = 0
) -> None:
    """Write a synthetic Miniscope session (N.avi, GREY) and meta.json."""
    from fastminian.testing.synth import write_session

    typer.echo(write_session(out, minutes, fps, segment, seed))


@app.command()
def score(
    run_dir: Annotated[Path | None, typer.Argument(help="Pipeline run directory.")] = None,
    raw: Annotated[Path | None, typer.Option(help="Raw dir (meta.json is in its parent).")] = None,
    minian: Annotated[Path | None, typer.Option(help="minian_units.npz to score.")] = None,
) -> None:
    """Score a pipeline run (or a Minian export) against the synthetic ground truth."""
    import json
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2])
    )  # repo root: bench/ is not packaged
    from bench import score as bench_score

    if minian is not None:
        if raw is None:
            raise typer.BadParameter("--minian needs --raw")
        result = bench_score.score_minian_npz(minian, raw)
        dest = minian.parent / (
            "score.json" if minian.stem == "minian_units" else f"score_{minian.stem}.json"
        )
    elif run_dir is not None:
        result = bench_score.score_pipeline_dir(run_dir, raw)
        dest = run_dir / "score.json"
    else:
        raise typer.BadParameter("give RUN_DIR or --minian")
    text = json.dumps(result, indent=2)
    dest.write_text(text + "\n")
    typer.echo(text)
