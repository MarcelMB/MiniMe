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
    name: Annotated[str, typer.Option(help="Run name under ../processed/.")] = "cafast",
    keep_store: Annotated[bool, typer.Option(help="Keep the temporary tile store.")] = False,
) -> None:
    """Run the pipeline on RAW_DIR; outputs go to RAW_DIR/../processed/NAME/."""
    import json

    from cafast import pipeline
    from cafast.params import Params

    prm = Params.from_json(params) if params is not None else Params()
    out = pipeline.run(raw_dir, prm, run_name=name, keep_store=keep_store, workers=workers)
    counts = json.loads((out / "report.json").read_text())["counts"]
    typer.echo(str(out))
    typer.echo(json.dumps(counts, indent=1))


@app.command()
def cnmf(
    run_dir: Annotated[Path, typer.Argument(help="A finished run (holds factors/).")],
    params: Annotated[
        Path | None, typer.Option(help="Params JSON; keys it leaves out come from the run.")
    ] = None,
    name: Annotated[str, typer.Option(help="Output name under RUN_DIR/reruns/.")] = "rerun",
) -> None:
    """Rerun window 2 (seeds, init, CNMF) from RUN_DIR's factors into RUN_DIR/reruns/NAME/."""
    import json

    from cafast import window2
    from cafast.autotune.run import check_not_reserved

    try:
        check_not_reserved(name)
        out = window2.rerun(run_dir, window2.run_params(run_dir, params), name=name)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    counts = json.loads((out / "report.json").read_text())["counts"]
    typer.echo(str(out))
    typer.echo(json.dumps(counts, indent=1))


@app.command()
def autotune(
    run_dir: Annotated[Path, typer.Argument(help="A finished run (holds factors/).")],
    diameter: Annotated[float, typer.Option(help="Cell diameter in px (factor frame).")],
    name: Annotated[str, typer.Option(help="Output name under RUN_DIR/reruns/.")] = "auto",
) -> None:
    """Propose window-2 parameters for RUN_DIR (sweep, then a rerun in RUN_DIR/reruns/NAME/)."""
    import json

    from cafast.autotune import run as at
    from cafast.autotune.sweep import Trial

    def show(t: Trial) -> None:
        res = (
            t.error if t.score is None else f"good {t.score['good']}  flagged {t.score['flagged']}"
        )
        typer.echo(f"{t.step:<10} {res}  ({t.seconds:.1f} s)")

    try:
        out = at.autotune(run_dir, diameter, name=name, on_trial=show)
    except (ValueError, RuntimeError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    doc = json.loads((out / at.DOC).read_text())
    typer.echo(str(out))
    for k, v in doc["proposed"].items():
        edge = "  [at edge]" if k in doc["at_edge"] else ""
        typer.echo(f"{k:<17} {doc['run_values'][k]!s:>8} -> {v!s:<8}{edge}")
    s = doc["score"]
    typer.echo(f"good {s['good']}  flagged {s['flagged']}  ({doc['wall_s']} s)")


@app.command()
def tune(
    run_dir: Annotated[Path, typer.Argument(help="A finished run (holds factors/).")],
    minian: Annotated[
        Path | None, typer.Option(help="Minian result npz (A, C, max_proj) to overlay.")
    ] = None,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1.")] = 8765,
    browser: Annotated[bool, typer.Option(help="Open the page in the default browser.")] = True,
) -> None:
    """Serve a localhost page to rerun window 2 of RUN_DIR and compare every stage."""
    import webbrowser

    from cafast.tune import server

    app_ = server.TuneApp(run_dir, minian_npz=minian)
    httpd = server.make_server(app_, port=port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    typer.echo(f"cafast tune: {url}  (Ctrl-C to stop)")
    if browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


@app.command()
def synth(
    out: Path, minutes: float = 10.0, fps: int = 30, segment: int = 1000, seed: int = 0
) -> None:
    """Write a synthetic Miniscope session (N.avi, GREY) and meta.json."""
    from cafast.testing.synth import write_session

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
