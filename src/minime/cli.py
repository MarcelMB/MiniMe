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
    name: Annotated[str, typer.Option(help="Run name under ../processed/.")] = "minime",
    keep_store: Annotated[bool, typer.Option(help="Keep the temporary tile store.")] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing run of this name, its reruns and tune files.")
    ] = False,
) -> None:
    """Run the pipeline on RAW_DIR; outputs go to RAW_DIR/../processed/NAME/."""
    import json

    from minime import pipeline
    from minime.params import Params

    out = pipeline.out_dir_for(raw_dir, name)
    if not overwrite and out.is_dir() and any(out.iterdir()):
        typer.echo(
            f"error: {out} already exists; choose another --name or pass --overwrite", err=True
        )
        raise typer.Exit(1)
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

    from minime import window2
    from minime.autotune.run import check_not_reserved

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
    diameter: Annotated[
        float | None,
        typer.Option(
            help="Cell diameter in px (factor frame); default: from the run's footprints."
        ),
    ] = None,
    name: Annotated[str, typer.Option(help="Output name under RUN_DIR/reruns/.")] = "auto",
    dup_corr: Annotated[
        float | None,
        typer.Option(help="Trace corr above which a neighbour within D is a duplicate."),
    ] = None,
    min_dist_frac: Annotated[
        float | None, typer.Option(help="Units closer than this x D always merge.")
    ] = None,
) -> None:
    """Propose window-2 parameters for RUN_DIR (sweep, then a rerun in RUN_DIR/reruns/NAME/)."""
    import json

    from minime.autotune import run as at
    from minime.autotune.sweep import MIN_DIST_FRAC, Trial

    dup_corr = at.DUP_CORR if dup_corr is None else dup_corr
    min_dist_frac = MIN_DIST_FRAC if min_dist_frac is None else min_dist_frac

    def show(t: Trial) -> None:
        res = (
            t.error if t.score is None else f"good {t.score['good']}  flagged {t.score['flagged']}"
        )
        typer.echo(f"{t.step:<10} {res}  ({t.seconds:.1f} s)")

    try:
        if diameter is None:
            diameter = at.estimate_diameter(run_dir)
            typer.echo(f"diameter {diameter} px, estimated from the run's footprints")
        out = at.autotune(
            run_dir,
            diameter,
            name=name,
            on_trial=show,
            dup_corr=dup_corr,
            min_dist_frac=min_dist_frac,
        )
    except (ValueError, RuntimeError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    doc = json.loads((out / at.DOC).read_text())
    typer.echo(str(out))
    typer.echo(f"dup_corr {dup_corr}  min_dist_frac {min_dist_frac}  diameter {diameter}")
    for k, v in doc["proposed"].items():
        edge = "  [at edge]" if k in doc["at_edge"] else ""
        typer.echo(f"{k:<17} {doc['run_values'][k]!s:>8} -> {v!s:<8}{edge}")
    s = doc["score"]
    typer.echo(f"good {s['good']}  flagged {s['flagged']}  ({doc['wall_s']} s)")
    if doc["run_score"] is not None and not doc["better_than_run"]:
        rs = doc["run_score"]
        typer.echo(
            f"nothing better than the run's own parameters (good {rs['good']}  flagged "
            f"{rs['flagged']}): the proposal is the run's values"
        )


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

    from minime.tune import server

    app_ = server.TuneApp(run_dir, minian_npz=minian)
    httpd = server.make_server(app_, port=port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    typer.echo(f"minime tune: {url}  (Ctrl-C to stop)")
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
    out: Path,
    minutes: float = 10.0,
    fps: int = 30,
    segment: int = 1000,
    seed: int = 0,
    side: int = 600,
    scenario: str = "clean",
) -> None:
    """Write a synthetic Miniscope session (N.avi, GREY) and meta.json.

    SCENARIO is one of minime.testing.synth.SCENARIOS (clean, realistic, one per effect).
    """
    from minime.testing.synth import SCENARIOS, write_session

    if scenario not in SCENARIOS:
        raise typer.BadParameter(f"unknown scenario {scenario!r}; one of {sorted(SCENARIOS)}")
    typer.echo(write_session(out, minutes, fps, segment, seed, side, SCENARIOS[scenario]))


@app.command()
def score(
    run_dir: Annotated[Path | None, typer.Argument(help="Pipeline run directory.")] = None,
    raw: Annotated[Path | None, typer.Option(help="Raw dir (meta.json is in its parent).")] = None,
    minian: Annotated[Path | None, typer.Option(help="minian_units.npz to score.")] = None,
) -> None:
    """Score a pipeline run (or a Minian export) against the synthetic ground truth."""
    import json

    from minime import score as scorer

    if minian is not None:
        if raw is None:
            raise typer.BadParameter("--minian needs --raw")
        result = scorer.score_minian_npz(minian, raw)
        dest = minian.parent / (
            "score.json" if minian.stem == "minian_units" else f"score_{minian.stem}.json"
        )
    elif run_dir is not None:
        result = scorer.score_pipeline_dir(run_dir, raw)
        dest = run_dir / "score.json"
    else:
        raise typer.BadParameter("give RUN_DIR or --minian")
    text = json.dumps(result, indent=2)
    dest.write_text(text + "\n")
    typer.echo(text)


@app.command()
def serve(
    folder: Annotated[
        Path | None, typer.Argument(help="Folder to look for runs in (default: home).")
    ] = None,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 (0 = a free one).")] = 0,
    no_browser: Annotated[bool, typer.Option(help="Do not open the browser.")] = False,
    token: Annotated[str | None, typer.Option(hidden=True)] = None,
) -> None:
    """Open the run viewer and launcher in the browser (local only)."""
    from minime.serve import server

    server.serve_forever(folder, port=port, open_browser=not no_browser, token=token)
