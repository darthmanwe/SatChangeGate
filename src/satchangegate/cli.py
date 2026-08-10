"""SatChangeGate command line interface.

Boolean flags are declared with paired ``--flag/--no-flag`` decls. Previously
several were declared with a single decl and a ``True`` default, so Typer
generated no negative counterpart and the value could never be set to False:
``run-optimus`` could never call the VLM, and ``eval`` could never exercise the
VLM path, which is why it always reported "VLM reduction: 100%".

Commands that can spend money default to *not* spending it; enabling the VLM is
an explicit act.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from satchangegate import __version__
from satchangegate.config import get_settings, load_env_file
from satchangegate.data.oscd import default_oscd_root, discover_pairs, verify_layout

app = typer.Typer(
    add_completion=False,
    help="Preprocessing-first satellite change detection with a cost-controlled review funnel.",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"satchangegate {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: ARG001 - consumed by the eager callback
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    load_env_file()


@app.command("download-oscd")
def download_oscd_cmd(
    out: Path = typer.Option(default_oscd_root(), "--out", help="Destination root."),
    force: bool = typer.Option(False, "--force/--no-force", help="Re-download even if present."),
    keep_archives: bool = typer.Option(
        False, "--keep-archives/--no-keep-archives", help="Keep the downloaded zips."
    ),
) -> None:
    """Download the 13-band OSCD dataset (~513 MB) and verify its checksums."""
    from satchangegate.data.download import download_oscd

    console.print("[dim]Fetching OSCD from the Hugging Face mirror (~513 MB)...[/dim]")
    root = download_oscd(out, force=force, keep_archives=keep_archives)
    ok, msg = verify_layout(root)
    console.print(f"[{'green' if ok else 'red'}]{msg}[/]")
    if not ok:
        raise typer.Exit(1)


@app.command("tiles")
def tiles_cmd(
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports/tile_index.json"), "--out"),
    tile_size: int = typer.Option(64, "--tile-size"),
) -> None:
    """Build the labelled tile index and report its class balance."""
    from satchangegate.data.tiles import build_tile_index, save_tile_index, summarise

    tiles = build_tile_index(root, tile_size=tile_size)
    if not tiles:
        console.print("[red]No tiles. Run: satchangegate download-oscd[/red]")
        raise typer.Exit(1)
    save_tile_index(tiles, out)
    table = Table(title=f"Tile index ({len(tiles)} tiles, {tile_size}px)")
    for col in ("split", "cities", "change", "no-change", "% change"):
        table.add_column(col)
    for split, counts in sorted(summarise(tiles).items()):
        total = counts["positive"] + counts["negative"]
        table.add_row(
            split,
            str(counts["cities"]),
            str(counts["positive"]),
            str(counts["negative"]),
            f"{100 * counts['positive'] / total:.1f}%",
        )
    console.print(table)
    console.print(f"[green]Wrote {out}[/green]")


@app.command("run")
def run_cmd(
    pair: str = typer.Option(..., "--pair", help="City name, e.g. beirut."),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    vlm: bool = typer.Option(False, "--vlm/--no-vlm", help="Call the vision model."),
    llm: bool = typer.Option(False, "--llm/--no-llm", help="Generate the analyst report."),
    model: str | None = typer.Option(None, "--model", help="Override the VLM model id."),
    negative_mode: str | None = typer.Option(
        None, "--negative-mode", help="identity | stable | photometric"
    ),
) -> None:
    """Run the full funnel on one OSCD city pair."""
    from satchangegate.pipeline import run_pair

    matches = [p for p in discover_pairs(root) if p.pair_id == pair]
    if not matches:
        console.print(f"[red]Pair {pair!r} not found under {root}.[/red]")
        console.print("[dim]Run `satchangegate download-oscd` or `--pair` a known city.[/dim]")
        raise typer.Exit(1)

    result = run_pair(
        matches[0],
        out_dir=out,
        skip_vlm=not vlm,
        skip_llm=not llm,
        negative_mode=negative_mode,
        vlm_model=model,
    )
    c = result.classical
    table = Table(title=f"{pair} — {c.classical_gate}")
    table.add_column("field")
    table.add_column("value")
    for key, value in [
        ("gate", f"{c.classical_gate} ({c.gate_reason})"),
        ("confidence", f"{c.gate_confidence:.3f}"),
        ("dates", f"{c.date_t1} -> {c.date_t2} ({c.season_delta_days}d seasonal sep.)"),
        ("SSIM / pHash", f"{c.ssim:.3f} / {c.phash_distance}"),
        ("dNDVI / dNDBI", f"{c.ndvi_delta_mean:+.4f} / {c.ndbi_delta_mean:+.4f}"),
        ("changed area", f"{c.changed_area_percent:.2f}%"),
        ("registration", f"{c.registration_error_px} px"),
        ("cloud / snow", f"{c.cloud_fraction_max} / {c.snow_fraction_max}"),
        ("VLM called", str(result.vlm_called)),
        ("cost", f"${result.cost_usd:.4f}"),
    ]:
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"[green]Artifacts: {result.result_path.parent}[/green]")


@app.command("eval")
def eval_cmd(
    split: str = typer.Option("test", "--split", help="train | test | all"),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Evaluate the gate on a labelled split and report metrics with 95% CIs."""
    from satchangegate.evaluate import run_eval

    summary = run_eval(root, split, out)
    if summary.get("error"):
        console.print(f"[red]{summary['error']}[/red]")
        raise typer.Exit(1)

    m = summary["metrics"]
    if split == "train":
        console.print(
            "[yellow]Note: 'train' is the tuning split — these are in-sample "
            "figures. Quote `--split test`.[/yellow]"
        )
    table = Table(title=f"Gate evaluation — {split} (n={m['n']})")
    for col in ("metric", "value", "95% CI"):
        table.add_column(col)
    table.add_row(
        "recall", f"{m['recall']:.3f}", f"{m['recall_ci95'][0]:.3f}-{m['recall_ci95'][1]:.3f}"
    )
    if m["has_negative_class"]:
        table.add_row(
            "precision",
            f"{m['precision']:.3f}",
            f"{m['precision_ci95'][0]:.3f}-{m['precision_ci95'][1]:.3f}",
        )
        table.add_row(
            "specificity",
            f"{m['specificity']:.3f}",
            f"{m['specificity_ci95'][0]:.3f}-{m['specificity_ci95'][1]:.3f}",
        )
        table.add_row("F1", f"{m['f1']:.3f}", "-")
        table.add_row("balanced acc.", f"{m['balanced_accuracy']:.3f}", "-")
    else:
        table.add_row("precision / F1", "undefined", "no negative class")
    console.print(table)
    console.print(
        f"Gate filtered [bold]{summary['gate_filtered_pct']:.1f}%[/bold] of tiles "
        f"before any VLM call."
    )


@app.command("tune")
def tune_cmd(
    split: str = typer.Option("train", "--split", help="Split to fit on."),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Fit gate thresholds on one split; the other split is held out and asserted disjoint."""
    from satchangegate.tune_gate import sweep

    best = sweep(root, split=split, out_dir=out)
    console.print(f"[green]In-sample balanced accuracy: {best.score:.3f}[/green]")
    console.print(f"[dim]Report: {out / '_gate_tuning_report.md'}[/dim]")
    console.print(f"[dim]Thresholds: {out / 'tuned_thresholds.yaml'}[/dim]")
    console.print(
        "[yellow]These are in-sample. Run `satchangegate eval --split test` "
        "for the number to quote.[/yellow]"
    )


@app.command("e2e")
def e2e_cmd(
    split: str = typer.Option("test", "--split"),
    n: int | None = typer.Option(None, "--n", help="Sample size; default all."),
    seed: int = typer.Option(42, "--seed"),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    vlm: bool = typer.Option(False, "--vlm/--no-vlm", help="Call the vision model on candidates."),
    max_vlm_calls: int | None = typer.Option(None, "--max-vlm-calls", help="Hard spend cap."),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Skip completed tiles."),
) -> None:
    """Run the funnel end to end and report measured cost."""
    from satchangegate.e2e import E2EConfig, run_e2e

    if vlm:
        console.print(
            "[yellow]--vlm will make paid API calls. "
            f"Cap: {max_vlm_calls if max_vlm_calls is not None else 'none'}.[/yellow]"
        )
    summary = run_e2e(
        root,
        out,
        config=E2EConfig(
            split=split, n=n, seed=seed, skip_vlm=not vlm, max_vlm_calls=max_vlm_calls
        ),
        resume=resume,
    )
    cost = summary["funnel_cost"]
    table = Table(title=f"Funnel — {split} (n={summary['n']})")
    for col in ("stage", "count", "share"):
        table.add_column(col)
    table.add_row("input", str(cost["n_pairs"]), "100%")
    table.add_row("gate filtered", str(cost["n_gate_filtered"]), f"{cost['gate_filtered_pct']}%")
    table.add_row("sent to VLM", str(cost["n_vlm_calls"]), f"{cost['vlm_call_rate_pct']}%")
    console.print(table)
    console.print(
        f"Measured cost [bold]${cost['cost_usd']['total']}[/bold]; "
        f"review-everything counterfactual ${cost['counterfactual_review_everything_usd']} "
        f"({cost['savings_pct']}% saved)."
    )


@app.command("dev-tests")
def dev_tests_cmd(
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Run the offline control battery (identity / stable / photometric)."""
    from satchangegate.dev_controls import run_dev_tests

    summary = run_dev_tests(root, out)
    table = Table(title="Dev control battery")
    for col in ("check", "expected", "actual", "pass"):
        table.add_column(col)
    for row in summary["checks"]:
        table.add_row(
            row["name"],
            row["expected"],
            row["actual"],
            "[green]yes[/green]" if row["passed"] else "[red]no[/red]",
        )
    console.print(table)
    console.print(f"{summary['n_passed']}/{summary['n_checks']} passed")
    if summary["n_passed"] != summary["n_checks"]:
        raise typer.Exit(1)


@app.command("verify")
def verify_cmd(root: Path = typer.Option(default_oscd_root(), "--root")) -> None:
    """Check that the dataset and configuration are usable."""
    ok, msg = verify_layout(root)
    console.print(f"OSCD: [{'green' if ok else 'red'}]{msg}[/]")
    settings = get_settings()
    console.print(f"Config: {len(settings.gate.model_dump())} gate thresholds loaded")
    console.print(json.dumps({"bands": list(settings.bands)}, indent=2))
    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
