"""CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from satchangegate.config import load_thresholds
from satchangegate.data.oscd import (
    discover_pairs,
    list_pairs,
    print_download_instructions,
    verify_layout,
)
from satchangegate.evaluate import run_eval
from satchangegate.pipeline import run_pair

app = typer.Typer(
    name="satchangegate",
    help="SatChangeGate: preprocessing-first satellite change detection PoC",
)
console = Console()


@app.command("download-oscd")
def download_oscd(
    out: Path = typer.Option(Path("data/raw/oscd"), "--out", help="OSCD root directory"),
) -> None:
    """Print OSCD download instructions and verify layout."""
    print_download_instructions(out)
    ok, msg = verify_layout(out)
    if ok:
        console.print(f"[green]OK:[/green] {msg}")
    else:
        console.print(f"[yellow]Not ready:[/yellow] {msg}")


@app.command("run")
def run(
    pair: str = typer.Option(..., "--pair", help="OSCD pair id, e.g. beirut"),
    oscd_root: Path = typer.Option(Path("data/raw/oscd"), "--oscd-root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    skip_vlm: bool = typer.Option(False, "--skip-vlm"),
    skip_llm: bool = typer.Option(False, "--skip-llm"),
) -> None:
    """Run pipeline for a single OSCD pair."""
    cfg = load_thresholds()
    pairs = discover_pairs(oscd_root) or list_pairs(oscd_root, "all")
    match = [p for p in pairs if p.pair_id == pair]
    if not match:
        console.print(f"[red]Pair not found:[/red] {pair}")
        raise typer.Exit(1)

    console.print(f"Running pipeline for [bold]{pair}[/bold]...")
    result = run_pair(match[0], cfg=cfg, out_dir=out, skip_vlm=skip_vlm, skip_llm=skip_llm)
    console.print(f"Gate: {result.classical_gate}")
    console.print(f"VLM called: {result.vlm_called}")
    if result.report_path:
        console.print(f"Report: {result.report_path}")


@app.command("eval")
def eval_cmd(
    split: str = typer.Option("test", "--split", help="train | test | all"),
    oscd_root: Path = typer.Option(Path("data/raw/oscd"), "--oscd-root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    skip_vlm: bool = typer.Option(True, "--skip-vlm", help="Skip VLM API calls in eval"),
    skip_llm: bool = typer.Option(True, "--skip-llm", help="Skip LLM API calls in eval"),
) -> None:
    """Evaluate pipeline on OSCD split and write summary."""
    console.print(f"Evaluating split=[bold]{split}[/bold]...")
    metrics = run_eval(
        oscd_root,
        split=split,
        out_dir=out,
        skip_vlm=skip_vlm,
        skip_llm=skip_llm,
    )

    if "error" in metrics:
        console.print(f"[yellow]{metrics['error']}[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Eval metrics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Pairs", str(metrics["n_pairs"]))
    table.add_row("Gate candidate %", f"{metrics['gate_candidate_pct']}%")
    table.add_row("Filtered %", f"{metrics['gate_filtered_pct']}%")
    table.add_row("VLM calls", str(metrics["vlm_calls"]))
    table.add_row("VLM reduction %", f"{metrics['vlm_reduction_pct']}%")
    prf = metrics["pair_level"]
    table.add_row("F1 (pair)", f"{prf['f1']:.3f}")
    table.add_row("Mean IoU", str(metrics["mean_iou"]))
    console.print(table)
    console.print(f"Summary: {out / '_eval_summary.md'}")


if __name__ == "__main__":
    app()
