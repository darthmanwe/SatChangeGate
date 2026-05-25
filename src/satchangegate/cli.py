"""CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from satchangegate.config import load_thresholds
from satchangegate.data.goes import default_goes_root, fetch_goes_abi_snapshot, verify_layout as verify_goes
from satchangegate.data.optimus import (
    default_optimus_root,
    download_optimus_metadata,
    download_optimus_index,
    download_optimus_tar,
    create_optimus_dev_fixture,
    extract_group_tar,
    get_bitemporal_frames,
    list_eval_tiles_in_group,
    list_eval_series,
    verify_layout as verify_optimus,
)
from satchangegate.data.oscd import (
    default_oscd_root,
    discover_pairs,
    list_pairs,
    print_download_instructions,
    verify_layout,
)
from satchangegate.dev_controls import run_dev_tests
from satchangegate.e2e_random_eval import run_e2e_random_eval
from satchangegate.evaluate import run_eval
from satchangegate.evaluate_optimus import run_optimus_eval
from satchangegate.pipeline import run_from_bands, run_pair

app = typer.Typer(
    name="satchangegate",
    help="SatChangeGate: preprocessing-first satellite change detection PoC",
)
console = Console()


@app.command("download-oscd")
def download_oscd(
    out: Path = typer.Option(default_oscd_root(), "--out", help="OSCD root directory"),
) -> None:
    """Print OSCD download instructions and verify layout."""
    print_download_instructions(out)
    ok, msg = verify_layout(out)
    if ok:
        console.print(f"[green]OK:[/green] {msg}")
    else:
        console.print(f"[yellow]Not ready:[/yellow] {msg}")


@app.command("download-optimus")
def download_optimus(
    out: Path = typer.Option(default_optimus_root(), "--out"),
    metadata_only: bool = typer.Option(False, "--metadata-only", help="Eval JSON only (~10 KB)"),
    with_index: bool = typer.Option(False, "--with-index", help="Also fetch index.json (~464 MB)"),
    dev_fixture: bool = typer.Option(
        False,
        "--dev-fixture",
        help="Build 2-frame OPTIMUS-like tree from OSCD (no ~25 GB tar)",
    ),
    tile_id: str = typer.Option(
        "4472_3910",
        "--tile-id",
        help="Eval tile for tar download or dev fixture",
    ),
    oscd_pair: str = typer.Option("beirut", "--oscd-pair", help="OSCD pair for dev fixture PNGs"),
    allow_large_download: bool = typer.Option(
        False,
        "--allow-large-download",
        help="Download images/{N}.tar (~25 GB) from Hugging Face",
    ),
) -> None:
    """Download OPTIMUS eval metadata; optional index, dev fixture, or one tar."""
    console.print(
        "[dim]Full OPTIMUS is ~12 TB on Hugging Face. Default dev path: metadata + "
        "--dev-fixture (uses OSCD PNGs). Each images/{{N}}.tar is ~25 GB.[/dim]"
    )
    download_optimus_metadata(out)
    console.print(f"[green]Metadata[/green] -> {out / '2024_dataset_evaluation.json'}")
    if with_index:
        idx = download_optimus_index(out)
        console.print(f"[green]Index[/green] -> {idx}")
    if dev_fixture:
        fixture = create_optimus_dev_fixture(tile_id, oscd_pair_id=oscd_pair, root=out)
        console.print(f"[green]Dev fixture[/green] -> {fixture}")
    elif not metadata_only and allow_large_download:
        tar = download_optimus_tar(tile_id, out, allow_large_download=True)
        console.print(f"[green]Sample tar[/green] -> {tar}")
    elif not metadata_only and not with_index:
        console.print(
            "[yellow]Skipping tar. Use --dev-fixture, --with-index, or "
            "--allow-large-download.[/yellow]"
        )
    ok, msg = verify_optimus(out)
    console.print(f"[green]OK:[/green] {msg}" if ok else f"[yellow]{msg}[/yellow]")


@app.command("download-goes")
def download_goes(
    when: str = typer.Option("2020-06-15T17:00:00Z", "--when", help="UTC ISO timestamp"),
    satellite: int = typer.Option(16, "--satellite", help="GOES-16 (East) or 17/18"),
    out: Path = typer.Option(default_goes_root(), "--out"),
) -> None:
    """Download one GOES ABI CONUS snapshot for weather/illumination context."""
    ctx = fetch_goes_abi_snapshot(when, satellite=satellite, out_dir=out)
    console.print(f"[green]GOES snapshot[/green] -> {ctx.local_path}")
    console.print(f"brightness_mean={ctx.brightness_mean}")


@app.command("run")
def run(
    pair: str = typer.Option(..., "--pair", help="OSCD pair id, e.g. beirut"),
    oscd_root: Path = typer.Option(default_oscd_root(), "--oscd-root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    skip_vlm: bool = typer.Option(False, "--skip-vlm"),
    skip_llm: bool = typer.Option(False, "--skip-llm"),
    negative_mode: str | None = typer.Option(
        None,
        "--negative-mode",
        help="identity | stable | photometric — negative-control transforms",
    ),
    attach_goes: bool = typer.Option(False, "--attach-goes", help="Attach GOES ABI context metadata"),
) -> None:
    """Run pipeline for a single OSCD pair."""
    cfg = load_thresholds()
    pairs = discover_pairs(oscd_root) or list_pairs(oscd_root, "all")
    match = [p for p in pairs if p.pair_id == pair]
    if not match:
        console.print(f"[red]Pair not found:[/red] {pair}")
        raise typer.Exit(1)

    goes_ctx = None
    if attach_goes:
        when = match[0].date_t2 or "2020-06-15T17:00:00Z"
        goes_ctx = fetch_goes_abi_snapshot(f"{when}T17:00:00Z")

    console.print(f"Running pipeline for [bold]{pair}[/bold] mode={negative_mode}...")
    result = run_pair(
        match[0],
        cfg=cfg,
        out_dir=out,
        skip_vlm=skip_vlm,
        skip_llm=skip_llm,
        negative_mode=negative_mode,
        goes_context=goes_ctx,
    )
    console.print(f"Gate: {result.classical_gate}")
    console.print(f"VLM called: {result.vlm_called}")
    if result.report_path:
        console.print(f"Report: {result.report_path}")


@app.command("run-optimus")
def run_optimus_cmd(
    tile_id: str = typer.Option(..., "--tile-id", help="OPTIMUS eval tile e.g. 4472_3910"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    skip_vlm: bool = typer.Option(True, "--skip-vlm"),
    skip_llm: bool = typer.Option(True, "--skip-llm"),
    optimus_root: Path = typer.Option(default_optimus_root(), "--optimus-root"),
) -> None:
    """Run pipeline on first/last frames of an OPTIMUS eval time series."""
    cfg = load_thresholds()
    labels = list_eval_series(optimus_root)
    label_map = {s.tile_id: s.label for s in labels}
    b1, b2, t1, t2 = get_bitemporal_frames(tile_id, optimus_root)
    result = run_from_bands(
        f"optimus_{tile_id}",
        b1,
        b2,
        cfg=cfg,
        out_dir=out,
        date_t1=t1,
        date_t2=t2,
        skip_vlm=skip_vlm,
        skip_llm=skip_llm,
        ground_truth_label=label_map.get(tile_id),
    )
    console.print(f"Gate: {result.classical_gate} (ground_truth={label_map.get(tile_id)})")


@app.command("dev-tests")
def dev_tests_cmd(
    pair: str = typer.Option("beirut", "--pair"),
    skip_vlm: bool = typer.Option(True, "--skip-vlm"),
    skip_llm: bool = typer.Option(True, "--skip-llm"),
    no_goes: bool = typer.Option(False, "--no-goes"),
) -> None:
    """Run small-sample dev test battery (OSCD negatives, OPTIMUS, GOES)."""
    run_dev_tests(
        oscd_pair=pair,
        skip_vlm=skip_vlm,
        skip_llm=skip_llm,
        fetch_goes=not no_goes,
    )


@app.command("e2e-random")
def e2e_random_cmd(
    n: int = typer.Option(100, "--n", help="Number of random pairs"),
    seed: int = typer.Option(42, "--seed"),
    skip_vlm: bool = typer.Option(False, "--skip-vlm"),
    with_llm: bool = typer.Option(False, "--with-llm"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Random OSCD+OPTIMUS E2E run with VLM on gated candidates."""
    console.print(f"Running [bold]{n}[/bold] random pairs (seed={seed}, VLM={not skip_vlm})...")
    metrics = run_e2e_random_eval(
        n=n,
        seed=seed,
        out_dir=out,
        skip_vlm=skip_vlm,
        skip_llm=not with_llm,
    )
    table = Table(title="E2E random summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Completed", str(metrics["n_completed"]))
    table.add_row("Errors", str(metrics["n_errors"]))
    table.add_row("Gate candidate %", f"{metrics['gate_candidate_pct']}%")
    table.add_row("VLM calls", str(metrics["vlm_calls"]))
    table.add_row("VLM reduction %", f"{metrics['vlm_reduction_pct']}%")
    if metrics.get("vlm_verdict_counts"):
        table.add_row("VLM verdicts", str(metrics["vlm_verdict_counts"]))
    console.print(table)
    console.print(f"Report: {out / '_e2e_random_summary.md'}")


@app.command("eval-optimus")
def eval_optimus_cmd(
    group: int | None = typer.Option(
        None,
        "--group",
        help="Tar group index (e.g. 148, 425). Default: group with most eval labels locally available",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max labeled tiles to evaluate"),
    sample: int = typer.Option(
        20,
        "--sample",
        help="Also run gate on N unlabeled tiles from same tar (0=skip)",
    ),
    extract: bool = typer.Option(True, "--extract/--no-extract", help="Extract local tar first"),
    optimus_root: Path = typer.Option(default_optimus_root(), "--optimus-root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Evaluate classical gate on OPTIMUS labeled tiles (no VLM/LLM)."""
    if extract and group is not None:
        console.print(f"Extracting images/{group}.tar ...")
        extract_group_tar(group, optimus_root)
    elif extract and group is None:
        for tar in sorted((optimus_root / "images").glob("*.tar")):
            gi = int(tar.stem)
            labeled = list_eval_tiles_in_group(gi, optimus_root)
            if labeled:
                console.print(f"Found {len(labeled)} eval tiles in group {gi}; extracting ...")
                extract_group_tar(gi, optimus_root)
                group = gi
                break
        if group is None:
            tar = next((optimus_root / "images").glob("*.tar"), None)
            if tar:
                gi = int(tar.stem)
                console.print(
                    f"[yellow]No eval labels in local tars; extracting {tar.name} for unlabeled sample only.[/yellow]"
                )
                extract_group_tar(gi, optimus_root)
                group = gi

    metrics = run_optimus_eval(
        group_index=group,
        limit=limit,
        optimus_root=optimus_root,
        out_dir=out,
        skip_vlm=True,
        skip_llm=True,
        sample_unlabeled=sample,
    )
    if metrics.get("error"):
        console.print(f"[yellow]{metrics['error']}[/yellow]")
    else:
        prf = metrics["pair_level"]
        table = Table(title=f"OPTIMUS eval (group {metrics['group_index']})")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Labeled pairs", str(metrics["n_pairs"]))
        table.add_row("VLM candidate %", f"{metrics['gate_candidate_pct']}%")
        table.add_row("Filtered %", f"{metrics['gate_filtered_pct']}%")
        table.add_row("F1", f"{prf['f1']:.3f}")
        table.add_row("Precision", f"{prf['precision']:.3f}")
        table.add_row("Recall", f"{prf['recall']:.3f}")
        table.add_row("Specificity", f"{prf['specificity']:.3f}")
        console.print(table)
        console.print(f"Summary: {out / '_optimus_eval_summary.md'}")


@app.command("eval")
def eval_cmd(
    split: str = typer.Option("test", "--split", help="train | test | all"),
    oscd_root: Path = typer.Option(default_oscd_root(), "--oscd-root"),
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
