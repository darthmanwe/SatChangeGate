"""SatChangeGate command line interface.

Boolean flags are declared with paired ``--flag/--no-flag`` decls. Previously
several were declared with a single decl and a ``True`` default, so Typer
generated no negative counterpart and the value could never be set to False --
which is why ``eval`` could never exercise the VLM path and always reported
"VLM reduction: 100%".

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
    pixel_metrics: bool = typer.Option(
        False,
        "--pixel-metrics/--no-pixel-metrics",
        help="Also score the change mask pixel-for-pixel against the dataset's own mask.",
    ),
) -> None:
    """Evaluate the gate on a labelled split and report metrics with 95% CIs."""
    from satchangegate.evaluate import run_eval

    summary = run_eval(root, split, out, pixel_metrics=pixel_metrics)
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
        f"Tier 0 refused [bold]{summary['tier0_refused_pct']:.1f}%[/bold] of tiles "
        f"(unusable imagery); the gate filtered "
        f"[bold]{summary['gate_filtered_pct']:.1f}%[/bold] of the rest. "
        f"Total reduction before any API call: "
        f"[bold]{summary['total_reduction_pct']:.1f}%[/bold]."
    )
    console.print(
        f"[dim]Scored on {len(summary['cities_scored'])} of {len(summary['cities'])} "
        f"cities in this split.[/dim]"
    )
    if summary.get("pixel_metrics"):
        pm = summary["pixel_metrics"]
        console.print(
            f"[dim]Pixel level: F1 {pm['f1']:.3f}, IoU {pm['iou']:.3f} "
            f"(triage gate, not a segmentation model).[/dim]"
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
    sample: str = typer.Option(
        "stratified",
        "--sample",
        help="Which candidates to verify under a cap: stratified | sequential.",
    ),
    batch: bool = typer.Option(
        False, "--batch/--no-batch", help="Submit via the Batch API (half rate, asynchronous)."
    ),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Skip completed tiles."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Discard an existing ledger that holds paid verifications.",
    ),
) -> None:
    """Run the funnel end to end and report measured cost."""
    from satchangegate.e2e import (
        SAMPLE_STRATEGIES,
        E2EConfig,
        LedgerWouldBeDestroyedError,
        run_e2e,
    )

    if sample not in SAMPLE_STRATEGIES:
        console.print(f"[red]--sample must be one of {SAMPLE_STRATEGIES}[/red]")
        raise typer.Exit(1)
    if vlm:
        console.print(
            "[yellow]--vlm will make paid API calls. "
            f"Cap: {max_vlm_calls if max_vlm_calls is not None else 'none'}.[/yellow]"
        )
        if batch:
            console.print(
                "[dim]Batch API: half rate both directions, results may take minutes "
                "to hours. The batch id is written next to the ledger as soon as it "
                "is submitted, so a crash cannot strand the spend.[/dim]"
            )
        if sample == "sequential":
            console.print(
                "[yellow]--sample sequential verifies cities alphabetically until the "
                "cap runs out. It is not representative of the split; it exists only "
                "to reproduce the earlier run.[/yellow]"
            )
    try:
        summary = run_e2e(
            root,
            out,
            config=E2EConfig(
                split=split,
                n=n,
                seed=seed,
                skip_vlm=not vlm,
                max_vlm_calls=max_vlm_calls,
                sample=sample,
                batch=batch,
                overwrite=overwrite,
            ),
            resume=resume,
        )
    except LedgerWouldBeDestroyedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    cost = summary["funnel_cost"]
    table = Table(title=f"Funnel — {split} (n={summary['n']})")
    for col in ("stage", "count", "share"):
        table.add_column(col)
    table.add_row("input", str(cost["n_pairs"]), "100%")
    table.add_row(
        "Tier 0 refused", str(cost["n_tier0_refused"]), f"{cost['tier0_refused_pct']}% of input"
    )
    table.add_row(
        "gate filtered",
        str(cost["n_gate_filtered"]),
        f"{cost['gate_filtered_pct']}% of assessable",
    )
    table.add_row("candidates", str(cost["n_candidates"]), "-")
    table.add_row("sent to VLM", str(cost["n_vlm_calls"]), f"{cost['vlm_call_rate_pct']}% of input")
    console.print(table)
    console.print(
        f"Total reduction before any API call: [bold]{cost['total_reduction_pct']}%[/bold]. "
        f"Measured cost [bold]${cost['cost_usd']['total']}[/bold]."
    )
    if cost["n_batch_calls"]:
        console.print(
            f"Batching saved [bold]${cost['batch_saving_usd']}[/bold] "
            f"({cost['batch_saving_pct']}%) on the same calls."
        )
    console.print(
        f"[dim]Verified in {summary['cities_verified']} of "
        f"{summary['cities_with_candidates']} cities with candidates.[/dim]"
    )


@app.command("vlm-report")
def vlm_report_cmd(
    split: str = typer.Option("test", "--split"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Recompute what the VLM tier added, from the e2e ledger. Makes no API calls."""
    from satchangegate.vlm_report import run_vlm_report

    try:
        summary = run_vlm_report(split, out)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if summary.get("error"):
        console.print(f"[red]{summary['error']}[/red]")
        raise typer.Exit(1)

    prec = summary["precision"]
    table = Table(title=f"Second tier — {split} ({summary['n_calls']} calls)")
    for col in ("stage", "precision", "95% CI"):
        table.add_column(col)
    for label, key in (("gate alone", "gate_alone"), ("gate + VLM", "gate_plus_vlm")):
        pr = prec[key]
        table.add_row(
            label,
            f"{pr['value']:.3f} ({pr['successes']}/{pr['total']})",
            f"{pr['ci95'][0]:.3f}-{pr['ci95'][1]:.3f}",
        )
    console.print(table)
    console.print(
        f"[dim]Verified across {len(summary['cities_verified'])} cities: "
        f"{', '.join(summary['cities_verified'])}[/dim]"
    )
    console.print(f"[dim]Report: {out / '_e2e_vlm_report.md'}[/dim]")


@app.command("baselines")
def baselines_cmd(
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Compare the rule gate against learned models on the same features/split."""
    try:
        from satchangegate.baseline import run_baselines
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    summary = run_baselines(root, out)
    op = summary["shipped_gate_operating_point"]
    table = Table(
        title=f"Held-out cities (n={summary['n_test']}, "
        f"chance precision {summary['positive_prevalence_test']:.3f})"
    )
    for col in ("model", "avg precision", "ROC AUC", f"precision @ recall {op['recall']:.2f}"):
        table.add_column(col)
    for m in summary["models"]:
        table.add_row(
            m["name"],
            f"{m['average_precision']:.3f}",
            f"{m['roc_auc']:.3f}",
            f"{m['precision_at_gate_recall']:.3f}",
        )
    console.print(table)
    console.print(f"[dim]Report: {out / '_baselines.md'} · curve: {out / 'pr_curves.png'}[/dim]")


@app.command("fit-scorer")
def fit_scorer_cmd(
    split: str = typer.Option("train", "--split", help="Split to fit on."),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/models/gate_scorer.pkl"), "--out"),
) -> None:
    """Fit and persist the learned scorer, so it can be swapped in for the rules."""
    from satchangegate.data.tiles import build_tile_index
    from satchangegate.evaluate import compute_tile_features
    from satchangegate.tune_gate import assert_disjoint

    try:
        from satchangegate.scorer import fit_scorer, save_scorer
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    tiles = build_tile_index(root)
    fit_tiles = [t for t in tiles if t.split == split]
    held_tiles = [t for t in tiles if t.split != split]
    if not fit_tiles:
        console.print(f"[red]No tiles in split {split!r}[/red]")
        raise typer.Exit(1)
    assert_disjoint({t.city for t in fit_tiles}, {t.city for t in held_tiles})

    settings = get_settings()
    console.print(f"[dim]Computing features for {len(fit_tiles)} tiles...[/dim]")
    fit_rows = compute_tile_features(root, fit_tiles, settings)
    held_rows = compute_tile_features(root, held_tiles, settings) if held_tiles else None

    artifact = fit_scorer(fit_rows, held_rows)
    path = save_scorer(artifact, out)
    console.print(f"[green]Wrote {path}[/green]")
    console.print(f"[dim]Model card: {path.with_suffix('.card.json')}[/dim]")
    if artifact.metrics:
        m = artifact.metrics
        console.print(
            f"Held-out average precision [bold]{m['average_precision']:.3f}[/bold], "
            f"ROC AUC {m['roc_auc']:.3f} on {m['n_test']} tiles."
        )
    console.print(
        "[yellow]The rule gate remains the default. To use this instead, set "
        "scorer.kind: learned in thresholds.yaml (and scorer.threshold, which "
        "`satchangegate conformal` can pick with a guarantee). Read the model "
        "card first.[/yellow]"
    )


@app.command("conformal")
def conformal_cmd(
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    alpha: float = typer.Option(0.20, "--alpha", help="Target false-negative rate."),
    delta: float = typer.Option(0.10, "--delta", help="1 - confidence in that bound."),
) -> None:
    """Calibrate a risk-controlled threshold, then test whether it holds out of sample."""
    from satchangegate.conformal import run_conformal

    summary = run_conformal(root, out, alpha=alpha, delta=delta)
    cal, verdict = summary["calibration"], summary["held_out_test"]
    if not cal["risk_controlled"]:
        console.print(
            f"[yellow]No threshold could be certified at alpha={alpha}. The "
            "calibration sample is too small to bound the risk that tightly; no "
            "controlled operating point is published.[/yellow]"
        )
        raise typer.Exit(0)

    console.print(
        f"Calibrated lambda = [bold]{cal['lambda']:.3f}[/bold] "
        f"(calibration FNR {cal['calibration_fnr']:.3f}, "
        f"upper bound {cal['calibration_fnr_upper_bound']:.3f} <= {alpha})"
    )
    table = Table(title="Does the guarantee hold on held-out cities?")
    for col in ("city", "n", "positives", "recall", "within alpha"):
        table.add_column(col)
    for city, m in verdict["per_city"].items():
        held = (
            "-" if m["held"] is None else ("[green]yes[/green]" if m["held"] else "[red]no[/red]")
        )
        table.add_row(city, str(m["n"]), str(m["n_positive"]), f"{m['recall']:.3f}", held)
    console.print(table)
    console.print(
        f"Observed FNR {verdict['observed_fnr']:.3f} vs target {alpha}: "
        + (
            "[green]held overall[/green]"
            if verdict["held_overall"]
            else "[red]did not hold overall[/red]"
        )
    )
    console.print(f"[dim]{verdict['reading']}[/dim]")
    console.print(f"[dim]Report: {out / '_conformal.md'}[/dim]")


@app.command("operating-points")
def operating_points_cmd(
    split: str = typer.Option("test", "--split"),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    budget_usd: list[float] = typer.Option(
        None, "--budget-usd", help="Budget to tabulate; repeatable."
    ),
    cost_per_call_usd: float | None = typer.Option(
        None, "--cost-per-call", help="Override the measured per-verification price."
    ),
) -> None:
    """Tabulate what each review budget buys: threshold, recall, precision, spend."""
    from satchangegate.operating_points import DEFAULT_BUDGETS_USD, run_operating_points

    budgets = tuple(budget_usd) if budget_usd else DEFAULT_BUDGETS_USD
    summary = run_operating_points(
        root,
        out,
        split=split,
        cost_per_call_usd=cost_per_call_usd,
        budgets_usd=budgets,
    )
    table = Table(
        title=f"What a budget buys — {split} "
        f"(${summary['cost_per_call_usd']:.6f}/call, {summary['cost_source']})"
    )
    for col in ("budget", "calls", "threshold", "flagged", "recall", "precision", "spend"):
        table.add_column(col)
    for p in summary["operating_points"]:
        table.add_row(
            f"${p['budget_usd']:.2f}",
            str(p["affordable_calls"]),
            f"{p['threshold']:.3f}",
            str(p["n_flagged"]),
            f"{p['recall']:.3f}",
            f"{p['precision']:.3f}",
            f"${p['spend_usd']:.4f}",
        )
    console.print(table)
    console.print(
        f"[dim]Reviewing all {summary['n_scored']} assessable tiles would cost "
        f"${summary['review_everything_usd']}. Report: "
        f"{out / '_operating_points.md'}[/dim]"
    )


@app.command("embedding-coverage")
def embedding_coverage_cmd(
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Report how much of the dataset AlphaEarth embeddings can actually speak to."""
    import json as _json

    from satchangegate.data.embeddings import FIRST_EMBEDDING_YEAR, coverage_report

    report = coverage_report(discover_pairs(root))
    if not report["n_pairs"]:
        console.print("[red]No pairs found. Run: satchangegate download-oscd[/red]")
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)
    (out / "_embedding_coverage.json").write_text(_json.dumps(report, indent=2), encoding="utf-8")
    console.print(
        f"AlphaEarth coverage starts in {FIRST_EMBEDDING_YEAR}. Of "
        f"{report['n_pairs']} pairs: [bold]{report['n_t1_covered']}[/bold] have a "
        f"covered first acquisition, [bold]{report['n_t2_covered']}[/bold] a covered "
        f"second."
    )
    console.print(
        f"Bitemporal embedding change is measurable on "
        f"[bold]{report['n_bitemporal_measurable']}[/bold] of {report['n_pairs']} pairs "
        f"({report['pct_bitemporal_measurable']}%), which is why the shipped feature is "
        "single-date context at t2 and the change version is a labelled probe."
    )
    console.print(f"[dim]Wrote {out / '_embedding_coverage.json'}[/dim]")


@app.command("dev-tests")
def dev_tests_cmd(
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Run the offline control battery (identity / stable / photometric)."""
    from satchangegate.dev_controls import run_dev_tests

    summary = run_dev_tests(root, out)
    if summary.get("error") or not summary["n_checks"]:
        console.print(f"[red]{summary.get('error', 'no checks ran')}[/red]")
        raise typer.Exit(1)
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


@app.command("ab-normalize")
def ab_normalize_cmd(
    split: str = typer.Option("test", "--split"),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
) -> None:
    """Evaluate with PIF radiometric normalization off, then on, and compare.

    Closes a gap this repo should not have had: `_ab_normalize.json` was
    committed from 0.3.0 with nothing that could regenerate it. A negative
    result nobody can reproduce is an assertion, not a result.

    Expensive -- two full feature passes and two baseline fits.
    """
    from satchangegate.ab_normalize import run_ab_normalize

    console.print("[dim]Two full evaluations. This takes a while.[/dim]")
    summary = run_ab_normalize(root, out, split=split)
    off, on, delta = summary["off"], summary["on"], summary["delta_on_minus_off"]

    table = Table(title=f"PIF normalization A/B — {split}")
    for col in ("metric", "off", "on", "delta"):
        table.add_column(col)
    table.add_row(
        "gate F1",
        f"{off['gate']['f1']:.4f}",
        f"{on['gate']['f1']:.4f}",
        f"{delta['gate_f1']:+.4f}",
    )
    table.add_row(
        "gate precision",
        f"{off['gate']['precision']:.4f}",
        f"{on['gate']['precision']:.4f}",
        f"{delta['gate_precision']:+.4f}",
    )
    for name in off["models"]:
        table.add_row(
            f"{name} AP",
            f"{off['models'][name]:.4f}",
            f"{on['models'][name]:.4f}",
            f"{delta[f'{name}_ap']:+.4f}",
        )
    console.print(table)
    console.print(f"[dim]{summary['reading']}[/dim]")
    console.print(f"[dim]Report: {out / '_ab_normalize.md'}[/dim]")


@app.command("run-images")
def run_images_cmd(
    t1: Path = typer.Option(..., "--t1", help="Before image."),
    t2: Path = typer.Option(..., "--t2", help="After image."),
    bands: str = typer.Option(..., "--bands", help='Band mapping, e.g. "B04=1,B03=2,B02=3".'),
    reflectance_scale: float = typer.Option(
        10000.0, "--reflectance-scale", help="Divide raw values by this to get reflectance."
    ),
    reflectance_offset: float = typer.Option(0.0, "--reflectance-offset"),
    date_t1: str | None = typer.Option(None, "--date-t1", help="ISO date of the before image."),
    date_t2: str | None = typer.Option(None, "--date-t2"),
    alignment: str = typer.Option(
        "georeferenced", "--alignment", help="georeferenced | already_aligned"
    ),
    resolution_m: float | None = typer.Option(
        None, "--resolution-m", help="Ground sample distance."
    ),
    name: str = typer.Option("upload", "--name", help="Run identifier."),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    vlm: bool = typer.Option(False, "--vlm/--no-vlm", help="Call the vision model."),
) -> None:
    """Run the funnel over two image files under a declared contract.

    Nothing physical is inferred. Bands are named rather than positional, the
    reflectance scale is stated rather than guessed from dtype, and footprints
    must genuinely overlap -- two equal-sized rasters of different continents
    would otherwise produce a confident answer about nothing.
    """
    from satchangegate.services import RunImagesRequest, ServiceUnavailable
    from satchangegate.services.operations import run_images_service

    try:
        request = RunImagesRequest(
            t1=t1,
            t2=t2,
            bands=bands,
            reflectance_scale=reflectance_scale,
            reflectance_offset=reflectance_offset,
            date_t1=date_t1,
            date_t2=date_t2,
            alignment=alignment,  # type: ignore[arg-type]
            resolution_m=resolution_m,
            name=name,
            out=out,
            vlm=vlm,
        )
        result = run_images_service(request)
    except (ValueError, ServiceUnavailable) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    data = result.data
    ingest = data["ingest"]
    cap = ingest["capability"]
    console.print(
        f"[dim]{ingest['width']}x{ingest['height']} px, "
        f"{ingest['valid_fraction']:.1%} valid, CRS {ingest['crs'] or 'none'}[/dim]"
    )
    for note in ingest["warnings"]:
        console.print(f"[yellow]{note}[/yellow]")

    if data["lane"] == "full":
        c = data["classical"]
        table = Table(title=f"{name} — {c['classical_gate']}")
        table.add_column("field")
        table.add_column("value")
        for key, value in (
            ("gate", f"{c['classical_gate']} ({c['gate_reason']})"),
            ("confidence", f"{c['gate_confidence']:.3f}"),
            ("dNDVI / dNDBI", f"{c['ndvi_delta_mean']:+.4f} / {c['ndbi_delta_mean']:+.4f}"),
            ("changed area", f"{c['changed_area_percent']:.2f}%"),
            ("registration", f"{c['registration_error_px']} px"),
            ("VLM called", str(data["vlm_called"])),
            ("cost", f"${data['cost_usd']:.4f}"),
        ):
            table.add_row(key, str(value))
        console.print(table)
    else:
        r = data["structural"]
        console.print(f"[yellow]No gate decision: {r['gate_reason']}[/yellow]")
        console.print(f"[dim]Missing for the gate: {', '.join(cap['missing_for_gate'])}.[/dim]")
        table = Table(title=f"{name} — structural evidence only")
        table.add_column("field")
        table.add_column("value")
        for key, value in (
            ("SSIM", f"{r['ssim']:.4f}"),
            ("pHash distance", r["phash_distance"]),
            ("change-vector mean", f"{r['cva_magnitude_mean']:.4f}"),
            ("changed area (CVA)", f"{r['changed_area_percent']:.2f}%"),
            ("components", r["n_components"]),
            ("unavailable", f"{len(r['unavailable'])} spectral features"),
        ):
            table.add_row(key, str(value))
        console.print(table)
        console.print(f"[dim]{data['note']}[/dim]")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback only."),
    port: int = typer.Option(8000, "--port"),
    root: Path = typer.Option(default_oscd_root(), "--root"),
    out: Path = typer.Option(Path("data/reports"), "--out"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open a browser."),
    allow_spend: bool = typer.Option(
        False, "--allow-spend/--no-allow-spend", help="Permit paid API calls from the UI."
    ),
    spend_cap_usd: float = typer.Option(1.00, "--spend-cap-usd", help="Hard session cap."),
) -> None:
    """Serve the local review UI.

    Refuses a non-loopback host rather than warning about one. This process holds
    whatever is in `.env`, and "it is only on my LAN" is not an access policy;
    sharing it is a separate feature that needs real authentication.
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[red]Refusing to bind {host!r}.[/red] This process holds a live API key "
            "and has no authentication beyond a per-launch token."
        )
        console.print(
            "[dim]Loopback only: --host 127.0.0.1. Sharing needs real auth and a "
            "transport that is not plain HTTP.[/dim]"
        )
        raise typer.Exit(1)

    try:
        import uvicorn

        from satchangegate.webui.app import AppConfig, create_app
    except ImportError as exc:
        console.print(f"[red]The web UI needs its optional extra: {exc}[/red]")
        console.print('[dim]pip install -e ".[ui]"[/dim]')
        raise typer.Exit(1) from exc

    config = AppConfig(
        oscd_root=root,
        reports=out,
        allow_spend=allow_spend,
        spend_cap_usd=spend_cap_usd,
    )
    url = f"http://{host}:{port}/"
    console.print(f"[green]SatChangeGate UI[/green] {url}")
    if allow_spend:
        console.print(
            f"[yellow]Paid calls are ENABLED, capped at ${spend_cap_usd:.2f} for this "
            "session.[/yellow]"
        )
    else:
        console.print("[dim]Paid calls are disabled. --allow-spend turns them on.[/dim]")
    console.print("[dim]Ctrl-C to stop.[/dim]")

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")


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
