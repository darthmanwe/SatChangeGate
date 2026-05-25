"""Small-sample dev test runner for negative controls, OPTIMUS, and GOES."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from satchangegate.config import load_thresholds
from satchangegate.data.goes import default_goes_root, fetch_goes_abi_snapshot, verify_layout as verify_goes
from satchangegate.data.optimus import (
    default_optimus_root,
    download_optimus_metadata,
    ensure_optimus_dev_fixtures,
    get_bitemporal_frames,
    list_eval_series,
    verify_layout as verify_optimus,
)
from satchangegate.data.oscd import default_oscd_root, discover_pairs
from satchangegate.pipeline import run_from_bands, run_pair

console = Console()


def run_dev_tests(
    *,
    oscd_pair: str = "beirut",
    optimus_tile: str = "4472_3910",
    optimus_no_change_tile: str = "4472_3910",
    skip_vlm: bool = True,
    skip_llm: bool = True,
    fetch_goes: bool = True,
) -> dict:
    """Run a small fixed battery of checks; returns summary dict."""
    cfg = load_thresholds()
    out_dir = Path("data/reports/dev_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    pairs = discover_pairs(default_oscd_root())
    match = [p for p in pairs if p.pair_id == oscd_pair]
    if not match:
        raise FileNotFoundError(f"OSCD pair not found: {oscd_pair}")
    pair = match[0]

    for mode in ("identity", "stable", "photometric"):
        try:
            r = run_pair(
                pair,
                cfg=cfg,
                out_dir=out_dir,
                skip_vlm=skip_vlm,
                skip_llm=skip_llm,
                negative_mode=mode,
            )
            rows.append(
                {
                    "test": f"oscd_{mode}",
                    "id": r.pair_id,
                    "gate": r.classical_gate,
                    "expected_gate": "no_change" if mode in ("identity", "stable") else "any",
                    "pass": r.classical_gate == "no_change"
                    if mode in ("identity", "stable")
                    else r.classical_gate in ("no_change", "candidate_change"),
                }
            )
        except Exception as e:
            rows.append({"test": f"oscd_{mode}", "id": oscd_pair, "gate": "error", "pass": False, "error": str(e)})

    # OPTIMUS — use dev fixtures when full tars are not present
    try:
        download_optimus_metadata(default_optimus_root())
        ensure_optimus_dev_fixtures(default_optimus_root(), no_change_tile=optimus_no_change_tile)
        labels = list_eval_series(default_optimus_root(), label=0, limit=1)
        tile = labels[0].tile_id if labels else optimus_no_change_tile
        b1, b2, t1, t2 = get_bitemporal_frames(tile)
        r = run_from_bands(
            f"optimus_{tile}",
            b1,
            b2,
            cfg=cfg,
            out_dir=out_dir,
            date_t1=t1,
            date_t2=t2,
            skip_vlm=skip_vlm,
            skip_llm=skip_llm,
            ground_truth_label=0,
        )
        rows.append(
            {
                "test": "optimus_no_change",
                "id": tile,
                "gate": r.classical_gate,
                "expected_gate": "no_change",
                "pass": r.classical_gate == "no_change",
                "ground_truth": 0,
            }
        )
    except Exception as e:
        rows.append({"test": "optimus_no_change", "id": optimus_tile, "gate": "error", "pass": False, "error": str(e)})

    # OPTIMUS change eval tile (label 1) — optional second sample
    try:
        labels1 = list_eval_series(default_optimus_root(), label=1, limit=1)
        tile1 = labels1[0].tile_id if labels1 else optimus_tile
        b1, b2, t1, t2 = get_bitemporal_frames(tile1)
        r = run_from_bands(
            f"optimus_{tile1}",
            b1,
            b2,
            cfg=cfg,
            out_dir=out_dir,
            date_t1=t1,
            date_t2=t2,
            skip_vlm=skip_vlm,
            skip_llm=skip_llm,
            ground_truth_label=1,
        )
        rows.append(
            {
                "test": "optimus_change",
                "id": tile1,
                "gate": r.classical_gate,
                "expected_gate": "candidate_change",
                "pass": r.classical_gate == "candidate_change",
                "ground_truth": 1,
            }
        )
    except Exception as e:
        rows.append({"test": "optimus_change", "id": optimus_tile, "gate": "error", "pass": False, "error": str(e)})

    goes_ok = False
    goes_msg = ""
    if fetch_goes:
        try:
            ctx = fetch_goes_abi_snapshot("2020-06-15T17:00:00Z")
            goes_ok = ctx.local_path is not None
            goes_msg = str(ctx.local_path)
            run_pair(
                pair,
                cfg=cfg,
                out_dir=out_dir,
                skip_vlm=True,
                skip_llm=True,
                goes_context=ctx,
                run_id=f"{oscd_pair}_with_goes",
            )
        except Exception as e:
            goes_msg = str(e)
        rows.append({"test": "goes_fetch", "id": "CONUS", "gate": goes_msg, "pass": goes_ok})

    passed = sum(1 for r in rows if r.get("pass"))
    summary = {"passed": passed, "total": len(rows), "results": rows}
    (out_dir / "_dev_tests_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = Table(title="Dev test battery")
    table.add_column("Test")
    table.add_column("ID")
    table.add_column("Gate")
    table.add_column("Pass")
    for row in rows:
        table.add_row(
            row["test"],
            str(row.get("id", "")),
            str(row.get("gate", "")),
            "yes" if row.get("pass") else "no",
        )
    console.print(table)
    console.print(f"Summary: {passed}/{len(rows)} passed -> {out_dir / '_dev_tests_summary.json'}")
    return summary
