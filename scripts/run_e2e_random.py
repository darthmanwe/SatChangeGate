#!/usr/bin/env python3
"""Run random mixed-source E2E eval with VLM."""

from __future__ import annotations

import argparse

from satchangegate.e2e_random_eval import run_e2e_random_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Random OSCD+OPTIMUS E2E eval with VLM")
    p.add_argument("-n", "--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--with-llm", action="store_true")
    args = p.parse_args()
    metrics = run_e2e_random_eval(
        n=args.count,
        seed=args.seed,
        skip_vlm=args.skip_vlm,
        skip_llm=not args.with_llm,
    )
    print(f"Done: {metrics['n_completed']} pairs, VLM calls={metrics['vlm_calls']}")
    print(f"Report: data/reports/_e2e_random_summary.md")


if __name__ == "__main__":
    main()
