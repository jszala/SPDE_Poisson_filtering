"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .experiment import resource_summary, run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spde-pf",
        description="Run the synthetic 2D SPDE/Poisson particle-filter study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run an experiment profile")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the profile and print a conservative resource estimate",
    )
    run.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    summary = resource_summary(config, workers=args.workers)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return
    run_experiment(
        config,
        args.output,
        workers=args.workers,
        progress=not args.no_progress,
    )
    print(f"results written to {args.output.resolve()}")


__all__ = ["main"]
