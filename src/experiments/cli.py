"""CLI: ``uv run bias-run --config path.yaml``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.experiments.pipeline import load_and_expand
from src.experiments.runner import run_pipeline_run, write_rollup
from src.training.config import ROOT, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a self-contained detect-and-combine pipeline YAML.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Pipeline YAML (merged with configs/pipelines/defaults.yaml)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Materialize backend data for each expert and exit",
    )
    parser.add_argument("--train", action="store_true", help="Train experts")
    parser.add_argument("--predict", action="store_true", help="Dump split predictions")
    parser.add_argument("--eval", dest="do_eval", action="store_true", help="Combine and evaluate")
    parser.add_argument("--all", action="store_true", help="train + predict + eval")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = load_and_expand(args.config)
    print(f"Expanded {len(runs)} run(s) from {args.config}")
    prepare_only = bool(args.prepare_only)
    do_train = args.train or args.all
    do_predict = args.predict or args.all
    do_eval = args.do_eval or args.all
    if not prepare_only and not (do_train or do_predict or do_eval):
        do_train = True
        do_predict = True
        do_eval = True
    reports = []
    for run in runs:
        print(json.dumps(run.job_meta(), default=str))
        reports.append(
            run_pipeline_run(
                run,
                do_train=do_train,
                do_predict=do_predict,
                do_eval=do_eval,
                skip_missing=args.skip_missing,
                prepare_only=prepare_only,
            )
        )
    stem = Path(args.config).stem
    root = resolve_path(runs[0].output.get("root", "outputs/experiments")) if runs else ROOT / "outputs" / "experiments"
    rollup = Path(root) / stem / "rollup.csv"
    write_rollup(reports, rollup)
    print(f"Wrote {rollup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
