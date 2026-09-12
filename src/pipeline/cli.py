"""CLI: run a full detection pipeline from one YAML recipe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pipeline.load import list_datasets, list_experts, list_pipelines, load_pipeline
from src.pipeline.run import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one config-defined pipeline (dataset + experts + post-hoc stack).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Pipeline YAML under configs/pipelines/ (name or path). Defaults to asymmetric.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Dataset name or training YAML (e.g. hit_uav or configs/training/hit_uav.yaml)",
    )
    parser.add_argument(
        "--experts",
        default=None,
        help="Comma-separated expert names, e.g. yolo26n,rfdetr_nano,dino,rtmdet",
    )
    parser.add_argument("--noise", default=None, help="Noise family: clean, L, O, C, Mix")
    parser.add_argument("--ratio", type=int, default=None, help="Noise percent (1 means 1%%)")
    parser.add_argument("--join", default=None, help="Join method: nms, soft_nms, wbf, learned")
    parser.add_argument("--suppress", default=None, help="Per-expert suppress: nms or soft_nms")
    parser.add_argument("--no-calibrate", action="store_true", help="Disable calibration")
    parser.add_argument("--gate", action="store_true", help="Enable the cal-set softmax gate")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--eval", dest="do_eval", action="store_true")
    parser.add_argument("--all", action="store_true", help="train + predict + eval")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    parser.add_argument("--list", action="store_true", help="Print datasets, experts, pipelines and exit")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved recipe and exit")
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if args.noise is not None or args.ratio is not None:
        noise = {}
        if args.noise is not None:
            noise["family"] = args.noise
        if args.ratio is not None:
            noise["ratio"] = args.ratio
        out["noise"] = noise
    pipe: dict = {}
    if args.join is not None:
        pipe["join"] = {"method": args.join}
    if args.suppress is not None:
        pipe["suppress"] = {"method": args.suppress}
    if args.no_calibrate:
        pipe["calibrate"] = {"enabled": False}
    if args.gate:
        pipe["gate"] = {"enabled": True}
    if pipe:
        out["pipeline"] = pipe
    if args.experts:
        out["experts"] = [item.strip() for item in args.experts.split(",") if item.strip()]
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print(json.dumps(
            {
                "datasets": list_datasets(),
                "experts": list_experts(),
                "pipelines": list_pipelines(),
            },
            indent=2,
        ))
        return 0
    config = args.config
    if config is None and args.dataset is None:
        config = Path("configs/pipelines/asymmetric.yaml")
    recipe = load_pipeline(config, dataset=args.dataset, overrides=_overrides(args))
    if args.print_config:
        printable = {k: v for k, v in recipe.items() if k != "expert_specs"}
        printable["experts"] = recipe["expert_specs"]
        print(json.dumps(printable, indent=2, default=str))
        return 0
    do_train = args.train or args.all
    do_predict = args.predict or args.all
    do_eval = args.do_eval or args.all
    if not (do_train or do_predict or do_eval):
        do_eval = True
    report = run_pipeline(
        recipe,
        do_train=do_train,
        do_predict=do_predict,
        do_eval=do_eval,
        skip_missing=args.skip_missing,
    )
    print(json.dumps({k: report[k] for k in ("name", "status", "output", "n_experts_loaded") if k in report}, indent=2))
    return 0 if report.get("status") != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
