"""CLI entrypoint for config-driven training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.training.config import load_config
from src.training.registry import get_trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a detector from a YAML config under configs/training/.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Path to a dataset training YAML (merged with configs/training/defaults.yaml)",
    )
    parser.add_argument(
        "--defaults",
        type=Path,
        default=None,
        help="Optional override for defaults.yaml",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only convert COCO->YOLO and write data.yaml; skip training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, defaults_path=args.defaults)
    trainer = get_trainer(cfg["backend"])

    print(f"Config: {cfg['_config_path']}")
    print(f"Backend: {cfg['backend']}")
    print(f"Run name: {cfg['name']}")
    print(f"Dataset: {cfg['dataset']['root']} / {cfg['dataset']['ann_dir']}")

    data_path = trainer.prepare(cfg)
    print(f"Prepared data: {data_path}")

    if args.prepare_only:
        meta = data_path.parent / "prepare_meta.json"
        if meta.exists():
            print(meta.read_text(encoding="utf-8"))
        return 0

    out_dir = trainer.train(cfg, data_path)
    summary = {
        "run_name": cfg["name"],
        "output": str(out_dir),
        "data": str(data_path),
        "model": cfg["model"]["name"],
        "imgsz": cfg["train"]["imgsz"],
        "batch": cfg["train"]["batch"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Training output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
