"""CLI: dump unified COCO detections for a trained run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.training.config import load_config
from src.training.registry import get_trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run detector inference and dump detections.")
    parser.add_argument("--config", "-c", required=True, type=Path)
    parser.add_argument("--defaults", type=Path, default=None)
    parser.add_argument("--split", default="val", help="split name (test, cal, val, or train)")
    parser.add_argument("--weights", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, defaults_path=args.defaults)
    if args.weights is not None:
        cfg.setdefault("model", {})["weights"] = str(args.weights)
    trainer = get_trainer(cfg["backend"])
    data_path = trainer.prepare(cfg)
    out = trainer.predict(cfg, data_path, args.split)
    print(json.dumps({"split": args.split, "predictions": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
