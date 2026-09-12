"""Write train-only L/O/C/Mix COCO JSONs under annotations_noise/.

Prefer scripts/data/prepare_noisy_trains.py, which checks layout first.

    uv run python scripts/data/generate_noise.py --all
    uv run python scripts/data/generate_noise.py --dataset hit_uav
"""
from __future__ import annotations

import argparse
import json
import sys

from src.data.paths import NOISE_FAMILIES, RATIO_GRID
from src.data.prepare import (
    DATA_CONFIG_DIR,
    generate_noisy_trains_for_config,
    load_data_config,
    resolve_data_config_paths,
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noisy train COCO JSONs into annotations_noise/.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Data YAML stem under configs/data/ (e.g. hit_uav). Use --all for every config.",
    )
    parser.add_argument("--all", action="store_true", help="Process every configs/data/*.yaml")
    parser.add_argument(
        "--families",
        default="L,O,C,Mix",
        help="Comma-separated noise families (default: L,O,C,Mix)",
    )
    parser.add_argument(
        "--ratios",
        default=",".join(str(r) for r in RATIO_GRID if r > 0),
        help="Comma-separated integer percents (default: 1,2,5,10,20,30,50)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing noisy JSONs")
    parser.add_argument(
        "--carve",
        action="store_true",
        help="For carve_cal datasets, carve cal from train before writing noise",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        dataset = "all"
    elif args.dataset:
        dataset = args.dataset
    else:
        print("Pass --dataset NAME or --all", file=sys.stderr)
        return 2
    families = tuple(_parse_csv(args.families))
    unknown = [f for f in families if f not in NOISE_FAMILIES]
    if unknown:
        raise SystemExit(f"Unknown noise families {unknown}. Known: {list(NOISE_FAMILIES)}")
    ratios = tuple(int(r) for r in _parse_csv(args.ratios))
    paths = resolve_data_config_paths(dataset)
    if not paths:
        print(f"No data configs in {DATA_CONFIG_DIR}")
        return 1
    reports = []
    for path in paths:
        cfg = load_data_config(path)
        report = generate_noisy_trains_for_config(
            cfg,
            families=families,
            ratios=ratios,
            force=args.force,
            carve=args.carve,
        )
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("name", "root") if k in report}, indent=2))
    print(json.dumps(reports, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
