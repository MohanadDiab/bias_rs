"""Check COCO dataset layout, then write noisy train JSONs.

Expected layout (val and cal/test stay clean; only train is noised):

    datasets/<name>/
      images/{train,val,test}/     # test when the dataset has it
      <ann_dir>/instances_{split}.json

Noisy trains:

    datasets/<name>/annotations_noise/<ann_dir>/<family>_<pct>/instances_train.json

Examples:

    uv run python scripts/data/prepare_noisy_trains.py --all
    uv run python scripts/data/prepare_noisy_trains.py --dataset hit_uav
    uv run python scripts/data/prepare_noisy_trains.py --all --check-only
    uv run python scripts/data/prepare_noisy_trains.py --dataset dota_1024 --carve
"""
from __future__ import annotations

import argparse
import json
import sys

from src.data.layout import check_dataset_layout, check_noise_outputs
from src.data.paths import NOISE_FAMILIES, RATIO_GRID
from src.data.prepare import (
    DATA_CONFIG_DIR,
    generate_noisy_trains_for_config,
    load_data_config,
    resolve_data_config_paths,
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_check(report: dict) -> None:
    status = "ok" if report.get("ok") else "FAIL"
    print(f"[{status}] {report.get('name')}  {report.get('root', '')}")
    for split, info in (report.get("splits") or {}).items():
        print(f"  images/{split}: {info.get('n_files', 0)} files")
    for ann_dir, splits in (report.get("annotations") or {}).items():
        for split, summary in splits.items():
            print(
                f"  {ann_dir}/instances_{split}.json: "
                f"images={summary.get('n_images')} anns={summary.get('n_annotations')} "
                f"cats={summary.get('n_categories')}"
            )
    for warning in report.get("warnings") or []:
        print(f"  warning: {warning}")
    for error in report.get("errors") or []:
        print(f"  error: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirm dataset COCO layout, then generate noisy train JSONs.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Data YAML stem under configs/data/ (e.g. hit_uav). Use --all for every config.",
    )
    parser.add_argument("--all", action="store_true", help="Process every configs/data/*.yaml")
    parser.add_argument("--check-only", action="store_true", help="Validate layout and exit")
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
        help="For datasets without test, carve cal from train before the layout check and noise write",
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
        print(f"Unknown noise families {unknown}. Known: {list(NOISE_FAMILIES)}", file=sys.stderr)
        return 2
    ratios = tuple(int(r) for r in _parse_csv(args.ratios))
    paths = resolve_data_config_paths(dataset)
    if not paths:
        print(f"No data configs in {DATA_CONFIG_DIR}")
        return 1

    failed = False
    for path in paths:
        cfg = load_data_config(path)
        if args.carve and cfg.get("carve_cal"):
            from src.data.prepare import prepare_dataset

            prepare_dataset(cfg, force=args.force)
        layout = check_dataset_layout(cfg, allow_missing_cal=False)
        _print_check(layout)
        if not layout["ok"]:
            failed = True
            continue
        if args.check_only:
            continue
        noise = generate_noisy_trains_for_config(
            cfg,
            families=families,
            ratios=ratios,
            force=args.force,
            carve=False,
        )
        written = 0
        skipped = 0
        for block in noise.get("noise") or []:
            for run in block.get("runs") or []:
                if run.get("skipped"):
                    skipped += 1
                else:
                    written += 1
        print(f"  noise: wrote {written}, skipped existing {skipped}")
        after = check_noise_outputs(cfg, families=families, ratios=ratios)
        if not after["ok"]:
            failed = True
            for error in after["errors"]:
                print(f"  error: {error}")
        else:
            print(f"  noise outputs: {len(after['written'])} train JSONs")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
