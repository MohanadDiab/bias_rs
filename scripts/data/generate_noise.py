"""Write train-only L/O/C/Mix COCO JSONs under annotations_noise/.

Val, test, and cal stay clean. Images are not copied.

    datasets/<name>/annotations_noise/<ann_dir>/<family>_<pct>/instances_train.json

Examples:

    uv run python scripts/data/generate_noise.py --all
    uv run python scripts/data/generate_noise.py --dataset hit_uav
    uv run python scripts/data/generate_noise.py --dataset dota_1024 --families L --ratios 10,30 --force
"""
from __future__ import annotations

import argparse
import json
import sys

from src.data.paths import NOISE_FAMILIES, RATIO_GRID
from src.data.prepare import (
    DATA_CONFIG_DIR,
    ann_dir_list,
    dataset_root,
    generate_noise,
    load_data_config,
    missing_cal_dirs,
    prepare_dataset,
    resolve_data_config_paths,
)
from src.data.splits import DEFAULT_SEED


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


def generate_for_config(
    cfg: dict[str, Any],
    *,
    families: tuple[str, ...],
    ratios: tuple[int, ...],
    force: bool,
    carve: bool,
) -> dict:
    name = cfg.get("name")
    root = dataset_root(cfg)
    if not root.exists():
        return {"name": name, "skipped": True, "reason": f"missing root {root}"}
    if bool(cfg.get("carve_cal", False)):
        missing = missing_cal_dirs(root, cfg)
        if missing:
            if not carve:
                raise SystemExit(
                    f"{name}: missing instances_{cfg.get('cal_split', 'cal')}.json "
                    f"in {missing}. Carve first: uv run bias-prepare --dataset {name} "
                    f"(or pass --carve)."
                )
            prepare_dataset(cfg, force=force)
    seed = int(cfg.get("seed", DEFAULT_SEED))
    noise_stats = []
    for ann_dir in ann_dir_list(cfg):
        train_json = root / ann_dir / "instances_train.json"
        if not train_json.exists():
            noise_stats.append({"ann_dir": ann_dir, "skipped": True, "reason": f"missing {train_json}"})
            continue
        noise_stats.append(
            {
                "ann_dir": ann_dir,
                "runs": generate_noise(
                    root,
                    ann_dir,
                    families=families,
                    ratios=ratios,
                    seed=seed,
                    force=force,
                ),
            }
        )
    return {"name": name, "root": str(root), "noise": noise_stats}


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
        report = generate_for_config(
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
