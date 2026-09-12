"""Prepare clean cal splits (DOTA 1024 / plant 640 only). Noise is written by scripts/data/generate_noise.py."""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.data.coco import category_names, load_coco, save_coco
from src.data.noise import apply_family
from src.data.paths import RATIO_GRID, noise_dir
from src.data.splits import DEFAULT_CAL_RATIO, DEFAULT_SEED, carve_shared_images
from src.training.config import ROOT

DATA_CONFIG_DIR = ROOT / "configs" / "data"


def load_data_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Data config must be a mapping: {path}")
    return cfg


def discover_data_configs() -> list[Path]:
    return sorted(p for p in DATA_CONFIG_DIR.glob("*.yaml") if p.name != "defaults.yaml")


def resolve_data_config_paths(dataset: str) -> list[Path]:
    if dataset == "all":
        return discover_data_configs()
    path = Path(dataset)
    if not path.suffix:
        path = DATA_CONFIG_DIR / f"{dataset}.yaml"
    if not path.is_absolute():
        candidate = ROOT / path
        path = candidate if candidate.exists() else path
    return [path]


def dataset_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["root"])
    if not root.is_absolute():
        root = ROOT / root
    return root


def ann_dir_list(cfg: dict[str, Any]) -> list[str]:
    dirs = list(cfg.get("ann_dirs") or [cfg.get("ann_dir")])
    return [d for d in dirs if d]


def write_class_map(root: Path, ann_dir: str, coco: dict[str, Any]) -> Path:
    out = root / ann_dir / "class_map.json"
    names = category_names(coco)
    payload = {
        "ann_dir": ann_dir,
        "classes": [{"id": i, "name": names[i]} for i in sorted(names)],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def generate_noise(
    root: Path,
    ann_dir: str,
    *,
    families: Sequence[str] = ("L", "O", "C", "Mix"),
    ratios: Sequence[int] = RATIO_GRID,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> list[dict[str, Any]]:
    train_json = root / ann_dir / "instances_train.json"
    coco = load_coco(train_json)
    write_class_map(root, ann_dir, coco)
    summaries = []
    seen: set[tuple[str, int]] = set()
    targets: list[tuple[str, int]] = []
    for family in families:
        for pct in ratios:
            key = (str(family), int(pct))
            if key[1] <= 0 or key in seen:
                continue
            seen.add(key)
            targets.append(key)
    for family, pct in targets:
        out_dir = noise_dir(root, ann_dir, family, pct)
        out_json = out_dir / "instances_train.json"
        meta_json = out_dir / "noise_meta.json"
        if out_json.exists() and not force:
            summaries.append({"skipped": True, "family": family, "pct": pct, "path": str(out_json)})
            continue
        rng = random.Random(seed + pct + sum(ord(c) for c in family))
        noisy, stats = apply_family(coco, family, pct, rng)
        save_coco(noisy, out_json)
        meta = {
            "noise_on": "train_only",
            "source_train": str(train_json),
            "seed": seed,
            **stats,
        }
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summaries.append({"skipped": False, "path": str(out_json), **stats})
    return summaries


def missing_cal_dirs(root: Path, cfg: dict[str, Any]) -> list[str]:
    cal_split = str(cfg.get("cal_split") or "cal")
    return [d for d in ann_dir_list(cfg) if not (root / d / f"instances_{cal_split}.json").exists()]


def prepare_dataset(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    root = dataset_root(cfg)
    if not root.exists():
        return {"name": cfg.get("name"), "skipped": True, "reason": f"missing root {root}"}
    if not bool(cfg.get("carve_cal", False)):
        return {
            "name": cfg.get("name"),
            "root": str(root),
            "cal": {"skipped": True, "reason": "carve_cal is false; using existing cal_split"},
        }
    ann_dirs = ann_dir_list(cfg)
    split_stats = carve_shared_images(
        root,
        ann_dirs,
        train_split=cfg.get("train_split", "train"),
        cal_split=cfg.get("cal_split", "cal"),
        cal_ratio=float(cfg.get("cal_ratio", DEFAULT_CAL_RATIO)),
        seed=int(cfg.get("seed", DEFAULT_SEED)),
        force=force,
    )
    return {
        "name": cfg.get("name"),
        "root": str(root),
        "cal": split_stats,
    }


def generate_noisy_trains_for_config(
    cfg: dict[str, Any],
    *,
    families: Sequence[str],
    ratios: Sequence[int],
    force: bool = False,
    carve: bool = False,
) -> dict[str, Any]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Carve a cal split from train for datasets without a native test split.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Data YAML stem under configs/data/ (or 'all')",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing carved cal split")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_data_config_paths(args.dataset)
    if not paths:
        print(f"No data configs in {DATA_CONFIG_DIR}")
        return 1
    reports = []
    for path in paths:
        cfg = load_data_config(path)
        report = prepare_dataset(cfg, force=args.force)
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("name", "root") if k in report}, indent=2))
    print(json.dumps(reports, indent=2, default=str))
    return 0
