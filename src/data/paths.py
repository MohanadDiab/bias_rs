"""Resolved annotation paths for clean vs noisy trains."""
from __future__ import annotations

from pathlib import Path
from typing import Any

NOISE_FAMILIES = ("L", "O", "C", "Mix")
RATIO_GRID = (0, 1, 2, 5, 10, 20, 30, 50)


def ratio_pct(ratio: float | int) -> int:
    """Interpret config ratios as integer percents (1 means 1%, not 100%)."""
    return int(round(float(ratio)))


def noise_dir(root: str | Path, ann_dir: str, family: str, pct: int) -> Path:
    return Path(root) / "annotations_noise" / ann_dir / f"{family}_{pct}"


def noise_train_json(root: str | Path, ann_dir: str, family: str, pct: int) -> Path:
    return noise_dir(root, ann_dir, family, pct) / "instances_train.json"


def clean_split_json(root: str | Path, ann_dir: str, split: str) -> Path:
    return Path(root) / ann_dir / f"instances_{split}.json"


def resolve_split_json(cfg: dict[str, Any], split: str) -> Path:
    """Train may come from annotations_noise/; val/cal/test stay in the clean ann_dir."""
    ds = cfg["dataset"]
    root = Path(ds["root"])
    ann_dir = ds["ann_dir"]
    noise = ds.get("noise") or {}
    family = noise.get("family")
    pct = ratio_pct(noise.get("ratio", 0)) if noise else 0
    if split == ds.get("train_split", "train") and family and pct > 0:
        path = noise_train_json(root, ann_dir, str(family), pct)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    return clean_split_json(root, ann_dir, split)
