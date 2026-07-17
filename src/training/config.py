"""Load and merge training YAML configs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = ROOT / "configs" / "training" / "defaults.yaml"

REQUIRED_TOP = ("name", "backend", "dataset", "model", "train", "output")
REQUIRED_DATASET = ("root", "ann_dir", "train_split", "val_split")
REQUIRED_TRAIN = ("imgsz", "batch")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def validate_config(cfg: dict[str, Any], path: Path | None = None) -> None:
    loc = f" ({path})" if path else ""
    for key in REQUIRED_TOP:
        if key not in cfg:
            raise ValueError(f"Missing config key '{key}'{loc}")
    for key in REQUIRED_DATASET:
        if key not in cfg["dataset"]:
            raise ValueError(f"Missing dataset.{key}{loc}")
    for key in REQUIRED_TRAIN:
        if key not in cfg["train"]:
            raise ValueError(f"Missing train.{key}{loc}")
    if "name" not in cfg["model"]:
        raise ValueError(f"Missing model.name{loc}")


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: str | Path, defaults_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    defaults = Path(defaults_path).resolve() if defaults_path else DEFAULTS_PATH
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not defaults.exists():
        raise FileNotFoundError(defaults)

    cfg = deep_merge(load_yaml(defaults), load_yaml(config_path))
    validate_config(cfg, config_path)

    cfg["dataset"]["root"] = str(resolve_path(cfg["dataset"]["root"]))
    cfg["train"]["project"] = str(resolve_path(cfg["train"].get("project", cfg["output"]["root"])))
    cfg["output"]["root"] = str(resolve_path(cfg["output"]["root"]))
    cfg["_config_path"] = str(config_path)
    return cfg
