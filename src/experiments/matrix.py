"""Expand experiment matrix YAMLs into concrete jobs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import RATIO_GRID
from src.training.config import ROOT, deep_merge, load_config, load_yaml

DATASET_CONFIGS = [
    ROOT / "configs/training/ai_tod_v1.yaml",
    ROOT / "configs/training/ai_tod_v2.yaml",
    ROOT / "configs/training/dota_1024_v1.yaml",
    ROOT / "configs/training/dota_1024_v15.yaml",
    ROOT / "configs/training/hit_uav.yaml",
    ROOT / "configs/training/hrsc2016_ms_640.yaml",
    ROOT / "configs/training/plant_detection_640.yaml",
]


def load_matrix(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Matrix must be a mapping: {path}")
    return data


def dataset_paths(matrix: dict[str, Any]) -> list[Path]:
    listed = matrix.get("datasets")
    if not listed or listed == "all":
        return [p for p in DATASET_CONFIGS if p.exists()]
    paths = []
    for item in listed:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        paths.append(path)
    return paths


def expert_specs(matrix: dict[str, Any], set_name: str = "asymmetric") -> list[dict[str, Any]]:
    experts = matrix.get("experts") or {}
    if isinstance(experts, list):
        return experts
    return list(experts.get(set_name) or experts.get("asymmetric") or [])


def ratio_list(matrix: dict[str, Any]) -> list[int]:
    ratios = matrix.get("noise_ratios", list(RATIO_GRID))
    return [int(r) for r in ratios]


def expand_jobs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    family = str(matrix.get("family", "L"))
    fusions = list(matrix.get("fusion") or ["wbf"])
    moes = list(matrix.get("moe") or ["simple"])
    expert_sets = list(matrix.get("expert_sets") or ["asymmetric"])
    jobs = []
    for ds_path in dataset_paths(matrix):
        for set_name in expert_sets:
            specs = expert_specs(matrix, set_name)
            if not specs:
                continue
            for pct in ratio_list(matrix):
                for fusion in fusions:
                    for moe in moes:
                        jobs.append(
                            {
                                "family": family if pct > 0 else "clean",
                                "pct": pct,
                                "dataset_yaml": str(ds_path),
                                "expert_set": set_name,
                                "experts": deepcopy(specs),
                                "fusion": fusion,
                                "moe": moe,
                            }
                        )
    return jobs


def compose_expert_cfg(
    dataset_yaml: str | Path,
    expert: dict[str, Any],
    family: str,
    pct: int,
) -> dict[str, Any]:
    cfg = load_config(dataset_yaml)
    overlay = {}
    model_yaml = expert.get("config")
    if model_yaml:
        path = Path(model_yaml)
        if not path.is_absolute():
            path = ROOT / path
        overlay = load_yaml(path)
    overlay = deep_merge(overlay, {k: v for k, v in expert.items() if k != "config"})
    if "backend" in overlay:
        cfg["backend"] = overlay["backend"]
    if "model" in overlay:
        cfg["model"] = deep_merge(cfg.get("model", {}), overlay["model"])
    elif "name" in overlay and isinstance(overlay["name"], str) and overlay["name"].endswith(".pt"):
        cfg["model"]["name"] = overlay["name"]
    expert_id = str(expert.get("expert_id") or overlay.get("expert_id") or Path(str(cfg["model"]["name"])).stem)
    cfg["expert_id"] = expert_id
    ds_name = Path(dataset_yaml).stem
    noise_tag = f"{family}_{pct}" if pct else "clean"
    cfg["name"] = f"{ds_name}__{expert_id}__{noise_tag}"
    if pct > 0 and family != "clean":
        cfg["dataset"]["noise"] = {"family": family, "ratio": pct}
    else:
        cfg["dataset"]["noise"] = {}
    cfg["output"]["root"] = str(Path(cfg["output"]["root"]))
    return cfg
