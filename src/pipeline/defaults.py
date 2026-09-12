"""Default recipe for a single end-to-end detection pipeline."""
from __future__ import annotations

from typing import Any

PIPELINE_DEFAULTS: dict[str, Any] = {
    "name": None,
    "dataset": None,
    "experts": ["yolo26n", "rfdetr_nano", "dino", "rtmdet"],
    "noise": {"family": "clean", "ratio": 0},
    "train": {},
    "pipeline": {
        "suppress": {
            "enabled": True,
            "method": "nms",
            "conf_thr": 0.3,
            "iou": 0.65,
            "sigma": 0.5,
        },
        "calibrate": {
            "enabled": True,
            "methods": ["ir", "ts", "platt", "lr", "beta", "dirichlet"],
        },
        "gate": {"enabled": False},
        "join": {"method": "wbf", "iou": 0.55},
    },
    "eval": {"split": "val", "plots": True},
    "output": {"root": "outputs/pipelines"},
}

EXPERT_DIR = "configs/training/models"
DATASET_DIR = "configs/training"
PIPELINE_DIR = "configs/pipelines"
