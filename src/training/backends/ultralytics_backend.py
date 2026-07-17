"""Ultralytics YOLO detect backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.base import BaseTrainer
from src.training.data.coco_to_yolo import prepare_yolo_dataset

# Keys passed through to model.train(); Ultralytics keeps its own defaults for the rest.
TRAIN_KEYS = (
    "imgsz",
    "batch",
    "epochs",
    "device",
    "workers",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "patience",
    "seed",
    "cos_lr",
    "close_mosaic",
    "amp",
    "exist_ok",
    "pretrained",
    "resume",
)


class UltralyticsTrainer(BaseTrainer):
    def prepare(self, cfg: dict[str, Any]) -> Path:
        return prepare_yolo_dataset(cfg)

    def train(self, cfg: dict[str, Any], data_path: Path) -> Path:
        from ultralytics import YOLO

        model_name = cfg["model"]["name"]
        train_cfg = cfg["train"]
        project = Path(train_cfg.get("project", cfg["output"]["root"]))
        run_name = cfg["name"]

        kwargs: dict[str, Any] = {
            "data": str(data_path),
            "project": str(project),
            "name": run_name,
        }
        for key in TRAIN_KEYS:
            if key in train_cfg:
                kwargs[key] = train_cfg[key]

        model = YOLO(model_name)
        results = model.train(**kwargs)
        save_dir = Path(getattr(results, "save_dir", project / run_name))
        return save_dir.resolve()
