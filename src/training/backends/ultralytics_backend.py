"""Ultralytics YOLO detect backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.base import BaseTrainer
from src.training.data.coco_to_yolo import prepare_yolo_dataset

# Keys passed through to model.train(); Ultralytics keeps its own defaults for the rest.
# `device` is always auto-resolved from available CUDA GPUs (see resolve_train_device).
TRAIN_KEYS = (
    "imgsz",
    "batch",
    "epochs",
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


def resolve_train_device() -> str | list[int]:
    """Use every visible CUDA GPU; fall back to CPU when none are available.

    Returns a single index for one GPU, a list for multi-GPU (Ultralytics DDP),
    or ``\"cpu\"`` when CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    n = torch.cuda.device_count()
    if n <= 0:
        return "cpu"
    if n == 1:
        return 0
    return list(range(n))


class UltralyticsTrainer(BaseTrainer):
    def prepare(self, cfg: dict[str, Any]) -> Path:
        return prepare_yolo_dataset(cfg)

    def train(self, cfg: dict[str, Any], data_path: Path) -> Path:
        from ultralytics import YOLO

        model_name = cfg["model"]["name"]
        train_cfg = cfg["train"]
        project = Path(train_cfg.get("project", cfg["output"]["root"]))
        run_name = cfg["name"]
        device = resolve_train_device()

        kwargs: dict[str, Any] = {
            "data": str(data_path),
            "project": str(project),
            "name": run_name,
            "device": device,
        }
        for key in TRAIN_KEYS:
            if key in train_cfg:
                kwargs[key] = train_cfg[key]

        print(f"Training devices: {device}")
        model = YOLO(model_name)
        results = model.train(**kwargs)
        save_dir = Path(getattr(results, "save_dir", project / run_name))
        return save_dir.resolve()
