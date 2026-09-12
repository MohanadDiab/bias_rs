"""Ultralytics YOLO detect backend."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.data.coco import load_coco
from src.data.paths import resolve_split_json
from src.data.schema import Detection, dump_detections
from src.training.base import BaseTrainer
from src.training.data.coco_to_yolo import prepare_yolo_dataset
from src.training.weights import expert_id_from_cfg, resolve_weights

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
    """Use every visible CUDA GPU; fall back to CPU when none are available."""
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


def _write_run_meta(out_dir: Path, payload: dict[str, Any]) -> None:
    path = out_dir / "run_meta.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


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
        t0 = time.perf_counter()
        model = YOLO(model_name)
        results = model.train(**kwargs)
        save_dir = Path(getattr(results, "save_dir", project / run_name)).resolve()
        _write_run_meta(
            save_dir,
            {
                "backend": "ultralytics",
                "train_seconds": time.perf_counter() - t0,
                "model": model_name,
                "run_name": run_name,
            },
        )
        return save_dir

    def predict(self, cfg: dict[str, Any], data_path: Path, split: str) -> Path:
        from ultralytics import YOLO

        weights = resolve_weights(cfg)
        model = YOLO(weights)
        coco = load_coco(resolve_split_json(cfg, split))
        images_dir = Path(cfg["dataset"]["root"]) / "images" / split
        expert_id = expert_id_from_cfg(cfg)
        cat_ids = sorted(int(c["id"]) for c in coco["categories"])
        name_to_id = {Path(im["file_name"]).name: int(im["id"]) for im in coco["images"]}
        imgsz = cfg["train"]["imgsz"]
        device = resolve_train_device()

        t0 = time.perf_counter()
        dets: list[Detection] = []
        n_img = 0
        conf = float(cfg.get("predict", {}).get("conf", 0.3))
        results = model.predict(
            source=str(images_dir),
            imgsz=imgsz,
            device=device,
            conf=conf,
            stream=True,
            verbose=False,
        )
        for result in results:
            n_img += 1
            image_id = name_to_id.get(Path(result.path).name)
            if image_id is None or result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                cls_i = int(box.cls[0].item())
                category_id = cat_ids[cls_i] if 0 <= cls_i < len(cat_ids) else cls_i
                dets.append(
                    Detection(
                        image_id=image_id,
                        category_id=category_id,
                        bbox=[x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
                        score=float(box.conf[0].item()),
                        expert_id=expert_id,
                    )
                )
        elapsed = time.perf_counter() - t0
        project = Path(cfg["train"].get("project", cfg["output"]["root"]))
        out_dir = project / cfg["name"]
        out_path = dump_detections(dets, out_dir / f"predictions_{split}.json")
        _write_run_meta(
            out_dir,
            {
                f"predict_{split}_seconds": elapsed,
                f"predict_{split}_n_images": n_img,
                f"predict_{split}_n_dets": len(dets),
                "weights": weights,
                "ms_per_image": (elapsed / n_img * 1000.0) if n_img else None,
            },
        )
        return out_path
