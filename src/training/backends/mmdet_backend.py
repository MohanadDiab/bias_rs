"""MMDetection backend for DINO and RTMDet, with Tiny-FPN fallback."""
from __future__ import annotations

import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

from src.data.coco import load_coco, save_coco
from src.data.files import link_or_copy
from src.data.paths import resolve_split_json
from src.data.schema import Detection, dump_detections
from src.training.base import BaseTrainer
from src.training.backends.tiny_fpn import TinyFPNTrainer
from src.training.weights import expert_id_from_cfg, resolve_weights


def mmdet_available() -> bool:
    try:
        import mmdet  # noqa: F401
        import mmengine  # noqa: F401
    except ImportError:
        return False
    return True


def _safe_key(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def architecture_from_cfg(cfg: dict[str, Any]) -> str:
    name = str(cfg["model"].get("name", "dino")).lower()
    if "rtmdet" in name:
        return "rtmdet"
    return "dino"


class MMDetTrainer(BaseTrainer):
    """Train DINO or RTMDet via MMDetection when installed; else Tiny-FPN."""

    def __init__(self) -> None:
        self._fallback = None if mmdet_available() else TinyFPNTrainer()

    def prepare(self, cfg: dict[str, Any]) -> Path:
        if self._fallback is not None:
            return self._fallback.prepare(cfg)
        ds = cfg["dataset"]
        noise = ds.get("noise") or {}
        family = noise.get("family") or "clean"
        pct = int(round(float(noise.get("ratio", 0))))
        cache_key = _safe_key(f"{Path(ds['root']).name}__{ds['ann_dir']}__{family}_{pct}__mmdet")
        cache = Path(cfg["output"]["root"]) / "_data_cache" / cache_key
        for split, dest in (
            (ds["train_split"], cache / "train"),
            (ds["val_split"], cache / "val"),
        ):
            dest.mkdir(parents=True, exist_ok=True)
            coco = load_coco(resolve_split_json(cfg, split))
            img_src = Path(ds["root"]) / "images" / split
            images = []
            for im in coco["images"]:
                name = Path(im["file_name"]).name
                link_or_copy(img_src / name, dest / name)
                images.append({**im, "file_name": name})
            save_coco({**coco, "images": images}, dest / "annotations.json")
        (cache / "prepare_meta.json").write_text(json.dumps({"cache": str(cache)}, indent=2), encoding="utf-8")
        return cache

    def train(self, cfg: dict[str, Any], data_path: Path) -> Path:
        if self._fallback is not None:
            warnings.warn("mmdet is not installed; using tiny_fpn fallback", RuntimeWarning, stacklevel=2)
            cfg = {**cfg, "model": {**cfg.get("model", {}), "name": architecture_from_cfg(cfg)}}
            return self._fallback.train(cfg, data_path)
        from mmengine.config import Config
        from mmengine.runner import Runner

        out_dir = Path(cfg["train"].get("project", cfg["output"]["root"])) / cfg["name"]
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        mm_cfg = _build_mmdet_config(cfg, data_path, out_dir)
        cfg_file = out_dir / "mmdet_config.py"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(mm_cfg, encoding="utf-8")
        t0 = time.perf_counter()
        runner = Runner.from_cfg(Config.fromfile(str(cfg_file)))
        runner.train()
        (out_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "backend": "mmdet",
                    "arch": architecture_from_cfg(cfg),
                    "train_seconds": time.perf_counter() - t0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_dir.resolve()

    def predict(self, cfg: dict[str, Any], data_path: Path, split: str) -> Path:
        if self._fallback is not None:
            return self._fallback.predict(cfg, data_path, split)
        from mmdet.apis import inference_detector, init_detector

        arch = architecture_from_cfg(cfg)
        out_dir = Path(cfg["train"].get("project", cfg["output"]["root"])) / cfg["name"]
        config_file = out_dir / "mmdet_config.py"
        weights = resolve_weights(cfg)
        model = init_detector(str(config_file), weights, device="cuda:0")
        coco = load_coco(resolve_split_json(cfg, split))
        images_dir = Path(cfg["dataset"]["root"]) / "images" / split
        expert_id = expert_id_from_cfg(cfg)
        cat_ids = sorted(int(c["id"]) for c in coco["categories"])
        dets: list[Detection] = []
        t0 = time.perf_counter()
        conf = float(cfg.get("predict", {}).get("conf", 0.3))
        for im in coco["images"]:
            result = inference_detector(model, str(images_dir / Path(im["file_name"]).name))
            pred = getattr(result, "pred_instances", result)
            bboxes = _to_numpy(getattr(pred, "bboxes", []))
            scores = _to_numpy(getattr(pred, "scores", []))
            labels = _to_numpy(getattr(pred, "labels", []))
            for box, score, lab in zip(bboxes, scores, labels, strict=False):
                if float(score) < conf:
                    continue
                x1, y1, x2, y2 = [float(x) for x in box[:4]]
                lab_i = int(lab)
                category_id = cat_ids[lab_i] if 0 <= lab_i < len(cat_ids) else lab_i
                dets.append(
                    Detection(
                        image_id=int(im["id"]),
                        category_id=category_id,
                        bbox=[x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
                        score=float(score),
                        expert_id=expert_id,
                    )
                )
        elapsed = time.perf_counter() - t0
        path = dump_detections(dets, out_dir / f"predictions_{split}.json")
        n_img = max(len(coco["images"]), 1)
        (out_dir / "run_meta.json").write_text(
            json.dumps({"backend": "mmdet", "arch": arch, "ms_per_image": elapsed / n_img * 1000.0}, indent=2),
            encoding="utf-8",
        )
        return path


def _to_numpy(value: Any):
    if value is None:
        return []
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _build_mmdet_config(cfg: dict[str, Any], data_path: Path, out_dir: Path) -> str:
    """Minimal COCO-style MMDet 3.x config; relies on zoo base configs when present."""
    arch = architecture_from_cfg(cfg)
    coco = load_coco(data_path / "train" / "annotations.json")
    num_classes = len(coco["categories"])
    imgsz = int(cfg["train"]["imgsz"])
    epochs = int(cfg["train"]["epochs"])
    batch = int(cfg["train"]["batch"])
    metainfo = {"classes": [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]}
    base = "dino-4scale_r50_8xb2-12e_coco.py" if arch == "dino" else "rtmdet_s_8xb32-300e_coco.py"
    return f"""
default_scope = 'mmdet'
work_dir = r'{out_dir.as_posix()}'
metainfo = {metainfo!r}
num_classes = {num_classes}
imgsz = {imgsz}

try:
    _base_ = ['{base}']
except Exception:
    _base_ = []

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={epochs}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

dataset_type = 'CocoDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(imgsz, imgsz), keep_ratio=True),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    batch_size={max(batch, 1)},
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=r'{data_path.as_posix()}',
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=r'{data_path.as_posix()}',
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=train_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=r'{(data_path / "val" / "annotations.json").as_posix()}', metric='bbox')
test_evaluator = val_evaluator
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=16)
"""
