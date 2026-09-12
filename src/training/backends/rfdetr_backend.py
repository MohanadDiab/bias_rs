"""RF-DETR training / predict backend (official ``rfdetr`` package)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.data.coco import load_coco, save_coco
from src.data.files import link_or_copy
from src.data.paths import resolve_split_json
from src.data.schema import Detection, dump_detections
from src.training.base import BaseTrainer
from src.training.weights import expert_id_from_cfg, resolve_weights

RFDETR_CLASS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def _safe_key(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _model_key(cfg: dict[str, Any]) -> str:
    name = str(cfg["model"].get("name", "nano")).lower()
    name = name.replace("rfdetr", "").replace("-", "").replace("_", "").strip() or "nano"
    if name.endswith(".pt"):
        name = "nano"
    return name if name in RFDETR_CLASS else "nano"


def _load_rfdetr_class(key: str):
    import rfdetr

    cls_name = RFDETR_CLASS[key]
    if hasattr(rfdetr, cls_name):
        return getattr(rfdetr, cls_name)
    # older package aliases
    aliases = {
        "nano": ["RFDETRNano", "RFDETRBase"],
        "small": ["RFDETRSmall", "RFDETRBase"],
        "medium": ["RFDETRMedium", "RFDETRBase"],
        "base": ["RFDETRBase"],
        "large": ["RFDETRLarge", "RFDETRBase"],
    }
    for cand in aliases.get(key, ["RFDETRBase"]):
        if hasattr(rfdetr, cand):
            return getattr(rfdetr, cand)
    raise ImportError(f"rfdetr has no class for key '{key}'")


def _export_split(cfg: dict[str, Any], split: str, dest_dir: Path, ann_name: str) -> None:
    coco = load_coco(resolve_split_json(cfg, split))
    images_src = Path(cfg["dataset"]["root"]) / "images" / split
    dest_dir.mkdir(parents=True, exist_ok=True)
    exported = {**coco, "images": []}
    for im in coco["images"]:
        name = Path(im["file_name"]).name
        src = images_src / name
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, dest_dir / name)
        exported["images"].append({**im, "file_name": name})
    save_coco(exported, dest_dir / ann_name)


class RFDetrTrainer(BaseTrainer):
    def prepare(self, cfg: dict[str, Any]) -> Path:
        ds = cfg["dataset"]
        noise = ds.get("noise") or {}
        family = noise.get("family") or "clean"
        pct = int(round(float(noise.get("ratio", 0))))
        cache_key = _safe_key(f"{Path(ds['root']).name}__{ds['ann_dir']}__{family}_{pct}__rfdetr")
        cache_root = Path(cfg["output"]["root"]) / "_data_cache" / cache_key
        train_dir = cache_root / "train"
        valid_dir = cache_root / "valid"
        _export_split(cfg, ds["train_split"], train_dir, "_annotations.coco.json")
        _export_split(cfg, ds["val_split"], valid_dir, "_annotations.coco.json")
        (cache_root / "prepare_meta.json").write_text(
            json.dumps({"cache": str(cache_root)}, indent=2),
            encoding="utf-8",
        )
        return cache_root

    def train(self, cfg: dict[str, Any], data_path: Path) -> Path:
        key = _model_key(cfg)
        cls = _load_rfdetr_class(key)
        project = Path(cfg["train"].get("project", cfg["output"]["root"]))
        out_dir = project / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        model = cls()
        kwargs: dict[str, Any] = {
            "dataset_dir": str(data_path),
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch"],
            "output_dir": str(out_dir),
        }
        t0 = time.perf_counter()
        model.train(**kwargs)
        meta = {
            "backend": "rfdetr",
            "train_seconds": time.perf_counter() - t0,
            "model": key,
            "run_name": cfg["name"],
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir.resolve()

    def predict(self, cfg: dict[str, Any], data_path: Path, split: str) -> Path:
        from PIL import Image

        key = _model_key(cfg)
        cls = _load_rfdetr_class(key)
        weights = resolve_weights(cfg)
        init_kwargs: dict[str, Any] = {}
        if Path(weights).exists() and Path(weights).suffix in {".pth", ".pt"}:
            init_kwargs["pretrain_weights"] = weights
        try:
            model = cls(**init_kwargs) if init_kwargs else cls()
        except TypeError:
            model = cls()
        coco = load_coco(resolve_split_json(cfg, split))
        images_dir = Path(cfg["dataset"]["root"]) / "images" / split
        expert_id = expert_id_from_cfg(cfg)
        cat_ids = sorted(int(c["id"]) for c in coco["categories"])
        dets: list[Detection] = []
        t0 = time.perf_counter()
        for im in coco["images"]:
            name = Path(im["file_name"]).name
            path = images_dir / name
            image = Image.open(path).convert("RGB")
            raw = model.predict(image, threshold=float(cfg.get("predict", {}).get("conf", 0.3)))
            xyxy, scores, classes = _unpack_rfdetr(raw)
            for box, score, cls_i in zip(xyxy, scores, classes, strict=True):
                x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                category_id = cat_ids[int(cls_i)] if 0 <= int(cls_i) < len(cat_ids) else int(cls_i)
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
        project = Path(cfg["train"].get("project", cfg["output"]["root"]))
        out_dir = project / cfg["name"]
        out_path = dump_detections(dets, out_dir / f"predictions_{split}.json")
        n_img = len(coco["images"])
        meta_path = out_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update(
            {
                f"predict_{split}_seconds": elapsed,
                f"predict_{split}_n_images": n_img,
                f"predict_{split}_n_dets": len(dets),
                "ms_per_image": (elapsed / n_img * 1000.0) if n_img else None,
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_path


def _unpack_rfdetr(raw: Any) -> tuple[list, list, list]:
    if raw is None:
        return [], [], []
    if hasattr(raw, "xyxy"):
        xyxy = raw.xyxy.tolist() if hasattr(raw.xyxy, "tolist") else list(raw.xyxy)
        scores = raw.confidence.tolist() if hasattr(raw.confidence, "tolist") else list(raw.confidence)
        classes = raw.class_id.tolist() if hasattr(raw.class_id, "tolist") else list(raw.class_id)
        return xyxy, scores, classes
    if isinstance(raw, dict):
        return raw.get("xyxy", []), raw.get("scores", raw.get("confidence", [])), raw.get("labels", raw.get("class_id", []))
    if isinstance(raw, (list, tuple)) and raw and hasattr(raw[0], "xyxy"):
        return _unpack_rfdetr(raw[0])
    return [], [], []
