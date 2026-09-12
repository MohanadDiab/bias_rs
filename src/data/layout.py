"""Confirm the on-disk COCO layout used by training and noise generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.coco import load_coco
from src.data.prepare import ann_dir_list, dataset_root
from src.data.paths import noise_train_json

REQUIRED_COCO_KEYS = ("images", "annotations", "categories")


def _err(report: dict[str, Any], message: str) -> None:
    report["ok"] = False
    report["errors"].append(message)


def _warn(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def check_coco_json(path: Path, *, images_dir: Path, report: dict[str, Any], loc: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path)}
    try:
        coco = load_coco(path)
    except (OSError, ValueError) as exc:
        _err(report, f"{loc}: cannot read {path}: {exc}")
        return summary
    for key in REQUIRED_COCO_KEYS:
        if key not in coco:
            _err(report, f"{loc}: missing COCO key '{key}' in {path.name}")
    images = coco.get("images") or []
    anns = coco.get("annotations") or []
    cats = coco.get("categories") or []
    summary.update({"n_images": len(images), "n_annotations": len(anns), "n_categories": len(cats)})
    ids = [int(im["id"]) for im in images if "id" in im]
    if len(ids) != len(set(ids)):
        _err(report, f"{loc}: duplicate image ids in {path.name}")
    id_set = set(ids)
    missing_files = 0
    bad_names = 0
    for im in images:
        raw = str(im.get("file_name", ""))
        if Path(raw).name != raw or not raw:
            bad_names += 1
        name = Path(raw).name
        if not name or not (images_dir / name).exists():
            missing_files += 1
    if bad_names:
        _err(report, f"{loc}: {bad_names} file_name values are not basenames in {path.name}")
    if missing_files:
        _err(
            report,
            f"{loc}: {missing_files} images in {path.name} are missing under {images_dir}",
        )
    dangling = 0
    for ann in anns:
        if int(ann.get("image_id", -1)) not in id_set:
            dangling += 1
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            _err(report, f"{loc}: annotation {ann.get('id')} has bbox that is not [x, y, w, h]")
            break
    if dangling:
        _err(report, f"{loc}: {dangling} annotations point at missing image ids in {path.name}")
    return summary


def check_dataset_layout(
    cfg: dict[str, Any],
    *,
    allow_missing_cal: bool = False,
) -> dict[str, Any]:
    """Validate images/{train,val[,test|cal]} and <ann_dir>/instances_<split>.json."""
    name = str(cfg.get("name") or "dataset")
    root = dataset_root(cfg)
    train_split = str(cfg.get("train_split") or "train")
    val_split = str(cfg.get("val_split") or "val")
    cal_split = str(cfg.get("cal_split") or "test")
    carve = bool(cfg.get("carve_cal", False))
    report: dict[str, Any] = {
        "ok": True,
        "name": name,
        "root": str(root),
        "errors": [],
        "warnings": [],
        "splits": {},
    }
    if not root.exists():
        _err(report, f"{name}: dataset root does not exist: {root}")
        return report
    images_root = root / "images"
    if not images_root.is_dir():
        _err(report, f"{name}: missing images/ directory")
        return report
    required_img = [train_split, val_split]
    if not (carve and allow_missing_cal):
        required_img.append(cal_split)
    for split in required_img:
        split_dir = images_root / split
        if not split_dir.is_dir():
            _err(report, f"{name}: missing images/{split}/")
            continue
        n_files = sum(1 for p in split_dir.iterdir() if p.is_file())
        report["splits"][split] = {"images_dir": str(split_dir), "n_files": n_files}
        if n_files == 0:
            _err(report, f"{name}: images/{split}/ is empty")

    required_json = [train_split, val_split]
    if not (carve and allow_missing_cal):
        required_json.append(cal_split)
    for ann_dir in ann_dir_list(cfg):
        ann_path = root / ann_dir
        loc = f"{name}/{ann_dir}"
        if not ann_path.is_dir():
            _err(report, f"{loc}: annotation directory missing")
            continue
        checkpoints = list(ann_path.glob("**/.ipynb_checkpoints/**"))
        if checkpoints:
            _warn(report, f"{loc}: found .ipynb_checkpoints (ignored by training, but remove them)")
        for split in required_json:
            json_path = ann_path / f"instances_{split}.json"
            if not json_path.exists():
                _err(report, f"{loc}: missing {json_path.name}")
                continue
            img_dir = images_root / split
            summary = check_coco_json(json_path, images_dir=img_dir, report=report, loc=f"{loc}/{split}")
            report.setdefault("annotations", {}).setdefault(ann_dir, {})[split] = summary
    return report


def check_noise_outputs(
    cfg: dict[str, Any],
    *,
    families: tuple[str, ...],
    ratios: tuple[int, ...],
) -> dict[str, Any]:
    """After generation: noisy train JSONs exist; val/cal JSONs are still in the clean ann dir."""
    name = str(cfg.get("name") or "dataset")
    root = dataset_root(cfg)
    report: dict[str, Any] = {"ok": True, "name": name, "errors": [], "warnings": [], "written": []}
    for ann_dir in ann_dir_list(cfg):
        val_json = root / ann_dir / f"instances_{cfg.get('val_split', 'val')}.json"
        cal_json = root / ann_dir / f"instances_{cfg.get('cal_split', 'test')}.json"
        if not val_json.exists():
            _err(report, f"{name}/{ann_dir}: clean val JSON missing after noise write")
        if not cal_json.exists():
            _err(report, f"{name}/{ann_dir}: clean cal JSON missing after noise write")
        for family in families:
            for pct in ratios:
                if int(pct) <= 0:
                    continue
                path = noise_train_json(root, ann_dir, str(family), int(pct))
                if not path.exists():
                    _err(report, f"{name}/{ann_dir}: missing noisy train {path}")
                else:
                    report["written"].append(str(path))
    return report
