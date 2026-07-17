"""Convert COCO HBB annotations to Ultralytics YOLO label layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _safe_key(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def coco_bbox_to_yolo(
    bbox: list[float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """COCO [x, y, w, h] -> YOLO normalized (cx, cy, w, h)."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clip to valid range after rounding noise
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    if nw <= 0 or nh <= 0:
        return None
    return cx, cy, nw, nh


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def convert_split(
    coco_path: Path,
    images_dir: Path,
    labels_dir: Path,
    images_out_dir: Path,
) -> dict[str, Any]:
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_yolo = {c["id"]: i for i, c in enumerate(cats)}
    names = {i: c["name"] for i, c in enumerate(cats)}

    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    n_labels = 0
    n_images = 0
    for im in coco["images"]:
        file_name = Path(im["file_name"]).name
        src = images_dir / file_name
        if not src.exists():
            raise FileNotFoundError(src)
        _link_or_copy(src, images_out_dir / file_name)

        stem = Path(file_name).stem
        lines: list[str] = []
        w, h = int(im["width"]), int(im["height"])
        for ann in anns_by_img.get(im["id"], []):
            yolo_cls = cat_to_yolo.get(ann["category_id"])
            if yolo_cls is None:
                continue
            yolo = coco_bbox_to_yolo(ann["bbox"], w, h)
            if yolo is None:
                continue
            cx, cy, bw, bh = yolo
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_labels += 1
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        n_images += 1

    return {"images": n_images, "boxes": n_labels, "names": names}


def prepare_yolo_dataset(cfg: dict[str, Any]) -> Path:
    """Build YOLO cache + data.yaml; return path to data.yaml."""
    ds = cfg["dataset"]
    root = Path(ds["root"])
    ann_dir = root / ds["ann_dir"]
    train_split = ds["train_split"]
    val_split = ds["val_split"]

    cache_key = _safe_key(f"{Path(ds['root']).name}__{ds['ann_dir']}")
    cache_root = Path(cfg["output"]["root"]) / "_data_cache" / cache_key
    labels_root = cache_root / "labels"
    images_root = cache_root / "images"

    stats = {}
    names: dict[int, str] | None = None
    for split in (train_split, val_split):
        coco_path = ann_dir / f"instances_{split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(coco_path)
        split_stats = convert_split(
            coco_path,
            root / "images" / split,
            labels_root / split,
            images_root / split,
        )
        stats[split] = {k: split_stats[k] for k in ("images", "boxes")}
        names = split_stats["names"]

    assert names is not None
    data_yaml = {
        "path": str(cache_root.resolve()),
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "names": {int(k): v for k, v in names.items()},
    }
    out_yaml = cache_root / "data.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    meta = {"cache": str(cache_root), "stats": stats, "names": names}
    (cache_root / "prepare_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return out_yaml
