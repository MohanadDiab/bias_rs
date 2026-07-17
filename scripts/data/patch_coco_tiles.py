"""Tile axis-aligned COCO datasets into overlapping patches.

Default: tile=640, overlap=20% (stride=512). Dimensions smaller than tile
are left at full extent (non-square patches). Images with both sides <= tile
are copied whole.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"


def window_starts(dim: int, tile: int, stride: int) -> list[int]:
    if dim <= tile:
        return [0]
    starts = list(range(0, dim - tile + 1, stride))
    last = dim - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def clip_bbox(
    bbox: list[float],
    left: int,
    up: int,
    right: int,
    down: int,
    area_thresh: float,
) -> list[float] | None:
    """Intersect COCO bbox [x,y,w,h] with tile; return tile-local bbox or None."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    ix1 = max(x, left)
    iy1 = max(y, up)
    ix2 = min(x2, right)
    iy2 = min(y2, down)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw < 2 or ih < 2:
        return None
    orig_area = max(w * h, 1e-6)
    inter_area = iw * ih
    if inter_area / orig_area < area_thresh:
        return None
    return [ix1 - left, iy1 - up, iw, ih]


def patch_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    tile: int,
    stride: int,
    area_thresh: float,
) -> dict:
    ann_path = src_root / "annotations" / f"instances_{split}.json"
    img_dir = src_root / "images" / split
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    img_by_id = {im["id"]: im for im in coco["images"]}

    out_img = dst_root / "images" / split
    out_ann = dst_root / "annotations" / f"instances_{split}.json"
    if out_img.exists():
        shutil.rmtree(out_img)
    out_img.mkdir(parents=True)

    out = {
        "info": {
            **coco.get("info", {}),
            "description": f"{src_root.name} patched tile={tile} stride={stride}",
        },
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [],
        "annotations": [],
    }
    next_img_id = 1
    next_ann_id = 1
    n_whole = 0
    n_tiles = 0

    for im in sorted(coco["images"], key=lambda x: x["id"]):
        name = Path(im["file_name"]).name
        src = img_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        with Image.open(src) as pil:
            pil = pil.convert("RGB")
            width, height = pil.size
            xs = window_starts(width, tile, stride)
            ys = window_starts(height, tile, stride)
            stem = Path(name).stem
            ext = Path(name).suffix.lower() or ".png"
            if ext not in {".png", ".jpg", ".jpeg", ".bmp"}:
                ext = ".png"

            whole = width <= tile and height <= tile
            if whole:
                n_whole += 1
            else:
                n_tiles += len(xs) * len(ys)

            for left in xs:
                for up in ys:
                    right = min(left + tile, width)
                    down = min(up + tile, height)
                    # When dim < tile, span full extent
                    crop_w = right - left
                    crop_h = down - up
                    crop = pil.crop((left, up, right, down))
                    patch_name = f"{stem}__{left}_{up}{ext}"
                    crop.save(out_img / patch_name)

                    out["images"].append(
                        {
                            "id": next_img_id,
                            "file_name": patch_name,
                            "width": crop_w,
                            "height": crop_h,
                        }
                    )
                    for ann in anns_by_img.get(im["id"], []):
                        clipped = clip_bbox(ann["bbox"], left, up, right, down, area_thresh)
                        if clipped is None:
                            continue
                        x, y, w, h = clipped
                        out["annotations"].append(
                            {
                                "id": next_ann_id,
                                "image_id": next_img_id,
                                "category_id": ann["category_id"],
                                "bbox": [x, y, w, h],
                                "area": float(w * h),
                                "iscrowd": ann.get("iscrowd", 0),
                                "segmentation": [],
                            }
                        )
                        next_ann_id += 1
                    next_img_id += 1

    out_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann, "w", encoding="utf-8") as f:
        json.dump(out, f)

    return {
        "split": split,
        "src_images": len(coco["images"]),
        "out_images": len(out["images"]),
        "out_annotations": len(out["annotations"]),
        "whole_images": n_whole,
        "tiled_windows": n_tiles,
    }


def patch_dataset(
    name: str,
    out_name: str,
    splits: list[str],
    tile: int,
    overlap: float,
    area_thresh: float,
) -> dict:
    src = DATASETS / name
    dst = DATASETS / out_name
    stride = max(1, int(round(tile * (1.0 - overlap))))
    print(f"{name} -> {out_name}: tile={tile} overlap={overlap} stride={stride}")
    summary = {}
    for split in splits:
        stats = patch_split(src, dst, split, tile, stride, area_thresh)
        summary[split] = stats
        print(split, stats)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hrsc2016_ms", "plant_detection", "all"],
    )
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--area-thresh", type=float, default=0.5)
    args = parser.parse_args()

    configs = {
        "hrsc2016_ms": ("hrsc2016_ms_640", ["train", "val", "test"]),
        "plant_detection": ("plant_detection_640", ["train", "val"]),
    }
    targets = list(configs) if args.dataset == "all" else [args.dataset]
    all_summary = {}
    for name in targets:
        out_name, splits = configs[name]
        all_summary[name] = patch_dataset(
            name, out_name, splits, args.tile, args.overlap, args.area_thresh
        )
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
