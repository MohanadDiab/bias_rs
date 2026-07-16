"""Organize AI-TOD wo_xview images and filter complete COCO anns to available files."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "datasets" / "ai_tod"
ANN_SRC = AI / "complete_annotations"
IMG_SRC = AI / "images_wo_xview"

SPLIT_SOURCES = {
    "train": IMG_SRC / "aitod_wo_xview_train_imgs-001" / "images",
    "val": IMG_SRC / "aitod_wo_xview_trainval_imgs-003" / "images",  # filtered by val ann
    "test": IMG_SRC / "aitod_wo_xview_test_imgs-004" / "images",
}

ANN_FILES = {
    "train": ANN_SRC / "aitod_train.json",
    "val": ANN_SRC / "aitod_val.json",
    "test": ANN_SRC / "aitod_test_v1_1.0.json",
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def index_pngs(*dirs: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            mapping[p.name] = p
    return mapping


def filter_coco(ann_path: Path, available: dict[str, Path], out_json: Path, out_img_dir: Path) -> dict:
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    keep_images = []
    keep_ids = set()
    for im in data["images"]:
        name = Path(im["file_name"]).name
        if name not in available:
            continue
        keep_images.append({**im, "file_name": name})
        keep_ids.add(im["id"])
        link_or_copy(available[name], out_img_dir / name)

    keep_anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]

    out = {
        "info": data.get("info", {"description": "AI-TOD filtered to wo_xview images"}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": keep_images,
        "annotations": keep_anns,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)

    return {
        "src_images": len(data["images"]),
        "kept_images": len(keep_images),
        "src_annotations": len(data["annotations"]),
        "kept_annotations": len(keep_anns),
    }


def main() -> None:
    # Build a global index from all wo_xview packs (trainval overlaps train/val)
    all_dirs = [
        IMG_SRC / "aitod_wo_xview_train_imgs-001" / "images",
        IMG_SRC / "aitod_wo_xview_trainval_imgs-003" / "images",
        IMG_SRC / "aitod_wo_xview_test_imgs-004" / "images",
    ]
    available = index_pngs(*all_dirs)
    print(f"Available wo_xview pngs: {len(available)}")

    summary = {}
    for split, ann_path in ANN_FILES.items():
        out_img = AI / "images" / split
        out_ann = AI / "annotations" / f"instances_{split}.json"
        stats = filter_coco(ann_path, available, out_ann, out_img)
        summary[split] = stats
        print(split, stats)

    readme = AI / "README_LOCAL.md"
    readme.write_text(
        f"""# AI-TOD (processed, wo_xview subset)

- Raw annotations kept: `complete_annotations/`
- Raw images kept: `images_wo_xview/`
- Organized images: `images/{{train,val,test}}` (hardlinks/copies of available files)
- Filtered COCO: `annotations/instances_*.json`

## Note
Complete AI-TOD annotations reference more images than AI-TOD_wo_xview provides
(xView-derived patches). Those missing images are dropped from the filtered COCO.

## Counts
{json.dumps(summary, indent=2)}
""",
        encoding="utf-8",
    )
    print("AI-TOD processing complete")


if __name__ == "__main__":
    main()
