"""Convert Dataset Ninja Supervisely exports under datasets/ to uniform COCO layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

# Dataset root -> list of Supervisely dataset folder names (splits)
CONFIGS = {
    "hit_uav": ["train", "val", "test"],
    "hrsc2016_ms": ["train", "val", "test"],
    "plant_detection": ["ds0"],  # single pack; map to train
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def load_meta(dataset_root: Path) -> list[str]:
    meta = json.loads((dataset_root / "meta.json").read_text(encoding="utf-8"))
    return [c["title"] for c in meta["classes"]]


def parse_ann(ann_path: Path) -> list[dict]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    objects = []
    for obj in data.get("objects", []):
        # Dataset Ninja Supervisely: points may be top-level or under geometry
        exterior = None
        if "points" in obj and isinstance(obj["points"], dict):
            exterior = obj["points"].get("exterior")
        elif "geometry" in obj and isinstance(obj["geometry"], dict):
            pts = obj["geometry"].get("points", {})
            if isinstance(pts, dict):
                exterior = pts.get("exterior")
        if not exterior or len(exterior) < 2:
            continue
        xs = [p[0] for p in exterior]
        ys = [p[1] for p in exterior]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        objects.append(
            {
                "classTitle": obj.get("classTitle"),
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": max(0.0, (xmax - xmin) * (ymax - ymin)),
            }
        )
    return objects


def convert_split(
    dataset_root: Path,
    sly_split: str,
    coco_split: str,
    class_to_id: dict[str, int],
    categories: list[dict],
) -> dict:
    img_dir = dataset_root / sly_split / "img"
    ann_dir = dataset_root / sly_split / "ann"
    out_img = dataset_root / "images" / coco_split
    out_ann = dataset_root / "annotations" / f"instances_{coco_split}.json"
    out_img.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": f"{dataset_root.name} {coco_split}", "year": 2026},
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    image_id = 1
    ann_id = 1

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        ann_path = ann_dir / f"{img_path.name}.json"
        with Image.open(img_path) as im:
            width, height = im.size
        link_or_copy(img_path, out_img / img_path.name)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )
        if ann_path.exists():
            for obj in parse_ann(ann_path):
                title = obj["classTitle"]
                if title not in class_to_id:
                    continue
                bbox = obj["bbox"]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_to_id[title],
                        "bbox": bbox,
                        "area": obj["area"],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        image_id += 1

    out_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return {
        "split": coco_split,
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
    }


def convert_dataset(name: str, sly_splits: list[str]) -> dict:
    root = DATASETS / name
    classes = load_meta(root)
    categories = [{"id": i + 1, "name": c, "supercategory": c} for i, c in enumerate(classes)]
    class_to_id = {c: i + 1 for i, c in enumerate(classes)}
    stats = []
    for sly_split in sly_splits:
        coco_split = "train" if sly_split == "ds0" else sly_split
        stats.append(convert_split(root, sly_split, coco_split, class_to_id, categories))
    summary = {"classes": classes, "splits": stats}
    (root / "README_LOCAL.md").write_text(
        f"""# {name} (processed COCO)

- Raw Supervisely export kept in original split folders (`train`/`val`/`test` or `ds0`)
- Uniform COCO layout: `images/{{split}}` + `annotations/instances_{{split}}.json`

## Counts
{json.dumps(summary, indent=2)}
""",
        encoding="utf-8",
    )
    print(name, summary)
    return summary


def main() -> None:
    for name, splits in CONFIGS.items():
        convert_dataset(name, splits)
    print("Supervisely -> COCO conversion complete")


if __name__ == "__main__":
    main()
