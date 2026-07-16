"""Build dota_v1 / dota_v1.5 processed trees (images + labelTxt + COCO) via DOTA_devkit."""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DOTA_devkit"))
import dota_utils as util  # noqa: E402

DOTA_RAW = ROOT / "datasets" / "dota"
OUT_V1 = ROOT / "datasets" / "dota_v1"
OUT_V15 = ROOT / "datasets" / "dota_v1.5"

CLASSES_V1 = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]
CLASSES_V15 = CLASSES_V1 + ["container-crane"]

SPLITS = {
    "train": {
        "image_globs": [
            DOTA_RAW / "images" / "part1" / "images",
            DOTA_RAW / "images" / "part2" / "images",
            DOTA_RAW / "images" / "part3" / "images",
        ],
        "zip_v1": DOTA_RAW
        / "labelTxt-v1.0-20260716T161503Z-1-001"
        / "labelTxt-v1.0"
        / "labelTxt.zip",
        "zip_v15": DOTA_RAW
        / "labelTxt-v1.5-20260716T161504Z-1-001"
        / "labelTxt-v1.5"
        / "DOTA-v1.5_train.zip",
    },
    "val": {
        "image_globs": [DOTA_RAW / "val" / "part1" / "images"],
        "zip_v1": DOTA_RAW
        / "val"
        / "labelTxt-v1.0-20260716T165621Z-1-001"
        / "labelTxt-v1.0"
        / "labelTxt.zip",
        "zip_v15": DOTA_RAW
        / "val"
        / "labelTxt-v1.5-20260716T165623Z-1-001"
        / "labelTxt-v1.5"
        / "DOTA-v1.5_val.zip",
    },
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def collect_images(globs: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for img_dir in globs:
        if not img_dir.exists():
            continue
        for p in img_dir.glob("*.png"):
            mapping[p.name] = p
    return mapping


def extract_labels(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.endswith(".txt"):
                continue
            target = dest / name
            if target.exists():
                continue
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def link_images(image_map: dict[str, Path], dest: Path, label_dir: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for label in sorted(label_dir.glob("*.txt")):
        img_name = label.stem + ".png"
        src = image_map.get(img_name)
        if src is None:
            continue
        link_or_copy(src, dest / img_name)
        n += 1
    return n


def dota2coco_devkit(
    images_dir: Path,
    label_dir: Path,
    dest_json: Path,
    classes: list[str],
    version: str,
) -> dict:
    """DOTA2COCO-style export using DOTA_devkit parsers (parse_dota_poly2)."""
    data_dict: dict = {
        "info": {
            "contributor": "captain group",
            "description": f"DOTA {version} processed for bias_rs via DOTA_devkit",
            "url": "http://captain.whu.edu.cn/DOTAweb/",
            "version": version,
            "year": 2026,
        },
        "images": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": name}
            for i, name in enumerate(classes)
        ],
        "annotations": [],
    }
    class_to_id = {name: i + 1 for i, name in enumerate(classes)}

    inst_count = 1
    image_id = 1
    skipped_labels = 0

    label_files = sorted(label_dir.glob("*.txt"))
    for label_path in label_files:
        basename = util.custombasename(str(label_path))
        imagepath = images_dir / f"{basename}.png"
        if not imagepath.exists():
            skipped_labels += 1
            continue

        img = cv2.imread(str(imagepath))
        if img is None:
            skipped_labels += 1
            continue
        height, width = img.shape[:2]

        data_dict["images"].append(
            {
                "file_name": f"{basename}.png",
                "id": image_id,
                "width": width,
                "height": height,
            }
        )

        objects = util.parse_dota_poly2(str(label_path))
        for obj in objects:
            name = obj["name"]
            if name not in class_to_id:
                continue
            poly = obj["poly"]
            xmin = min(poly[0::2])
            ymin = min(poly[1::2])
            xmax = max(poly[0::2])
            ymax = max(poly[1::2])
            bw = xmax - xmin
            bh = ymax - ymin
            data_dict["annotations"].append(
                {
                    "id": inst_count,
                    "image_id": image_id,
                    "category_id": class_to_id[name],
                    "area": float(obj["area"]),
                    "segmentation": [poly],
                    "iscrowd": 0,
                    "bbox": [xmin, ymin, bw, bh],
                }
            )
            inst_count += 1
        image_id += 1

    dest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(data_dict, f)

    return {
        "images": len(data_dict["images"]),
        "annotations": len(data_dict["annotations"]),
        "skipped_labels": skipped_labels,
        "classes": classes,
    }


def process_split(
    out_root: Path,
    split: str,
    zip_path: Path,
    classes: list[str],
    version: str,
    image_map: dict[str, Path],
) -> dict:
    label_dir = out_root / "labelTxt" / split
    images_dir = out_root / "images" / split
    extract_labels(zip_path, label_dir)
    n_img = link_images(image_map, images_dir, label_dir)
    stats = dota2coco_devkit(
        images_dir,
        label_dir,
        out_root / "annotations" / f"instances_{split}.json",
        classes,
        version,
    )
    stats["linked_images"] = n_img
    stats["split"] = split
    print(f"{version} {split}", stats)
    return stats


def write_readme(path: Path, version: str, all_stats: list[dict]) -> None:
    lines = [
        f"# DOTA {version} (processed)",
        "",
        "- Source raw tree: `datasets/dota` (untouched)",
        "- Parser: `DOTA_devkit` (`dota_utils.parse_dota_poly2`, DOTA2COCO-style export)",
        "- OBB labels: `labelTxt/{{split}}`",
        "- COCO: `annotations/instances_{{split}}.json`",
        "",
        "## Splits",
    ]
    for s in all_stats:
        lines.extend(
            [
                f"### {s['split']}",
                f"- Images: {s['images']}",
                f"- Annotations: {s['annotations']}",
                f"- Linked images: {s['linked_images']}",
                "",
            ]
        )
    classes = all_stats[0]["classes"] if all_stats else []
    lines.append(f"## Classes ({len(classes)})")
    lines.append(", ".join(classes))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_version(
    out_root: Path,
    version: str,
    classes: list[str],
    zip_key: str,
) -> list[dict]:
    stats = []
    for split, cfg in SPLITS.items():
        image_map = collect_images(cfg["image_globs"])
        zip_path = cfg[zip_key]
        if not zip_path.exists():
            print(f"SKIP {version} {split}: missing {zip_path}")
            continue
        stats.append(process_split(out_root, split, zip_path, classes, version, image_map))
    write_readme(out_root / "README_LOCAL.md", version, stats)
    return stats


def main() -> None:
    process_version(OUT_V1, "v1.0", CLASSES_V1, "zip_v1")
    process_version(OUT_V15, "v1.5", CLASSES_V15, "zip_v15")
    print("DOTA processing complete")


if __name__ == "__main__":
    main()
