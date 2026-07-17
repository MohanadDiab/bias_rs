"""Patch DOTA into 1024x1024 tiles via DOTA_devkit ImgSplit (gap=200).

Round-trip: COCO OBB -> labelTxt -> splitbase -> COCO.
Assembles datasets/dota_1024 with shared images + annotations_v1 / annotations_v1.5.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DOTA_devkit"))
import dota_utils as util  # noqa: E402
from ImgSplit_multi_process import splitbase  # noqa: E402

DOTA = ROOT / "datasets" / "dota"
OUT = ROOT / "datasets" / "dota_1024"
WORK = DOTA / "_split_work"

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

VERSIONS = {
    "v1": ("annotations_v1", CLASSES_V1),
    "v1.5": ("annotations_v1.5", CLASSES_V15),
}
SPLITS = ("train", "val")
SUBSIZE = 1024
GAP = 200


def coco_to_labeltxt(coco: dict, label_dir: Path) -> None:
    """Write DOTA labelTxt files from COCO OBB segmentation."""
    label_dir.mkdir(parents=True, exist_ok=True)
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        lines = []
        for ann in anns_by_img.get(im["id"], []):
            seg = ann.get("segmentation")
            if not isinstance(seg, list) or not seg or not isinstance(seg[0], list):
                continue
            poly = seg[0]
            if len(poly) < 8:
                continue
            name = cat_by_id.get(ann["category_id"], "unknown")
            coords = " ".join(str(float(x)) for x in poly[:8])
            lines.append(f"{coords} {name} 0")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def stage_split_input(ann_dir_name: str, split: str, work_in: Path) -> Path:
    """Build work_in/{images,labelTxt} for one version+split."""
    img_src = DOTA / "images" / split
    ann_path = DOTA / ann_dir_name / f"instances_{split}.json"
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    images_dir = work_in / "images"
    label_dir = work_in / "labelTxt"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    images_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Hardlink/copy images into flat images/ for splitbase
    for im in coco["images"]:
        name = Path(im["file_name"]).name
        src = img_src / name
        dst = images_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

    coco_to_labeltxt(coco, label_dir)
    return work_in


def labeltxt_to_coco(
    images_dir: Path,
    label_dir: Path,
    classes: list[str],
    description: str,
) -> dict:
    """DOTA2COCO-style export from split patches."""
    class_to_id = {name: i + 1 for i, name in enumerate(classes)}
    data: dict = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2026,
        },
        "images": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": name} for i, name in enumerate(classes)
        ],
        "annotations": [],
    }
    inst_count = 1
    image_id = 1
    label_files = sorted(label_dir.glob("*.txt"))
    for label_path in label_files:
        basename = util.custombasename(str(label_path))
        imagepath = images_dir / f"{basename}.png"
        if not imagepath.exists():
            continue
        img = cv2.imread(str(imagepath))
        if img is None:
            continue
        height, width = img.shape[:2]
        data["images"].append(
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
            # Skip heavily truncated instances marked difficult=2 by splitter
            if str(obj.get("difficult", "0")) == "2":
                continue
            poly = obj["poly"]
            xmin = min(poly[0::2])
            ymin = min(poly[1::2])
            xmax = max(poly[0::2])
            ymax = max(poly[1::2])
            bw = xmax - xmin
            bh = ymax - ymin
            data["annotations"].append(
                {
                    "id": inst_count,
                    "image_id": image_id,
                    "category_id": class_to_id[name],
                    "area": float(bw * bh),
                    "segmentation": [],
                    "iscrowd": 0,
                    "bbox": [xmin, ymin, bw, bh],
                }
            )
            inst_count += 1
        image_id += 1
    return data


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def process_version_split(
    version_key: str,
    ann_dir_name: str,
    classes: list[str],
    split: str,
    num_process: int,
) -> Path:
    """Stage, split, return path to work_out for this version/split."""
    work_in = WORK / version_key / split / "in"
    work_out = WORK / version_key / split / "out"
    if work_out.exists():
        shutil.rmtree(work_out)
    work_out.mkdir(parents=True)

    print(f"[{version_key}/{split}] staging...")
    stage_split_input(ann_dir_name, split, work_in)
    n_img = len(list((work_in / "images").glob("*.png")))
    n_lbl = len(list((work_in / "labelTxt").glob("*.txt")))
    print(f"[{version_key}/{split}] staged images={n_img} labels={n_lbl}")

    print(f"[{version_key}/{split}] splitting subsize={SUBSIZE} gap={GAP}...")
    splitter = splitbase(
        str(work_in),
        str(work_out),
        gap=GAP,
        subsize=SUBSIZE,
        padding=True,
        num_process=num_process,
        ext=".png",
    )
    splitter.splitdata(1)
    n_out = len(list((work_out / "images").glob("*.png")))
    print(f"[{version_key}/{split}] produced {n_out} patches")
    return work_out


def assemble(num_process: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # Process both versions; use v1.5 images as shared tree
    outs: dict[str, dict[str, Path]] = {}
    for version_key, (ann_dir_name, classes) in VERSIONS.items():
        outs[version_key] = {}
        for split in SPLITS:
            outs[version_key][split] = process_version_split(
                version_key, ann_dir_name, classes, split, num_process
            )

    for split in SPLITS:
        v15_imgs = outs["v1.5"][split] / "images"
        v1_imgs = outs["v1"][split] / "images"
        names_v15 = {p.name for p in v15_imgs.glob("*.png")}
        names_v1 = {p.name for p in v1_imgs.glob("*.png")}
        if names_v15 != names_v1:
            only_v15 = len(names_v15 - names_v1)
            only_v1 = len(names_v1 - names_v15)
            raise RuntimeError(
                f"{split}: patch filename mismatch v1 vs v1.5 "
                f"(only_v15={only_v15} only_v1={only_v1})"
            )

        dest_img = OUT / "images" / split
        if dest_img.exists():
            shutil.rmtree(dest_img)
        dest_img.mkdir(parents=True)
        for src in sorted(v15_imgs.glob("*.png")):
            link_or_copy(src, dest_img / src.name)

        for version_key, (ann_dir_name, classes) in VERSIONS.items():
            work_out = outs[version_key][split]
            coco = labeltxt_to_coco(
                work_out / "images",
                work_out / "labelTxt",
                classes,
                f"DOTA {version_key} patched {SUBSIZE} gap{GAP}",
            )
            # Point file_name at shared images; ids already sequential
            out_ann = OUT / ann_dir_name / f"instances_{split}.json"
            out_ann.parent.mkdir(parents=True, exist_ok=True)
            with open(out_ann, "w", encoding="utf-8") as f:
                json.dump(coco, f)
            stats = {
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "classes": len(classes),
            }
            summary.setdefault(version_key, {})[split] = stats
            print(f"[{version_key}/{split}] COCO", stats)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-process", type=int, default=8)
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep staging/split work dirs under datasets/dota/_split_work",
    )
    args = parser.parse_args()

    summary = assemble(args.num_process)
    print(json.dumps(summary, indent=2))

    if not args.keep_work and WORK.exists():
        print(f"Removing work dir {WORK}")
        shutil.rmtree(WORK)
    print(f"Done. Output: {OUT}")


if __name__ == "__main__":
    main()
