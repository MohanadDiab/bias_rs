"""Stratified train/val split for a single-split COCO dataset."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def image_label(coco: dict) -> dict[int, int]:
    """Map each image id to its stratification label (single dominant class)."""
    labels: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        labels[ann["image_id"]].add(ann["category_id"])
    return {
        image_id: next(iter(cats))
        for image_id, cats in labels.items()
        if len(cats) == 1
    }


def allocate_val_counts(class_sizes: dict[int, int], val_ratio: float) -> dict[int, int]:
    """Proportional val counts per class (largest-remainder), at least one when possible."""
    total = sum(class_sizes.values())
    target = round(total * val_ratio)
    if target <= 0:
        raise ValueError("val_ratio too small for this dataset")

    floors = {c: int(n * target / total) for c, n in class_sizes.items()}
    assigned = sum(floors.values())
    remainders = sorted(
        ((c, (n * target / total) - floors[c]) for c, n in class_sizes.items()),
        key=lambda x: (-x[1], x[0]),
    )
    counts = dict(floors)
    for c, _ in remainders:
        if assigned >= target:
            break
        if counts[c] < class_sizes[c]:
            counts[c] += 1
            assigned += 1

    # Ensure every class with enough images has at least one val sample.
    for c, n in class_sizes.items():
        if n >= 2 and counts[c] == 0:
            donor = max(
                (d for d in counts if counts[d] > 1 and d != c),
                key=lambda d: counts[d],
                default=None,
            )
            if donor is not None:
                counts[donor] -= 1
                counts[c] = 1
    return counts


def stratified_split(
    image_ids: list[int],
    labels: dict[int, int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        by_class[labels[image_id]].append(image_id)

    val_counts = allocate_val_counts({c: len(ids) for c, ids in by_class.items()}, val_ratio)
    rng = random.Random(seed)

    val_ids: list[int] = []
    train_ids: list[int] = []
    for cls in sorted(by_class):
        ids = by_class[cls][:]
        rng.shuffle(ids)
        n_val = min(val_counts[cls], len(ids) - 1) if len(ids) > 1 else 0
        if len(ids) == 1:
            n_val = 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    return train_ids, val_ids


def subset_coco(coco: dict, image_ids: set[int], split_name: str) -> dict:
    selected = sorted(
        [img for img in coco["images"] if img["id"] in image_ids],
        key=lambda x: x["id"],
    )
    id_map = {img["id"]: i + 1 for i, img in enumerate(selected)}
    images = [{**img, "id": id_map[img["id"]]} for img in selected]
    annotations = []
    ann_id = 1
    for ann in coco["annotations"]:
        if ann["image_id"] not in image_ids:
            continue
        annotations.append({**ann, "id": ann_id, "image_id": id_map[ann["image_id"]]})
        ann_id += 1
    return {
        "info": {**coco["info"], "description": f"{coco['info'].get('description', '').split()[0]} {split_name}"},
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def split_dataset(dataset_root: Path, val_ratio: float, seed: int) -> dict:
    ann_path = dataset_root / "annotations" / "instances_train.json"
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    labels = image_label(coco)
    unlabeled = [img["id"] for img in coco["images"] if img["id"] not in labels]
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} images lack a single-class label")

    train_ids, val_ids = stratified_split(list(labels), labels, val_ratio, seed)
    train_set, val_set = set(train_ids), set(val_ids)

    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    file_by_id = {img["id"]: img["file_name"] for img in coco["images"]}

    for image_id in val_ids:
        src = train_dir / file_by_id[image_id]
        dst = val_dir / file_by_id[image_id]
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))

    train_coco = subset_coco(coco, train_set, "train")
    val_coco = subset_coco(coco, val_set, "val")
    (dataset_root / "annotations" / "instances_train.json").write_text(
        json.dumps(train_coco, ensure_ascii=False), encoding="utf-8"
    )
    (dataset_root / "annotations" / "instances_val.json").write_text(
        json.dumps(val_coco, ensure_ascii=False), encoding="utf-8"
    )

    class_names = {c["id"]: c["name"] for c in coco["categories"]}
    per_class = defaultdict(lambda: {"train": 0, "val": 0})
    for image_id in train_ids:
        per_class[labels[image_id]]["train"] += 1
    for image_id in val_ids:
        per_class[labels[image_id]]["val"] += 1

    return {
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "val_ratio": len(val_ids) / len(coco["images"]),
        "per_class": {class_names[c]: v for c, v in sorted(per_class.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root (e.g. datasets/plant_detection)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = split_dataset(args.dataset.resolve(), args.val_ratio, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
