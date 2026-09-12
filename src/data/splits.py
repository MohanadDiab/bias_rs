"""Carve a clean calibration split from original train images."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.coco import anns_by_image, dominant_class, load_coco, save_coco, subset_coco
from src.data.files import link_or_copy

DEFAULT_CAL_RATIO = 0.12
DEFAULT_SEED = 42
TRAIN_BACKUP = "instances_train_precalsplit.json"


def _stratify_labels(coco: dict[str, Any]) -> dict[int, int]:
    grouped = anns_by_image(coco)
    labels: dict[int, int] = {}
    for im in coco["images"]:
        image_id = int(im["id"])
        dom = dominant_class(grouped.get(image_id, []))
        labels[image_id] = -1 if dom is None else int(dom)
    return labels


def allocate_counts(class_sizes: dict[int, int], ratio: float) -> dict[int, int]:
    total = sum(class_sizes.values())
    target = max(1, round(total * ratio)) if total else 0
    floors = {c: int(n * target / total) if total else 0 for c, n in class_sizes.items()}
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


def stratified_cal_ids(
    coco: dict[str, Any],
    cal_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    labels = _stratify_labels(coco)
    by_class: dict[int, list[int]] = defaultdict(list)
    for image_id, cls in labels.items():
        by_class[cls].append(image_id)
    val_counts = allocate_counts({c: len(ids) for c, ids in by_class.items()}, cal_ratio)
    rng = random.Random(seed)
    cal_ids: list[int] = []
    train_ids: list[int] = []
    for cls in sorted(by_class):
        ids = by_class[cls][:]
        rng.shuffle(ids)
        n_cal = val_counts.get(cls, 0)
        if len(ids) <= 1:
            n_cal = 0
        else:
            n_cal = min(n_cal, len(ids) - 1)
        cal_ids.extend(ids[:n_cal])
        train_ids.extend(ids[n_cal:])
    return train_ids, cal_ids


def _image_src(root: Path, split: str, file_name: str) -> Path:
    return root / "images" / split / Path(file_name).name


def carve_cal_split(
    root: str | Path,
    ann_dir: str,
    *,
    train_split: str = "train",
    cal_split: str = "cal",
    cal_ratio: float = DEFAULT_CAL_RATIO,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    """Move a stratified subset of train images into a disjoint clean cal split.

    The original train JSON is backed up once as ``instances_train_precalsplit.json``.
    Cal labels stay clean (oracle-cal). Val is never touched.
    """
    root = Path(root)
    ann_path = root / ann_dir
    cal_json = ann_path / f"instances_{cal_split}.json"
    train_json = ann_path / f"instances_{train_split}.json"
    backup_json = ann_path / TRAIN_BACKUP

    if cal_json.exists() and not force:
        cal = load_coco(cal_json)
        train = load_coco(train_json)
        return {
            "skipped": True,
            "ann_dir": ann_dir,
            "train_images": len(train["images"]),
            "cal_images": len(cal["images"]),
            "cal_json": str(cal_json),
        }

    if not train_json.exists():
        raise FileNotFoundError(train_json)

    if not backup_json.exists():
        save_coco(load_coco(train_json), backup_json)

    source = load_coco(backup_json)
    train_ids, cal_ids = stratified_cal_ids(source, cal_ratio, seed)
    train_set, cal_set = set(train_ids), set(cal_ids)

    train_coco = subset_coco(source, train_set, description=f"{ann_dir} {train_split} (post cal carve)")
    cal_coco = subset_coco(source, cal_set, description=f"{ann_dir} {cal_split} (oracle-cal, clean)")

    cal_img_dir = root / "images" / cal_split
    cal_img_dir.mkdir(parents=True, exist_ok=True)
    for im in cal_coco["images"]:
        name = Path(im["file_name"]).name
        src = _image_src(root, train_split, name)
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, cal_img_dir / name)

    save_coco(train_coco, train_json)
    save_coco(cal_coco, cal_json)
    return {
        "skipped": False,
        "ann_dir": ann_dir,
        "train_images": len(train_coco["images"]),
        "cal_images": len(cal_coco["images"]),
        "train_boxes": len(train_coco["annotations"]),
        "cal_boxes": len(cal_coco["annotations"]),
        "cal_ratio_realized": len(cal_coco["images"]) / max(len(source["images"]), 1),
        "seed": seed,
        "oracle_cal": True,
    }


def carve_shared_images(
    root: str | Path,
    ann_dirs: list[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Carve cal using the first ann_dir's train images, then apply the same filenames to the rest."""
    root = Path(root)
    if not ann_dirs:
        return []
    primary = carve_cal_split(root, ann_dirs[0], **kwargs)
    results = [primary]
    if len(ann_dirs) == 1:
        return results

    cal_split = kwargs.get("cal_split", "cal")
    train_split = kwargs.get("train_split", "train")
    force = kwargs.get("force", False)
    cal_names = {
        Path(im["file_name"]).name
        for im in load_coco(root / ann_dirs[0] / f"instances_{cal_split}.json")["images"]
    }
    for ann_dir in ann_dirs[1:]:
        train_json = root / ann_dir / f"instances_{train_split}.json"
        backup = root / ann_dir / TRAIN_BACKUP
        cal_json = root / ann_dir / f"instances_{cal_split}.json"
        if cal_json.exists() and not force:
            results.append({"skipped": True, "ann_dir": ann_dir})
            continue
        if not backup.exists():
            save_coco(load_coco(train_json), backup)
        source = load_coco(backup)
        cal_ids = {int(im["id"]) for im in source["images"] if Path(im["file_name"]).name in cal_names}
        train_ids = {int(im["id"]) for im in source["images"]} - cal_ids
        save_coco(subset_coco(source, train_ids, f"{ann_dir} {train_split}"), train_json)
        cal_coco = subset_coco(source, cal_ids, f"{ann_dir} {cal_split}")
        save_coco(cal_coco, cal_json)
        cal_img_dir = root / "images" / cal_split
        for im in cal_coco["images"]:
            name = Path(im["file_name"]).name
            src = root / "images" / train_split / name
            if src.exists():
                link_or_copy(src, cal_img_dir / name)
        results.append(
            {
                "skipped": False,
                "ann_dir": ann_dir,
                "train_images": len(train_ids),
                "cal_images": len(cal_ids),
            }
        )
    return results
