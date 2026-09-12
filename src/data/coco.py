"""COCO JSON helpers."""
from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_coco(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f)


def anns_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def image_size_map(coco: dict[str, Any]) -> dict[int, tuple[int, int]]:
    return {int(im["id"]): (int(im["width"]), int(im["height"])) for im in coco["images"]}


def file_name_map(coco: dict[str, Any]) -> dict[int, str]:
    return {int(im["id"]): Path(im["file_name"]).name for im in coco["images"]}


def category_names(coco: dict[str, Any]) -> dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}


def subset_coco(coco: dict[str, Any], image_ids: set[int], description: str | None = None) -> dict[str, Any]:
    selected = sorted(
        [img for img in coco["images"] if int(img["id"]) in image_ids],
        key=lambda x: int(x["id"]),
    )
    keep = {int(img["id"]) for img in selected}
    annotations = [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in keep]
    info = deepcopy(coco.get("info", {}))
    if description:
        info["description"] = description
    return {
        "info": info,
        "licenses": deepcopy(coco.get("licenses", [])),
        "categories": deepcopy(coco.get("categories", [])),
        "images": deepcopy(selected),
        "annotations": deepcopy(annotations),
    }


def reindex_annotations(coco: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(coco)
    for i, ann in enumerate(out["annotations"], start=1):
        ann["id"] = i
    return out


def dominant_class(anns: list[dict[str, Any]]) -> int | None:
    if not anns:
        return None
    counts: dict[int, int] = defaultdict(int)
    for ann in anns:
        counts[int(ann["category_id"])] += 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
