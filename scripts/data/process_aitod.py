"""Export AI-TOD v2 COCO annotations filtered to on-disk images."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "datasets" / "ai_tod"
SPLITS = ("train", "val", "test")


def disk_filenames(split: str) -> set[str]:
    return {p.name for p in (AI / "images" / split).glob("*.png")}


def normalize_categories(categories: list[dict]) -> list[dict]:
    return [
        {"id": c["id"] + 1, "name": c["name"], "supercategory": c.get("supercategory", c["name"])}
        for c in categories
    ]


def filter_coco_to_disk(data: dict, available: set[str], description: str) -> dict:
    keep_images: list[dict] = []
    old_to_new: dict[int, int] = {}
    for im in sorted(data["images"], key=lambda x: x["id"]):
        name = Path(im["file_name"]).name
        if name not in available:
            continue
        new_id = len(keep_images) + 1
        old_to_new[im["id"]] = new_id
        keep_images.append({**im, "id": new_id, "file_name": name})

    keep_ids = set(old_to_new)
    keep_anns = []
    for ann in data["annotations"]:
        if ann["image_id"] not in keep_ids:
            continue
        keep_anns.append({
            **ann,
            "image_id": old_to_new[ann["image_id"]],
            "category_id": ann["category_id"] + 1,
        })
    for i, ann in enumerate(keep_anns, start=1):
        ann["id"] = i

    return {
        "info": {**data.get("info", {}), "description": description},
        "licenses": data.get("licenses", []),
        "categories": normalize_categories(data["categories"]),
        "images": keep_images,
        "annotations": keep_anns,
    }


def export_v2_split(split: str) -> dict:
    src = AI / "annotations_source_v2" / f"aitodv2_{split}.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    desc = "AI-TOD-v2 wo_xview subset"
    out = filter_coco_to_disk(data, disk_filenames(split), desc)
    out_path = AI / "annotations_v2" / f"instances_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return {"kept_images": len(out["images"]), "kept_annotations": len(out["annotations"])}


def export_v2() -> dict:
    summary = {}
    for split in SPLITS:
        stats = export_v2_split(split)
        summary[split] = stats
        print(split, stats)
    return summary


if __name__ == "__main__":
    export_v2()
