"""Clean AI-TOD v2 zero-area boxes and HIT-UAV dontcare class."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def drop_zero_area_boxes(ann_dir: Path) -> None:
    for split in ("train", "val", "test"):
        path = ann_dir / f"instances_{split}.json"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            coco = json.load(f)
        before = len(coco["annotations"])
        kept = []
        dropped = 0
        for ann in coco["annotations"]:
            bb = ann.get("bbox") or []
            if len(bb) != 4 or float(bb[2]) <= 0 or float(bb[3]) <= 0:
                dropped += 1
                continue
            kept.append(ann)
        coco["annotations"] = kept
        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"ai_tod_v2/{split}: {before} -> {len(kept)} (dropped {dropped})")


def remove_dontcare(ann_dir: Path) -> None:
    dontcare_names = {"dontcare", "don't care", "do not care"}
    for split in ("train", "val", "test"):
        path = ann_dir / f"instances_{split}.json"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            coco = json.load(f)

        cats_before = [(c["id"], c["name"]) for c in coco["categories"]]
        dontcare_ids = {
            c["id"] for c in coco["categories"] if c["name"].lower() in dontcare_names
        }
        print(f"hit_uav/{split}: cats before={cats_before} remove_ids={dontcare_ids}")

        before_ann = len(coco["annotations"])
        coco["annotations"] = [
            a for a in coco["annotations"] if a["category_id"] not in dontcare_ids
        ]
        dropped_ann = before_ann - len(coco["annotations"])

        remaining = [
            c
            for c in sorted(coco["categories"], key=lambda c: c["id"])
            if c["id"] not in dontcare_ids
        ]
        old_to_new = {c["id"]: i + 1 for i, c in enumerate(remaining)}
        new_cats = []
        for c in remaining:
            nc = {"id": old_to_new[c["id"]], "name": c["name"]}
            if "supercategory" in c:
                nc["supercategory"] = c["supercategory"]
            new_cats.append(nc)
        coco["categories"] = new_cats
        for ann in coco["annotations"]:
            ann["category_id"] = old_to_new[ann["category_id"]]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f)

        cats_after = [(c["id"], c["name"]) for c in coco["categories"]]
        print(
            f"hit_uav/{split}: anns {before_ann} -> {len(coco['annotations'])} "
            f"(dropped {dropped_ann}); cats={cats_after}"
        )


def main() -> None:
    drop_zero_area_boxes(ROOT / "datasets" / "ai_tod" / "annotations_v2")
    remove_dontcare(ROOT / "datasets" / "hit_uav" / "annotations")


if __name__ == "__main__":
    main()
