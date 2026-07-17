"""Strip OBB polygons from COCO JSON; keep axis-aligned bbox only (HBB)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

TARGETS = [
    ("dota", ["annotations_v1", "annotations_v1.5"]),
    ("dota_1024", ["annotations_v1", "annotations_v1.5"]),
    ("ai_tod", ["annotations_v1", "annotations_v2"]),
]


def has_polygon(seg) -> bool:
    return (
        isinstance(seg, list)
        and len(seg) > 0
        and isinstance(seg[0], list)
        and len(seg[0]) >= 6
    )


def convert_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)

    n_cleared = 0
    for ann in coco["annotations"]:
        if has_polygon(ann.get("segmentation")):
            n_cleared += 1
        bbox = ann.get("bbox", [0, 0, 0, 0])
        w = float(bbox[2]) if len(bbox) >= 4 else 0.0
        h = float(bbox[3]) if len(bbox) >= 4 else 0.0
        ann["segmentation"] = []
        ann["area"] = w * h

    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    return {
        "path": str(path),
        "annotations": len(coco["annotations"]),
        "cleared_polygons": n_cleared,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing",
    )
    args = parser.parse_args()

    results = []
    for dataset, ann_dirs in TARGETS:
        root = DATASETS / dataset
        for ann_dir_name in ann_dirs:
            ann_dir = root / ann_dir_name
            if not ann_dir.exists():
                print(f"SKIP missing {ann_dir}")
                continue
            for path in sorted(ann_dir.glob("instances_*.json")):
                if args.dry_run:
                    with open(path, encoding="utf-8") as f:
                        coco = json.load(f)
                    n = sum(1 for a in coco["annotations"] if has_polygon(a.get("segmentation")))
                    print(f"DRY {path}: {n}/{len(coco['annotations'])} polygons")
                    continue
                stats = convert_file(path)
                results.append(stats)
                print(
                    f"OK {path.relative_to(ROOT)}: "
                    f"cleared={stats['cleared_polygons']}/{stats['annotations']}"
                )

    if not args.dry_run:
        print(json.dumps({"files": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
