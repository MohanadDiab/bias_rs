import json
from pathlib import Path

DATASETS = [
    ("dota", ["annotations_v1", "annotations_v1.5"]),
    ("ai_tod", ["annotations_v1", "annotations_v2"]),
    ("hit_uav", ["annotations"]),
    ("hrsc2016_ms", ["annotations"]),
    ("plant_detection", ["annotations"]),
]

for ds, ann_dirs in DATASETS:
    root = Path("datasets") / ds
    print(f"\n### {ds}")
    img_root = root / "images"
    if img_root.exists():
        splits = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
        print("  image splits:", splits)
        for s in splits:
            n = len(list((img_root / s).glob("*")))
            print(f"    images/{s}: {n} files")
    for ann_dir_name in ann_dirs:
        ann_dir = root / ann_dir_name
        if not ann_dir.exists():
            print(f"  {ann_dir_name}: MISSING")
            continue
        print(f"  [{ann_dir_name}]")
        for ann in sorted(ann_dir.glob("instances_*.json")):
            d = json.load(open(ann, encoding="utf-8"))
            print(f"    {ann.name}: images={len(d['images'])} anns={len(d['annotations'])} cats={len(d['categories'])}")
