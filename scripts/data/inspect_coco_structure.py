import json
from pathlib import Path

datasets = ["dota_v1", "dota_v1.5", "ai_tod", "hit_uav", "hrsc2016_ms", "plant_detection"]
for ds in datasets:
    root = Path("datasets") / ds
    print(f"\n### {ds}")
    img_root = root / "images"
    if img_root.exists():
        splits = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
        print("  image splits:", splits)
        for s in splits:
            n = len(list((img_root / s).glob("*")))
            print(f"    images/{s}: {n} files")
    ann_dir = root / "annotations"
    if ann_dir.exists():
        for ann in sorted(ann_dir.glob("instances_*.json")):
            d = json.load(open(ann, encoding="utf-8"))
            print(f"  {ann.name}: keys={list(d.keys())}")
            print(f"    counts: images={len(d['images'])} anns={len(d['annotations'])} cats={len(d['categories'])}")
            if d["images"]:
                print(f"    image[0]: {d['images'][0]}")
            if d["annotations"]:
                a = d["annotations"][0]
                print(f"    ann[0] keys: {list(a.keys())}")
                seg = a.get("segmentation")
                print(f"    ann[0] segmentation: {type(seg).__name__} len={len(seg) if seg else 0}")
            if d["categories"]:
                print(f"    categories: {[c['name'] for c in d['categories']]}")
