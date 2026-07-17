"""Audit training YAMLs against on-disk COCO datasets."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from src.training.config import ROOT, load_config


def main() -> None:
    configs = sorted((ROOT / "configs/training").glob("*.yaml"))
    configs = [p for p in configs if p.name != "defaults.yaml"]

    print("=" * 80)
    print(f"Auditing {len(configs)} training configs")
    print("=" * 80)

    issues: list[tuple[str, str]] = []
    ok_count = 0

    for cfg_path in configs:
        print(f"\n### {cfg_path.name}")
        try:
            cfg = load_config(cfg_path)
        except Exception as e:
            issues.append((cfg_path.name, f"load_config failed: {e}"))
            print(f"  FAIL load: {e}")
            continue

        ds = cfg["dataset"]
        root = Path(ds["root"])
        ann_dir = root / ds["ann_dir"]
        train_split, val_split = ds["train_split"], ds["val_split"]
        imgsz = cfg["train"]["imgsz"]
        batch = cfg["train"]["batch"]
        name = cfg["name"]
        model = cfg["model"]["name"]
        task = cfg["model"].get("task")

        print(f"  run={name} model={model} task={task} imgsz={imgsz} batch={batch}")
        print(f"  root={root}")
        print(f"  ann_dir={ann_dir}")

        if not root.exists():
            issues.append((cfg_path.name, f"dataset root missing: {root}"))
            print("  FAIL missing root")
            continue
        if not ann_dir.exists():
            issues.append((cfg_path.name, f"ann_dir missing: {ann_dir}"))
            print("  FAIL missing ann_dir")
            continue

        names_by_split: dict[str, list[str]] = {}
        for split in (train_split, val_split):
            coco_path = ann_dir / f"instances_{split}.json"
            img_dir = root / "images" / split
            if not coco_path.exists():
                issues.append((cfg_path.name, f"missing {coco_path}"))
                print(f"  FAIL missing {coco_path.name}")
                continue
            if not img_dir.exists():
                issues.append((cfg_path.name, f"missing images/{split}"))
                print(f"  FAIL missing images/{split}")
                continue

            with open(coco_path, encoding="utf-8") as f:
                coco = json.load(f)
            cats = sorted(coco["categories"], key=lambda c: c["id"])
            names = [c["name"] for c in cats]
            names_by_split[split] = names
            disk_files = {p.name for p in img_dir.iterdir() if p.is_file()}
            coco_files = {Path(im["file_name"]).name for im in coco["images"]}
            missing_on_disk = coco_files - disk_files
            orphan_on_disk = disk_files - coco_files

            n_poly = 0
            n_bad_bbox = 0
            n_oob = 0
            by_id = {im["id"]: im for im in coco["images"]}
            for ann in coco["annotations"]:
                seg = ann.get("segmentation")
                if (
                    isinstance(seg, list)
                    and seg
                    and isinstance(seg[0], list)
                    and len(seg[0]) >= 6
                ):
                    n_poly += 1
                bb = ann.get("bbox")
                if not bb or len(bb) != 4 or bb[2] <= 0 or bb[3] <= 0:
                    n_bad_bbox += 1
                    continue
                im = by_id.get(ann["image_id"])
                if im:
                    x, y, w, h = bb
                    if (
                        x < -1
                        or y < -1
                        or x + w > im["width"] + 1
                        or y + h > im["height"] + 1
                    ):
                        n_oob += 1

            sizes = []
            for im in coco["images"][:20]:
                p = img_dir / Path(im["file_name"]).name
                if p.exists():
                    with Image.open(p) as img:
                        sizes.append(img.size)
            size_set = sorted(set(sizes))
            meta_wh = Counter((im["width"], im["height"]) for im in coco["images"])
            top_meta = meta_wh.most_common(3)

            cat_ids = [c["id"] for c in cats]
            contiguous = cat_ids == list(range(1, len(cats) + 1)) or cat_ids == list(
                range(len(cats))
            )

            n_img = len(coco["images"])
            n_ann = len(coco["annotations"])
            print(
                f"  {split}: images={n_img} anns={n_ann} classes={len(cats)}"
            )
            print(f"         names={names}")
            print(
                f"         missing_on_disk={len(missing_on_disk)} "
                f"orphan_disk={len(orphan_on_disk)}"
            )
            print(f"         polygons={n_poly} bad_bbox={n_bad_bbox} oob~={n_oob}")
            print(f"         meta_size_top={top_meta} sampled_disk_sizes={size_set}")
            print(f"         cat_ids={cat_ids} contiguousish={contiguous}")

            if missing_on_disk:
                sample = list(missing_on_disk)[:3]
                issues.append(
                    (
                        cfg_path.name,
                        f"{split}: {len(missing_on_disk)} COCO images missing "
                        f"on disk e.g. {sample}",
                    )
                )
            if n_poly:
                issues.append(
                    (
                        cfg_path.name,
                        f"{split}: {n_poly} polygons remain (expected HBB-only)",
                    )
                )
            if n_bad_bbox:
                issues.append(
                    (cfg_path.name, f"{split}: {n_bad_bbox} bad bboxes")
                )

            if len(meta_wh) == 1:
                (mw, mh), _ = top_meta[0]
                if max(mw, mh) != imgsz and abs(max(mw, mh) - imgsz) > 50:
                    msg = (
                        f"{split}: imgsz={imgsz} vs native {mw}x{mh} "
                        "(may be intentional resize)"
                    )
                    issues.append((cfg_path.name, msg))
                    print(f"         WARN imgsz mismatch: config={imgsz} native={mw}x{mh}")

        if train_split in names_by_split and val_split in names_by_split:
            if names_by_split[train_split] != names_by_split[val_split]:
                issues.append(
                    (
                        cfg_path.name,
                        "train/val class name mismatch: "
                        f"{names_by_split[train_split]} vs "
                        f"{names_by_split[val_split]}",
                    )
                )
                print("  FAIL train/val class mismatch")
            else:
                print("  OK train/val class names match")

        for extra in sorted(root.iterdir()):
            if extra.is_dir() and extra.name.startswith("annotations"):
                print(f"  ann_folder_present: {extra.name}")

        ok_count += 1

    print("\n" + "=" * 80)
    print(f"Configs loaded OK: {ok_count}/{len(configs)}")
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for name, msg in issues:
            print(f"  - [{name}] {msg}")
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
