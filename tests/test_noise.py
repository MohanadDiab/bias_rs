"""Synthetic COCO + isolated L/O/C noise protocol tests."""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

from src.data.boxes import box_iou_xywh
from src.data.noise import apply_class_flip, apply_family, apply_localization, apply_objectness
from src.data.splits import stratified_cal_ids
from src.training.data.coco_to_yolo import coco_bbox_to_yolo


def _toy_coco(n_images: int = 20, boxes_per: int = 5) -> dict:
    cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
    images = []
    anns = []
    aid = 1
    for i in range(n_images):
        images.append({"id": i + 1, "file_name": f"{i+1:03d}.jpg", "width": 100, "height": 100})
        for j in range(boxes_per):
            anns.append(
                {
                    "id": aid,
                    "image_id": i + 1,
                    "category_id": (j % 3) + 1,
                    "bbox": [10.0 + j, 10.0 + j, 20.0, 20.0],
                    "area": 400.0,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            aid += 1
    return {"info": {}, "licenses": [], "categories": cats, "images": images, "annotations": anns}


def test_yolo_box_convert_center():
    cx, cy, nw, nh = coco_bbox_to_yolo([25, 25, 50, 50], 100, 100)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6
    assert abs(nw - 0.5) < 1e-6
    assert abs(nh - 0.5) < 1e-6


def test_localization_ratio_and_no_class_flip():
    coco = _toy_coco()
    orig = deepcopy(coco)
    noisy, stats = apply_localization(coco, 20, random.Random(0))
    assert abs(stats["realized_ratio"] - 0.20) < 0.05 or stats["n_selected"] == int(round(0.2 * 100))
    orig_cls = {(int(a["id"]), int(a["category_id"])) for a in orig["annotations"]}
    new_cls = {(int(a["id"]), int(a["category_id"])) for a in noisy["annotations"]}
    assert orig_cls == new_cls
    # at least one box moved
    moved = False
    orig_box = {int(a["id"]): a["bbox"] for a in orig["annotations"]}
    for a in noisy["annotations"]:
        if orig_box[int(a["id"])] != a["bbox"]:
            moved = True
            break
    assert moved


def test_class_flip_does_not_jitter_boxes():
    coco = _toy_coco()
    orig = deepcopy(coco)
    noisy, stats = apply_class_flip(coco, 30, random.Random(1))
    assert stats["n_flipped"] > 0
    orig_box = {int(a["id"]): [float(x) for x in a["bbox"]] for a in orig["annotations"]}
    for a in noisy["annotations"]:
        assert orig_box[int(a["id"])] == [float(x) for x in a["bbox"]]
    flipped = sum(
        int(a["category_id"]) != int(b["category_id"])
        for a, b in zip(orig["annotations"], noisy["annotations"], strict=True)
    )
    assert flipped == stats["n_flipped"]


def test_objectness_omission_and_spurious():
    coco = _toy_coco()
    n = len(coco["annotations"])
    noisy, stats = apply_objectness(coco, 20, random.Random(2))
    assert stats["n_removed"] > 0
    assert stats["n_added"] > 0
    assert len(noisy["annotations"]) == n - stats["n_removed"] + stats["n_added"]


def test_c_family_isolation():
    coco = _toy_coco()
    orig = deepcopy(coco)
    noisy, _ = apply_family(coco, "C", 50, random.Random(3))
    for a, b in zip(orig["annotations"], noisy["annotations"], strict=True):
        assert a["bbox"] == b["bbox"]
        iou = box_iou_xywh(a["bbox"], b["bbox"])
        assert iou == 1.0


def test_stratified_cal_not_empty():
    coco = _toy_coco(n_images=30, boxes_per=3)
    train_ids, cal_ids = stratified_cal_ids(coco, 0.12, 42)
    assert cal_ids
    assert train_ids
    assert set(train_ids).isdisjoint(set(cal_ids))
    assert len(train_ids) + len(cal_ids) == 30


def test_generate_noise_writes_annotations_noise_and_resolve_split_json(tmp_path):
    from src.data.coco import save_coco
    from src.data.paths import noise_train_json, resolve_split_json
    from src.data.prepare import generate_noise

    coco = _toy_coco()
    ann = tmp_path / "annotations"
    save_coco(coco, ann / "instances_train.json")
    save_coco(coco, ann / "instances_val.json")
    save_coco(coco, ann / "instances_test.json")
    runs = generate_noise(tmp_path, "annotations", families=("L",), ratios=(10,))
    out = Path(runs[0]["path"])
    assert out == noise_train_json(tmp_path, "annotations", "L", 10)
    assert out.exists()
    cfg = {
        "dataset": {
            "root": str(tmp_path),
            "ann_dir": "annotations",
            "train_split": "train",
            "val_split": "val",
            "cal_split": "test",
            "noise": {"family": "L", "ratio": 10},
        }
    }
    assert resolve_split_json(cfg, "train") == out
    assert resolve_split_json(cfg, "test") == ann / "instances_test.json"
    assert resolve_split_json(cfg, "val") == ann / "instances_val.json"

