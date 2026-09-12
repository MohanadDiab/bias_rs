"""On-disk COCO layout checks used before noise generation."""
from __future__ import annotations

from src.data.coco import save_coco
from src.data.layout import check_dataset_layout
from src.data.prepare import generate_noisy_trains_for_config
from src.data.paths import noise_train_json


def _coco(file_name: str = "a.jpg") -> dict:
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 0,
                "segmentation": [],
            }
        ],
        "categories": [{"id": 1, "name": "a"}],
    }


def _dataset(tmp_path, *, splits=("train", "val", "test"), file_name="a.jpg"):
    root = tmp_path / "ds"
    for split in splits:
        img = root / "images" / split
        img.mkdir(parents=True)
        (img / file_name).write_bytes(b"x")
        save_coco(_coco(file_name), root / "annotations" / f"instances_{split}.json")
    return root


def test_layout_ok_when_test_is_cal(tmp_path):
    root = _dataset(tmp_path)
    cfg = {
        "name": "toy",
        "root": str(root),
        "ann_dirs": ["annotations"],
        "train_split": "train",
        "val_split": "val",
        "cal_split": "test",
        "carve_cal": False,
    }
    report = check_dataset_layout(cfg)
    assert report["ok"], report["errors"]


def test_layout_fails_without_test_images(tmp_path):
    root = _dataset(tmp_path, splits=("train", "val"))
    cfg = {
        "name": "toy",
        "root": str(root),
        "ann_dirs": ["annotations"],
        "train_split": "train",
        "val_split": "val",
        "cal_split": "test",
        "carve_cal": False,
    }
    report = check_dataset_layout(cfg)
    assert not report["ok"]
    assert any("images/test" in e for e in report["errors"])


def test_layout_rejects_nested_file_name(tmp_path):
    root = _dataset(tmp_path)
    save_coco(_coco("subdir/a.jpg"), root / "annotations" / "instances_train.json")
    cfg = {
        "name": "toy",
        "root": str(root),
        "ann_dirs": ["annotations"],
        "train_split": "train",
        "val_split": "val",
        "cal_split": "test",
        "carve_cal": False,
    }
    report = check_dataset_layout(cfg)
    assert not report["ok"]


def test_prepare_noisy_trains_keeps_clean_val(tmp_path):
    root = _dataset(tmp_path)
    cfg = {
        "name": "toy",
        "root": str(root),
        "ann_dirs": ["annotations"],
        "train_split": "train",
        "val_split": "val",
        "cal_split": "test",
        "carve_cal": False,
        "seed": 42,
    }
    generate_noisy_trains_for_config(cfg, families=("L",), ratios=(10,))
    assert noise_train_json(root, "annotations", "L", 10).exists()
    assert (root / "annotations" / "instances_val.json").exists()
    assert (root / "annotations" / "instances_test.json").exists()
