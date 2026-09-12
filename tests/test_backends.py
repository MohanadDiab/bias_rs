"""Trainer backends: prepare layout, RF-DETR train extras, checkpoint lookup."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.data.coco import save_coco
from src.data.schema import load_detections
from src.training.backends.mmdet_backend import architecture_from_cfg
from src.training.backends.rfdetr_backend import (
    RFDetrTrainer,
    _model_key,
    _unpack_rfdetr,
    require_rfdetr_train,
    rfdetr_train_kwargs,
)
from src.training.backends.ultralytics_backend import UltralyticsTrainer
from src.training.config import ROOT
from src.training.registry import get_trainer
from src.training.weights import find_run_checkpoint, resolve_weights


def _coco(file_name: str = "a.png") -> dict:
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


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)


def _dataset(tmp_path: Path, splits=("train", "val", "test")) -> Path:
    root = tmp_path / "ds"
    for split in splits:
        _write_png(root / "images" / split / "a.png")
        save_coco(_coco(), root / "annotations" / f"instances_{split}.json")
    return root


def _cfg(tmp_path: Path, *, backend: str = "rfdetr", name: str = "hit_uav__rfdetr_nano__clean") -> dict:
    root = _dataset(tmp_path)
    out = tmp_path / "out"
    return {
        "name": name,
        "backend": backend,
        "expert_id": "rfdetr_nano",
        "dataset": {
            "root": str(root),
            "ann_dir": "annotations",
            "train_split": "train",
            "val_split": "val",
            "cal_split": "test",
            "noise": {},
        },
        "model": {"name": "nano"},
        "train": {
            "imgsz": 640,
            "batch": 8,
            "epochs": 2,
            "project": str(out / "training"),
        },
        "predict": {"conf": 0.3},
        "output": {"root": str(out)},
    }


def test_pyproject_requires_rfdetr_train_extra() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "rfdetr[train]" in text


def test_require_rfdetr_train_imports() -> None:
    require_rfdetr_train()


@pytest.mark.parametrize(
    ("backend", "cls_name"),
    [
        ("ultralytics", "UltralyticsTrainer"),
        ("rfdetr", "RFDetrTrainer"),
        ("mmdet", "MMDetTrainer"),
        ("tiny_fpn", "TinyFPNTrainer"),
    ],
)
def test_registry_backends(backend: str, cls_name: str) -> None:
    assert type(get_trainer(backend)).__name__ == cls_name


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("nano", "nano"),
        ("rfdetr-nano", "nano"),
        ("RFDETR_small", "small"),
        ("yolo26n.pt", "nano"),
        ("unknown", "nano"),
    ],
)
def test_rfdetr_model_key(name: str, key: str) -> None:
    assert _model_key({"model": {"name": name}}) == key


def test_unpack_rfdetr_supervision_style() -> None:
    raw = SimpleNamespace(xyxy=[[0.0, 0.0, 4.0, 4.0]], confidence=[0.9], class_id=[0])
    xyxy, scores, classes = _unpack_rfdetr(raw)
    assert xyxy == [[0.0, 0.0, 4.0, 4.0]]
    assert scores == [0.9]
    assert classes == [0]


def test_unpack_rfdetr_dict_and_empty() -> None:
    assert _unpack_rfdetr(None) == ([], [], [])
    xyxy, scores, classes = _unpack_rfdetr({"xyxy": [[1, 2, 3, 4]], "scores": [0.2], "labels": [1]})
    assert xyxy == [[1, 2, 3, 4]]
    assert scores == [0.2]
    assert classes == [1]


def test_rfdetr_train_kwargs_disable_tensorboard(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    kwargs = rfdetr_train_kwargs(cfg, tmp_path / "data", tmp_path / "run")
    assert kwargs["dataset_dir"] == str(tmp_path / "data")
    assert kwargs["epochs"] == 2
    assert kwargs["batch_size"] == 8
    assert kwargs["tensorboard"] is False
    assert kwargs["resolution"] == 640
    assert kwargs["device"] in {"cpu", "cuda"}


def test_rfdetr_prepare_writes_roboflow_layout(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cache = RFDetrTrainer().prepare(cfg)
    assert (cache / "train" / "_annotations.coco.json").exists()
    assert (cache / "valid" / "_annotations.coco.json").exists()
    assert (cache / "train" / "a.png").exists()
    assert (cache / "valid" / "a.png").exists()


def test_ultralytics_prepare_writes_yolo_layout(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, backend="ultralytics", name="hit_uav__yolo26n__clean")
    cfg["model"]["name"] = "yolo26n.pt"
    data_yaml = UltralyticsTrainer().prepare(cfg)
    assert data_yaml.name == "data.yaml"
    labels = data_yaml.parent / "labels" / "train" / "a.txt"
    assert labels.exists()
    assert labels.read_text(encoding="utf-8").startswith("0 ")


def test_rfdetr_train_uses_kwargs_and_writes_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeModel:
        def train(self, **kwargs):
            captured.update(kwargs)
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(kwargs["output_dir"]) / "checkpoint_best_total.pth").write_bytes(b"w")

    import src.training.backends.rfdetr_backend as rb

    monkeypatch.setattr(rb, "require_rfdetr_train", lambda: None)
    monkeypatch.setattr(rb, "_load_rfdetr_class", lambda key: FakeModel)
    cfg = _cfg(tmp_path)
    out = RFDetrTrainer().train(cfg, tmp_path / "data")
    assert captured["tensorboard"] is False
    assert captured["dataset_dir"] == str(tmp_path / "data")
    assert (out / "run_meta.json").exists()
    assert (out / "checkpoint_best_total.pth").exists()


def test_rfdetr_predict_writes_detections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def predict(self, image, threshold=0.3):
            return SimpleNamespace(xyxy=[[0.0, 0.0, 2.0, 2.0]], confidence=[0.88], class_id=[0])

    import src.training.backends.rfdetr_backend as rb

    monkeypatch.setattr(rb, "_load_rfdetr_class", lambda key: FakeModel)
    cfg = _cfg(tmp_path)
    path = RFDetrTrainer().predict(cfg, tmp_path / "data", "val")
    dets = load_detections(path)
    assert len(dets) == 1
    assert dets[0].expert_id == "rfdetr_nano"
    assert dets[0].score == pytest.approx(0.88)
    assert dets[0].bbox == [0.0, 0.0, 2.0, 2.0]


def test_find_run_checkpoint_prefers_rfdetr_best_total(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run = Path(cfg["train"]["project"]) / cfg["name"]
    run.mkdir(parents=True)
    (run / "checkpoint.pth").write_bytes(b"old")
    (run / "checkpoint_best_total.pth").write_bytes(b"best")
    found = find_run_checkpoint(cfg)
    assert found is not None
    assert found.name == "checkpoint_best_total.pth"


def test_resolve_weights_ignores_missing_explicit_best_pt(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run = Path(cfg["train"]["project"]) / cfg["name"]
    run.mkdir(parents=True)
    (run / "checkpoint_best_total.pth").write_bytes(b"best")
    cfg.setdefault("model", {})["weights"] = str(run / "weights" / "best.pt")
    assert Path(resolve_weights(cfg)).name == "checkpoint_best_total.pth"


@pytest.mark.parametrize(
    ("name", "arch"),
    [("dino", "dino"), ("rtmdet", "rtmdet"), ("rtmdet-ins", "rtmdet")],
)
def test_mmdet_architecture_from_cfg(name: str, arch: str) -> None:
    assert architecture_from_cfg({"model": {"name": name}}) == arch


def test_train_predict_skips_existing_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"train": 0}

    class Dummy:
        def prepare(self, cfg):
            path = tmp_path / "prepared"
            path.mkdir(exist_ok=True)
            return path

        def train(self, cfg, data_path):
            calls["train"] += 1
            out = Path(cfg["train"]["project"]) / cfg["name"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoint_best_total.pth").write_bytes(b"w")
            return out

        def predict(self, cfg, data_path, split):
            return Path(cfg["train"]["project"]) / cfg["name"] / f"predictions_{split}.json"

    monkeypatch.setattr("src.experiments.runner.get_trainer", lambda backend: Dummy())
    from src.experiments.runner import _train_predict

    cfg = _cfg(tmp_path)
    run = Path(cfg["train"]["project"]) / cfg["name"]
    run.mkdir(parents=True)
    (run / "checkpoint_best_total.pth").write_bytes(b"w")
    _train_predict(cfg, do_train=True, do_predict=False)
    assert calls["train"] == 0
    assert Path(cfg["model"]["weights"]).name == "checkpoint_best_total.pth"


def test_train_predict_trains_when_no_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"train": 0}

    class Dummy:
        def prepare(self, cfg):
            path = tmp_path / "prepared"
            path.mkdir(exist_ok=True)
            return path

        def train(self, cfg, data_path):
            calls["train"] += 1
            out = Path(cfg["train"]["project"]) / cfg["name"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoint_best_total.pth").write_bytes(b"w")
            return out

        def predict(self, cfg, data_path, split):
            return Path(cfg["train"]["project"]) / cfg["name"] / f"predictions_{split}.json"

    monkeypatch.setattr("src.experiments.runner.get_trainer", lambda backend: Dummy())
    from src.experiments.runner import _train_predict

    _train_predict(_cfg(tmp_path), do_train=True, do_predict=False)
    assert calls["train"] == 1
