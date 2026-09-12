"""Import every declared runtime dependency and src package."""
from __future__ import annotations

import importlib

import pytest

REQUIRED = [
    "yaml",
    "PIL",
    "numpy",
    "matplotlib",
    "pycocotools",
    "ensemble_boxes",
    "ultralytics",
    "torch",
    "torchvision",
    "cv2",
    "rfdetr",
    "pytorch_lightning",
    "torchmetrics",
    "mim",
    "mmdet",
    "mmengine",
    "mmcv",
]

SRC_PACKAGES = [
    "src.data",
    "src.data.prepare",
    "src.data.noise",
    "src.data.splits",
    "src.data.layout",
    "src.training",
    "src.training.cli",
    "src.training.predict",
    "src.training.registry",
    "src.training.backends.ultralytics_backend",
    "src.training.backends.rfdetr_backend",
    "src.training.backends.mmdet_backend",
    "src.training.backends.tiny_fpn",
    "src.fusion",
    "src.calibration",
    "src.moe",
    "src.eval",
    "src.experiments",
    "src.experiments.runner",
    "src.experiments.matrix",
    "src.experiments.cli",
    "src.experiments.schema",
    "src.experiments.pipeline",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_required_dependency_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("name", SRC_PACKAGES)
def test_src_package_imports(name: str) -> None:
    importlib.import_module(name)


def test_ultralytics_and_torch_are_usable() -> None:
    import torch
    from ultralytics import YOLO

    assert torch.__version__
    assert YOLO is not None


def test_pycocotools_coco_api() -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    assert COCO is not None
    assert COCOeval is not None


def test_rfdetr_api() -> None:
    import rfdetr

    assert getattr(rfdetr, "RFDETRNano", None) is not None


def test_rfdetr_train_extra_imports() -> None:
    import pytorch_lightning
    import rfdetr.training

    assert pytorch_lightning.__version__
    assert rfdetr.training is not None


def test_openmim_api() -> None:
    import mim

    assert mim is not None


def test_mmdet_stack() -> None:
    import mmcv
    import mmengine
    import mmdet

    assert mmcv.__version__
    assert mmengine.__version__
    assert mmdet.__version__


def test_mmdet_trainer_is_registered() -> None:
    from src.training.registry import get_trainer

    trainer = get_trainer("mmdet")
    assert type(trainer).__name__ == "MMDetTrainer"


def test_rfdetr_trainer_is_registered() -> None:
    from src.training.registry import get_trainer

    trainer = get_trainer("rfdetr")
    assert type(trainer).__name__ == "RFDetrTrainer"
